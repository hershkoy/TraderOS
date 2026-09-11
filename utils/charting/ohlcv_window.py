"""Bar-count OHLCV windows for the Charts page (not calendar spans)."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
ProgressFn = Optional[Callable[[int, str], None]]

DEFAULT_BEFORE = 50
DEFAULT_AFTER = 50
MAX_PAD = 1000
TF_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 390,
    "1w": 1950,
    "1M": 7800,
}
TF_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]


def clamp_pad(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(0, min(n, MAX_PAD))


def slice_bar_window(n: int, center: int, before: int, after: int) -> Tuple[int, int]:
    """Inclusive center, exclusive end. Clamped to [0, n)."""
    if n <= 0:
        return 0, 0
    c = min(max(int(center), 0), n - 1)
    start = max(0, c - max(0, int(before)))
    end = min(n, c + max(0, int(after)) + 1)
    if start >= end:
        return c, min(n, c + 1)
    return start, end


def center_index(index: pd.DatetimeIndex, around: Optional[pd.Timestamp]) -> int:
    """Nearest bar at or before `around`; last bar if around is None."""
    n = len(index)
    if n <= 0:
        return 0
    if around is None or pd.isna(around):
        return n - 1
    ts = pd.Timestamp(around)
    idx_tz = getattr(index, "tz", None)
    if ts.tzinfo is None:
        if idx_tz is not None:
            ts = ts.tz_localize("UTC")
    else:
        if idx_tz is None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        else:
            ts = ts.tz_convert(idx_tz)
    pos = int(index.searchsorted(ts, side="right") - 1)
    return min(max(pos, 0), n - 1)


def parse_around_ts(text: str) -> Optional[pd.Timestamp]:
    raw = (text or "").strip()
    if not raw:
        return None
    t = pd.to_datetime(raw, utc=False, errors="coerce")
    if t is None or pd.isna(t):
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    if t is None or pd.isna(t):
        return None
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        if len(raw) <= 10:
            ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
        if len(raw) <= 10:
            ts = ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts


def _to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy()
    if "timestamp" in out.columns:
        out = out.set_index("timestamp")
    elif "ts" in out.columns:
        out = out.set_index("ts")
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif col == "volume":
            out[col] = 0.0
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep].sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC")
    return out


def _source_timeframe(target: str, available: list) -> Optional[str]:
    have = {str(t) for t in (available or [])}
    if target in have:
        return target
    if target not in TF_ORDER:
        return None
    smaller = [t for t in TF_ORDER[: TF_ORDER.index(target)] if t in have]
    return smaller[-1] if smaller else None


def candidate_source_timeframes(target: str) -> list:
    """Exact TF first, then smaller natives (no DISTINCT scan of market_data)."""
    tf = str(target or "").strip()
    if not tf:
        return []
    if tf not in TF_ORDER:
        return [tf]
    i = TF_ORDER.index(tf)
    return [tf] + list(reversed(TF_ORDER[:i]))


def _notify(progress: ProgressFn, pct: int, msg: str) -> None:
    if progress is None:
        return
    try:
        progress(int(pct), str(msg))
    except Exception:
        logger.debug("chart progress callback failed", exc_info=True)


def _scale_pads(src: str, dst: str, before: int, after: int) -> Tuple[int, int]:
    sm = TF_MINUTES.get(src, 1)
    dm = TF_MINUTES.get(dst, sm)
    ratio = max(1, int(round(float(dm) / float(sm))))
    return (
        min(MAX_PAD, before * ratio),
        min(MAX_PAD, after * ratio),
    )


def _has_more(df: pd.DataFrame, around_ts: Optional[pd.Timestamp], before_n: int, after_n: int) -> Tuple[bool, bool]:
    if df is None or df.empty:
        return False, False
    if around_ts is not None:
        n_left = int((df.index <= around_ts).sum())
        n_right = int((df.index > around_ts).sum())
        return n_left >= before_n + 1, (n_right >= after_n and after_n > 0)
    return len(df) >= max(1, before_n + after_n), False


def _fetch_window_df(client, symbol: str, timeframe: str, around_ts, before_n: int, after_n: int) -> pd.DataFrame:
    raw = client.get_market_data_window(
        symbol,
        timeframe,
        around=None if around_ts is None else around_ts.to_pydatetime(),
        before=before_n,
        after=after_n,
    )
    return _to_ohlcv(raw)


def load_ohlcv_window(
    symbol: str,
    timeframe: str,
    *,
    around: Optional[str] = None,
    before: int = DEFAULT_BEFORE,
    after: int = DEFAULT_AFTER,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    """TimescaleDB bar window. Never loads full history (that can take minutes)."""
    empty = {
        "df": pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
        "has_more_before": False,
        "has_more_after": False,
    }
    before_n = clamp_pad(before, DEFAULT_BEFORE)
    after_n = clamp_pad(after, DEFAULT_AFTER)
    around_ts = parse_around_ts(around or "")
    df = None
    has_more_before = False
    has_more_after = False
    _notify(progress, 12, "Connecting to TimescaleDB")
    try:
        from utils.db.timescaledb_client import get_timescaledb_client

        client = get_timescaledb_client()
        sources = candidate_source_timeframes(timeframe)
        n_src = max(len(sources), 1)
        for i, src in enumerate(sources):
            pct = 18 + int(40 * i / n_src)
            if src == timeframe:
                _notify(progress, pct, "Fetching %s %s window" % (symbol, src))
                src_before, src_after = before_n, after_n
            else:
                _notify(progress, pct, "No %s bars; trying %s" % (timeframe, src))
                src_before, src_after = _scale_pads(src, timeframe, before_n, after_n)
            try:
                src_df = _fetch_window_df(
                    client, symbol, src, around_ts, src_before, src_after
                )
            except Exception as exc:
                logger.warning("window query failed for %s %s: %s", symbol, src, exc)
                continue
            if src_df is None or src_df.empty:
                continue
            if src == timeframe:
                df = src_df
                has_more_before, has_more_after = _has_more(
                    df, around_ts, before_n, after_n
                )
                break
            _notify(
                progress,
                min(pct + 8, 70),
                "Aggregating %s to %s" % (src, timeframe),
            )
            try:
                from utils.data.data_aggregator import DataAggregator
            except ImportError:
                from utils.data_aggregator import DataAggregator

            agg = DataAggregator.aggregate_data(src_df, timeframe)
            if agg is None or agg.empty:
                continue
            idx = pd.DatetimeIndex(agg.index)
            center = center_index(idx, around_ts)
            start, end = slice_bar_window(len(idx), center, before_n, after_n)
            df = agg.iloc[start:end].copy()
            has_more_before = start > 0 or bool(
                _has_more(src_df, around_ts, src_before, src_after)[0]
            )
            has_more_after = end < len(idx) or bool(
                _has_more(src_df, around_ts, src_before, src_after)[1]
            )
            break
        if df is not None and not df.empty and not has_more_before and not has_more_after:
            has_more_before, has_more_after = _has_more(
                df, around_ts, before_n, after_n
            )
    except Exception as exc:
        logger.warning("load_ohlcv_window failed for %s %s: %s", symbol, timeframe, exc)
        df = None
        has_more_before = False
        has_more_after = False

    if df is None or df.empty:
        _notify(progress, 80, "No bars in window")
        return empty

    _notify(progress, 75, "Loaded %s bars" % len(df))
    return {
        "df": df,
        "has_more_before": bool(has_more_before),
        "has_more_after": bool(has_more_after),
    }
