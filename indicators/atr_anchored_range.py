"""ATR Anchored Range session overlay (TradeSeekers AAA %b style).

At each new ATR-timeframe session, mid is the session open (or prior close)
and half-range is ATR/2. Bands stay flat until the next session.

On an intraday chart with a higher ATR timeframe, ATR is the last completed
HTF bar (causal). On the same timeframe, ATR includes the current bar, matching
Pine `ta.atr`.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

SESSION_TZ = "America/New_York"
MODE_OPEN = "open"
MODE_PRIOR_CLOSE = "prior_close"
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
ATR_TF_ALIASES = {
    "1D": "1d",
    "D": "1d",
    "1W": "1w",
    "W": "1w",
    "1M": "1M",
    "M": "1M",
}


def normalize_mode(mode: Any) -> str:
    text = str(mode or "").strip().lower().replace("-", " ").replace("_", " ")
    if text in ("prior close", "priorclose", "po", "close"):
        return MODE_PRIOR_CLOSE
    return MODE_OPEN


def normalize_atr_tf(timeframe: Any) -> str:
    raw = str(timeframe or "1d").strip()
    if not raw:
        return "1d"
    alias = ATR_TF_ALIASES.get(raw.upper().replace(" ", ""))
    if alias:
        return alias
    if raw in TF_MINUTES:
        return raw
    return "1d"


def is_intraday(timeframe: Any) -> bool:
    tf = str(timeframe or "").strip()
    return TF_MINUTES.get(tf, TF_MINUTES["1d"]) < TF_MINUTES["1d"]


def _as_utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _utc_naive(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def htf_unique_bars_needed(
    first: Any, last: Any, period: int, atr_tf: Any = "1d"
) -> int:
    """Unique HTF bars to fetch: ATR warmup plus the loaded chart span.

    A fixed 60-bar lookback from the last candle is not enough after zoom-out
    preload (2000 15m bars is ~3 months; ATR(20) warmup sits on top of that).
    """
    period_n = max(1, int(period))
    warmup = period_n + 5
    span_days = 0
    if first is not None and last is not None:
        a = _as_utc(first)
        b = _as_utc(last)
        if b < a:
            a, b = b, a
        span_days = max(0, int((b - a).days) + 1)
    tf = normalize_atr_tf(atr_tf)
    if tf == "1w":
        return warmup * 6 + span_days + 10
    if tf == "1M":
        return warmup * 23 + span_days + 10
    return max(warmup + span_days + 10, warmup + 40)


def _session_date(ts: Any, *, intraday: bool) -> date:
    if intraday:
        return _as_utc(ts).tz_convert(SESSION_TZ).date()
    return _utc_naive(ts).date()


def session_ord(ts: Any, atr_tf: str, *, intraday: bool) -> int:
    """Comparable session id (int) for merge_asof.

    Intraday chart bars use America/New_York. Daily/HTF bars use the UTC
    calendar date so midnight-UTC 1d stamps stay on that session (not the
    previous NY evening).
    """
    tf = normalize_atr_tf(atr_tf)
    d = _session_date(ts, intraday=intraday)
    if tf == "1w":
        d = d - timedelta(days=d.weekday())
        return d.year * 10000 + d.month * 100 + d.day
    if tf == "1M":
        return d.year * 100 + d.month
    return d.year * 10000 + d.month * 100 + d.day


def session_ords(index: pd.Index, atr_tf: str, *, intraday: bool) -> np.ndarray:
    tf = normalize_atr_tf(atr_tf)
    out = np.empty(len(index), dtype=np.int64)
    for i, ts in enumerate(index):
        out[i] = session_ord(ts, tf, intraday=intraday)
    return out


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder RMA (Pine `ta.rma`): SMA seed, then alpha=1/period."""
    n = max(1, int(period))
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    alpha = 1.0 / float(n)
    seed = 0.0
    count = 0
    last = np.nan
    started = False
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            out[i] = last if started else np.nan
            continue
        if not started:
            seed += v
            count += 1
            if count >= n:
                last = seed / float(n)
                out[i] = last
                started = True
            continue
        last = last + alpha * (v - last)
        out[i] = last
    return pd.Series(out, index=series.index)


def wilder_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Pine `ta.atr`: RMA of true range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(tr, period)


def _resample_to_atr_tf(
    df: pd.DataFrame, atr_tf: str, *, chart_intraday: bool
) -> pd.DataFrame:
    """Build HTF OHLC from chart bars when a dedicated ATR window is missing."""
    ords = session_ords(df.index, atr_tf, intraday=chart_intraday)
    grouped = df.groupby(ords, sort=True)
    rows = []
    for _, g in grouped:
        if g.empty:
            continue
        rows.append(
            {
                "ts": g.index[-1],
                "open": g["open"].iloc[0],
                "high": g["high"].max(),
                "low": g["low"].min(),
                "close": g["close"].iloc[-1],
            }
        )
    if not rows:
        return df.iloc[0:0].copy()
    out = pd.DataFrame(rows).set_index("ts")
    return out


def _atr_asof_chart(
    chart_ords: np.ndarray,
    atr_ords: np.ndarray,
    atr_vals: np.ndarray,
    *,
    same_tf: bool,
) -> np.ndarray:
    htf = pd.DataFrame({"session": atr_ords, "atr": atr_vals})
    htf = htf.dropna(subset=["atr"])
    if htf.empty:
        return np.full(len(chart_ords), np.nan, dtype=float)
    htf = htf.drop_duplicates("session", keep="last").sort_values("session")
    left = pd.DataFrame({"session": chart_ords, "i": np.arange(len(chart_ords))})
    left = left.sort_values("session")
    merged = pd.merge_asof(
        left,
        htf,
        on="session",
        direction="backward",
        allow_exact_matches=bool(same_tf),
    )
    merged = merged.sort_values("i")
    return merged["atr"].to_numpy(dtype=float)


def atr_anchored_range(
    df: pd.DataFrame,
    atr_df: Optional[pd.DataFrame] = None,
    *,
    mode: Any = MODE_OPEN,
    atr_timeframe: Any = "1d",
    chart_timeframe: Any = "15m",
    period: int = 20,
    gp_one: float = 0.61,
    gp_two: float = 0.65,
) -> Dict[str, pd.Series]:
    """Return overlay series aligned to `df` (chart bars).

    Keys: mid, high, low, high2, low2, pb, gp_high1..4, gp_low1..4, atr.
    """
    if df is None or df.empty:
        empty = pd.Series(dtype=float)
        return {
            "mid": empty,
            "high": empty,
            "low": empty,
            "high2": empty,
            "low2": empty,
            "pb": empty,
            "atr": empty,
        }

    work = df.copy()
    for col in ("open", "high", "low", "close"):
        if col not in work.columns:
            raise ValueError("atr_anchored_range needs open/high/low/close")
        work[col] = pd.to_numeric(work[col], errors="coerce")

    mode_n = normalize_mode(mode)
    atr_tf = normalize_atr_tf(atr_timeframe)
    chart_tf = str(chart_timeframe or "").strip() or "15m"
    n_period = max(1, int(period))
    same_tf = TF_MINUTES.get(chart_tf, 1) >= TF_MINUTES.get(atr_tf, TF_MINUTES["1d"])
    chart_intraday = is_intraday(chart_tf)

    if atr_df is not None and not atr_df.empty:
        htf_work = atr_df.copy()
        htf_intraday = is_intraday(atr_tf)
        if same_tf:
            htf_intraday = chart_intraday
    elif same_tf:
        htf_work = work
        htf_intraday = chart_intraday
    else:
        htf_work = _resample_to_atr_tf(work, atr_tf, chart_intraday=chart_intraday)
        htf_intraday = chart_intraday

    for col in ("high", "low", "close"):
        if col not in htf_work.columns:
            raise ValueError("ATR frame needs high/low/close")
        htf_work[col] = pd.to_numeric(htf_work[col], errors="coerce")

    atr_series = wilder_atr(
        htf_work["high"], htf_work["low"], htf_work["close"], n_period
    )

    chart_ords = session_ords(work.index, atr_tf, intraday=chart_intraday)
    atr_ords = session_ords(htf_work.index, atr_tf, intraday=htf_intraday)
    atr_asof = _atr_asof_chart(
        chart_ords, atr_ords, atr_series.to_numpy(dtype=float), same_tf=same_tf
    )

    n = len(work)
    new_anchor = np.ones(n, dtype=bool)
    if n > 1:
        new_anchor[1:] = chart_ords[1:] != chart_ords[:-1]

    opens = work["open"].to_numpy(dtype=float)
    closes = work["close"].to_numpy(dtype=float)
    prior_close = np.roll(closes, 1)
    prior_close[0] = np.nan

    mid = np.full(n, np.nan, dtype=float)
    atr_snap = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if new_anchor[i]:
            if mode_n == MODE_PRIOR_CLOSE and np.isfinite(prior_close[i]):
                mid[i] = prior_close[i]
            else:
                mid[i] = opens[i]
            atr_snap[i] = atr_asof[i]
        else:
            mid[i] = mid[i - 1]
            atr_snap[i] = atr_snap[i - 1]

    half = atr_snap * 0.5
    high = mid + half
    low = mid - half
    high2 = high + half
    low2 = low - half
    width = high - low
    pb = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(width) & (width != 0) & np.isfinite(closes)
    pb[ok] = (closes[ok] - low[ok]) / width[ok]

    gp1 = float(gp_one)
    gp2 = float(gp_two)
    gp_high1 = low + width * gp1
    gp_high2 = low + width * gp2
    gp_low1 = high - width * gp1
    gp_low2 = high - width * gp2
    ext = high2 - mid
    gp_high3 = mid + ext * gp1
    gp_high4 = mid + ext * gp2
    gp_low3 = mid - ext * gp1
    gp_low4 = mid - ext * gp2

    idx = work.index
    return {
        "mid": pd.Series(mid, index=idx),
        "high": pd.Series(high, index=idx),
        "low": pd.Series(low, index=idx),
        "high2": pd.Series(high2, index=idx),
        "low2": pd.Series(low2, index=idx),
        "pb": pd.Series(pb, index=idx),
        "atr": pd.Series(atr_snap, index=idx),
        "gp_high1": pd.Series(gp_high1, index=idx),
        "gp_high2": pd.Series(gp_high2, index=idx),
        "gp_low1": pd.Series(gp_low1, index=idx),
        "gp_low2": pd.Series(gp_low2, index=idx),
        "gp_high3": pd.Series(gp_high3, index=idx),
        "gp_high4": pd.Series(gp_high4, index=idx),
        "gp_low3": pd.Series(gp_low3, index=idx),
        "gp_low4": pd.Series(gp_low4, index=idx),
    }


def overlay_payload(
    result: Mapping[str, pd.Series], *, show_gp: bool = False
) -> Dict[str, Union[list, bool]]:
    """JSON-safe overlay dict for the Charts page."""

    def _vals(key: str) -> list:
        series = result.get(key)
        if series is None:
            return []
        return [None if pd.isna(v) else float(v) for v in series]

    payload: Dict[str, Union[list, bool]] = {
        "high2": _vals("high2"),
        "high": _vals("high"),
        "low": _vals("low"),
        "low2": _vals("low2"),
        "mid": _vals("mid"),
        "pb": _vals("pb"),
        "show_gp": bool(show_gp),
    }
    if show_gp:
        for key in (
            "gp_high1",
            "gp_high2",
            "gp_low1",
            "gp_low2",
            "gp_high3",
            "gp_high4",
            "gp_low3",
            "gp_low4",
        ):
            payload[key] = _vals(key)
    return payload
