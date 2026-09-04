"""Realistic fills for 15m close-confirm and daily signals that use 15m prints.

15m: signal exists at the close of bar T (IB labels bars at period start, so
that clock is T+15m = the open of bar T+1). Fill is the next bar's mid.
The last RTH bar (15:45-16:00 ET) is a valid window; a confirming close at
16:00 has no following RTH 15m and is cancelled.

1d: given daily signal buy price X, take the first RTH 15m bar that day whose
range contains X, then the next same-session 15m bar. Fill is
(X + next_mid) / 2. If X never prints, or it only prints on the 15:45 bar,
cancel. Same wild-bar gate on the next 15m.

Cancel when (mid-low)/mid exceeds ``max_low_to_mid_pct`` (default 0.5%).
Optional ``max_chase_pct`` is off by default. Hooked via ``--realistic-fill`` in
``backtest_channel_touch_trades.py`` / ``backtest_channel_touch_h2_break.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, List, Mapping, Optional, Sequence, Union
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
LAST_RTH_15M = time(15, 45)
BAR_MINUTES = 15
DEFAULT_MAX_LOW_TO_MID_PCT = 0.005

REASON_FILLED = "filled"
REASON_NO_NEXT_BAR = "no_next_bar"
REASON_END_OF_SESSION = "end_of_session"
REASON_OVERNIGHT = "overnight"
REASON_WILD_RANGE = "wild_low_to_mid"
REASON_CHASE = "chase"
REASON_BAD_OHLC = "bad_ohlc"
REASON_BAD_SIGNAL = "bad_signal"
REASON_PRICE_NOT_PRINTED = "price_not_printed"

BarLike = Mapping[str, Any]
TsLike = Union[datetime, pd.Timestamp, str]


@dataclass(frozen=True)
class PurchaseResult:
    filled: bool
    reason: str
    fill_px: Optional[float] = None
    signal_time: Optional[datetime] = None
    signal_px: Optional[float] = None
    exec_bar_ts: Optional[datetime] = None
    mid: Optional[float] = None
    low_to_mid_pct: Optional[float] = None
    chase_pct: Optional[float] = None
    hit_bar_ts: Optional[datetime] = None


def bar_mid(high: float, low: float) -> float:
    return (float(high) + float(low)) / 2.0


def low_to_mid_pct(high: float, low: float, *, ref: Optional[float] = None) -> float:
    """(mid - low) / ref. Default ref is mid (half the bar range as a fraction of mid)."""
    mid = bar_mid(high, low)
    base = float(mid if ref is None else ref)
    if base <= 0:
        return float("nan")
    return (mid - float(low)) / base


def _as_utc_aware(ts: TsLike, *, naive_tz: str = "UTC") -> datetime:
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            raise ValueError("timestamp is NaT")
        if ts.tzinfo is None:
            ts = ts.tz_localize(naive_tz)
        return ts.tz_convert("UTC").to_pydatetime()
    if not isinstance(ts, datetime):
        raise TypeError("timestamp must be datetime, Timestamp, or str")
    if ts.tzinfo is None:
        zone = timezone.utc if naive_tz.upper() == "UTC" else ZoneInfo(naive_tz)
        return ts.replace(tzinfo=zone).astimezone(timezone.utc)
    return ts.astimezone(timezone.utc)


def as_et(ts: TsLike, *, naive_tz: str = "UTC") -> datetime:
    """IB/TimescaleDB naive 15m stamps are UTC. Session rules use America/New_York."""
    return _as_utc_aware(ts, naive_tz=naive_tz).astimezone(ET)


def _floor_15m(ts_et: datetime) -> datetime:
    minute = (ts_et.minute // BAR_MINUTES) * BAR_MINUTES
    return ts_et.replace(minute=minute, second=0, microsecond=0)


def is_rth_15m_bar_start(ts: TsLike, *, naive_tz: str = "UTC") -> bool:
    ts_et = _floor_15m(as_et(ts, naive_tz=naive_tz))
    if ts_et.weekday() >= 5:
        return False
    t = ts_et.time()
    if t.second != 0 or t.microsecond != 0:
        t = ts_et.replace(second=0, microsecond=0).time()
    if ts_et.minute % BAR_MINUTES != 0:
        return False
    return RTH_OPEN <= t <= LAST_RTH_15M


def is_last_rth_15m_bar(ts: TsLike, *, naive_tz: str = "UTC") -> bool:
    ts_et = _floor_15m(as_et(ts, naive_tz=naive_tz))
    return (
        is_rth_15m_bar_start(ts_et, naive_tz=naive_tz)
        and ts_et.hour == 15
        and ts_et.minute == 45
    )


def signal_time_from_bar(signal_bar_ts: TsLike, *, naive_tz: str = "UTC") -> datetime:
    """Clock when close-confirm exists: period start + 15m (open of the next bar)."""
    start = _floor_15m(as_et(signal_bar_ts, naive_tz=naive_tz))
    return start + timedelta(minutes=BAR_MINUTES)


def expected_exec_bar_start(signal_bar_ts: TsLike, *, naive_tz: str = "UTC") -> Optional[datetime]:
    """Period start of the bar you work after close-confirm, or None at 16:00."""
    start = _floor_15m(as_et(signal_bar_ts, naive_tz=naive_tz))
    if not is_rth_15m_bar_start(start, naive_tz=naive_tz):
        return None
    if start.hour == 15 and start.minute == 45:
        return None
    nxt = start + timedelta(minutes=BAR_MINUTES)
    if not is_rth_15m_bar_start(nxt, naive_tz=naive_tz):
        return None
    return nxt


def _px(bar: BarLike, *keys: str) -> Optional[float]:
    for key in keys:
        if key in bar and bar[key] is not None:
            try:
                val = float(bar[key])
            except (TypeError, ValueError):
                return None
            if val == val and val > 0:
                return val
    return None


def _bar_ts(bar: BarLike, *, naive_tz: str) -> Optional[datetime]:
    raw = None
    for key in ("ts", "timestamp", "time"):
        if key in bar and bar[key] is not None:
            raw = bar[key]
            break
    if raw is None:
        return None
    try:
        return as_et(raw, naive_tz=naive_tz)
    except (TypeError, ValueError):
        return None


def _hl_ok(high: Optional[float], low: Optional[float]) -> bool:
    return high is not None and low is not None and high >= low


def _result(
    reason: str,
    *,
    filled: bool = False,
    fill_px: Optional[float] = None,
    signal_time: Optional[datetime] = None,
    signal_px: Optional[float] = None,
    exec_bar_ts: Optional[datetime] = None,
    mid: Optional[float] = None,
    low_to_mid: Optional[float] = None,
    chase: Optional[float] = None,
    hit_bar_ts: Optional[datetime] = None,
) -> PurchaseResult:
    return PurchaseResult(
        filled=filled,
        reason=reason,
        fill_px=fill_px,
        signal_time=signal_time,
        signal_px=signal_px,
        exec_bar_ts=exec_bar_ts,
        mid=mid,
        low_to_mid_pct=low_to_mid,
        chase_pct=chase,
        hit_bar_ts=hit_bar_ts,
    )


def purchase_after_close_signal(
    signal_bar: BarLike,
    next_bar: Optional[BarLike] = None,
    *,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    """Fill at mid of the next same-session 15m bar after a close-confirm signal.

    ``max_low_to_mid_pct`` of None disables the wild-bar gate.
    ``max_chase_pct`` of None (default) disables the upside chase gate.
    """
    sig_ts = _bar_ts(signal_bar, naive_tz=naive_tz)
    sig_h = _px(signal_bar, "high")
    sig_l = _px(signal_bar, "low")
    sig_c = _px(signal_bar, "close")
    if sig_ts is None or sig_c is None or not _hl_ok(sig_h, sig_l):
        return _result(REASON_BAD_SIGNAL)
    if not is_rth_15m_bar_start(sig_ts, naive_tz=naive_tz):
        return _result(REASON_BAD_SIGNAL, signal_px=sig_c)

    sig_clock = signal_time_from_bar(sig_ts, naive_tz=naive_tz)
    want = expected_exec_bar_start(sig_ts, naive_tz=naive_tz)
    if want is None:
        return _result(
            REASON_END_OF_SESSION,
            signal_time=sig_clock,
            signal_px=sig_c,
        )

    if next_bar is None:
        return _result(
            REASON_NO_NEXT_BAR,
            signal_time=sig_clock,
            signal_px=sig_c,
            exec_bar_ts=want,
        )

    nxt_ts = _bar_ts(next_bar, naive_tz=naive_tz)
    nxt_h = _px(next_bar, "high")
    nxt_l = _px(next_bar, "low")
    if nxt_ts is None or not _hl_ok(nxt_h, nxt_l):
        return _result(
            REASON_BAD_OHLC,
            signal_time=sig_clock,
            signal_px=sig_c,
            exec_bar_ts=want,
        )

    nxt_floor = _floor_15m(nxt_ts)
    want_floor = _floor_15m(want)
    if nxt_floor != want_floor:
        overnight = nxt_floor.date() != sig_ts.date()
        return _result(
            REASON_OVERNIGHT if overnight else REASON_NO_NEXT_BAR,
            signal_time=sig_clock,
            signal_px=sig_c,
            exec_bar_ts=nxt_floor,
        )

    mid = bar_mid(nxt_h, nxt_l)
    l2m = low_to_mid_pct(nxt_h, nxt_l)
    chase = (mid - sig_c) / sig_c if sig_c else float("nan")
    if max_low_to_mid_pct is not None and l2m > float(max_low_to_mid_pct):
        return _result(
            REASON_WILD_RANGE,
            signal_time=sig_clock,
            signal_px=sig_c,
            exec_bar_ts=nxt_floor,
            mid=mid,
            low_to_mid=l2m,
            chase=chase,
        )
    if max_chase_pct is not None and chase > float(max_chase_pct):
        return _result(
            REASON_CHASE,
            signal_time=sig_clock,
            signal_px=sig_c,
            exec_bar_ts=nxt_floor,
            mid=mid,
            low_to_mid=l2m,
            chase=chase,
        )
    return _result(
        REASON_FILLED,
        filled=True,
        fill_px=mid,
        signal_time=sig_clock,
        signal_px=sig_c,
        exec_bar_ts=nxt_floor,
        mid=mid,
        low_to_mid=l2m,
        chase=chase,
    )


def _row_bar(df: pd.DataFrame, i: int, *, naive_tz: str) -> dict:
    row = df.iloc[i]
    ts = df.index[i]
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        ts = row["timestamp"]
    return {
        "ts": ts,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]) if "volume" in df.columns else float("nan"),
    }


def purchase_at_signal_index(
    df: pd.DataFrame,
    signal_i: int,
    *,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    """Same fill rule using a 15m OHLCV frame (DatetimeIndex or timestamp column)."""
    if df is None or df.empty or signal_i < 0 or signal_i >= len(df):
        return _result(REASON_BAD_SIGNAL)
    signal_bar = _row_bar(df, signal_i, naive_tz=naive_tz)
    next_bar = None
    if signal_i + 1 < len(df):
        next_bar = _row_bar(df, signal_i + 1, naive_tz=naive_tz)
    return purchase_after_close_signal(
        signal_bar,
        next_bar,
        max_low_to_mid_pct=max_low_to_mid_pct,
        max_chase_pct=max_chase_pct,
        naive_tz=naive_tz,
    )


def bar_contains_price(high: float, low: float, px: float, *, eps: float = 1e-8) -> bool:
    return float(low) - eps <= float(px) <= float(high) + eps


def _as_session_date(value: Any, *, naive_tz: str) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime) or isinstance(value, pd.Timestamp):
        return as_et(value, naive_tz=naive_tz).date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _coerce_15m_bars(
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    naive_tz: str,
) -> List[dict]:
    if bars_15m is None:
        return []
    if isinstance(bars_15m, pd.DataFrame):
        if bars_15m.empty:
            return []
        rows = [_row_bar(bars_15m, i, naive_tz=naive_tz) for i in range(len(bars_15m))]
    else:
        rows = [dict(b) for b in bars_15m]
    keyed = []
    for row in rows:
        ts = _bar_ts(row, naive_tz=naive_tz)
        if ts is None or not is_rth_15m_bar_start(ts, naive_tz=naive_tz):
            continue
        row = dict(row)
        row["ts"] = _floor_15m(ts)
        keyed.append(row)
    keyed.sort(key=lambda b: b["ts"])
    return keyed


def purchase_after_daily_signal(
    signal_buy_price: float,
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    session_date: Any = None,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    """Daily signal at price X: first 15m print of X that session, then blend with next mid.

    entry = (X + next_15m_mid) / 2. Wild/EOD rules match the 15m purchaser
    (next bar after the print; 15:45 print has no window).
    """
    try:
        px = float(signal_buy_price)
    except (TypeError, ValueError):
        return _result(REASON_BAD_SIGNAL)
    if px != px or px <= 0:
        return _result(REASON_BAD_SIGNAL)

    day = _as_session_date(session_date, naive_tz=naive_tz)
    bars = _coerce_15m_bars(bars_15m, naive_tz=naive_tz)
    if day is not None:
        bars = [b for b in bars if b["ts"].date() == day]
    if not bars:
        return _result(REASON_BAD_SIGNAL, signal_px=px)

    hit_i = None
    for i, bar in enumerate(bars):
        high = _px(bar, "high")
        low = _px(bar, "low")
        if not _hl_ok(high, low):
            continue
        if bar_contains_price(high, low, px):
            hit_i = i
            break
    if hit_i is None:
        return _result(REASON_PRICE_NOT_PRINTED, signal_px=px)

    hit = bars[hit_i]
    nxt = bars[hit_i + 1] if hit_i + 1 < len(bars) else None
    inner = purchase_after_close_signal(
        hit,
        nxt,
        max_low_to_mid_pct=max_low_to_mid_pct,
        max_chase_pct=None,
        naive_tz=naive_tz,
    )
    hit_ts = hit["ts"]
    if not inner.filled:
        return _result(
            inner.reason,
            signal_time=inner.signal_time,
            signal_px=px,
            exec_bar_ts=inner.exec_bar_ts,
            mid=inner.mid,
            low_to_mid=inner.low_to_mid_pct,
            chase=inner.chase_pct,
            hit_bar_ts=hit_ts,
        )
    mid = inner.mid
    if mid is None:
        return _result(
            REASON_BAD_OHLC,
            signal_time=inner.signal_time,
            signal_px=px,
            hit_bar_ts=hit_ts,
        )
    fill = (px + float(mid)) / 2.0
    chase = (float(mid) - px) / px
    if max_chase_pct is not None and chase > float(max_chase_pct):
        return _result(
            REASON_CHASE,
            signal_time=inner.signal_time,
            signal_px=px,
            exec_bar_ts=inner.exec_bar_ts,
            mid=mid,
            low_to_mid=inner.low_to_mid_pct,
            chase=chase,
            hit_bar_ts=hit_ts,
        )
    return _result(
        REASON_FILLED,
        filled=True,
        fill_px=fill,
        signal_time=inner.signal_time,
        signal_px=px,
        exec_bar_ts=inner.exec_bar_ts,
        mid=mid,
        low_to_mid=inner.low_to_mid_pct,
        chase=chase,
        hit_bar_ts=hit_ts,
    )


def exec_fill_15m_after_signal(
    df: pd.DataFrame,
    signal_i: int,
    *,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
) -> Optional[tuple]:
    """Next same-session 15m mid after close-confirm at ``signal_i``, or None."""
    got = purchase_at_signal_index(
        df,
        signal_i,
        max_low_to_mid_pct=max_low_to_mid_pct,
        max_chase_pct=max_chase_pct,
        naive_tz=naive_tz,
    )
    if not got.filled or got.fill_px is None:
        return None
    return int(signal_i) + 1, float(got.fill_px)


def exec_fill_daily_with_15m(
    signal_buy_price: float,
    df_15m: Optional[pd.DataFrame],
    session_date: Any,
    *,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
) -> Optional[float]:
    """Blend daily X with the next 15m mid after the first print of X."""
    if df_15m is None or df_15m.empty:
        return None
    got = purchase_after_daily_signal(
        signal_buy_price,
        df_15m,
        session_date=session_date,
        max_low_to_mid_pct=max_low_to_mid_pct,
        max_chase_pct=max_chase_pct,
        naive_tz=naive_tz,
    )
    if not got.filled or got.fill_px is None:
        return None
    return float(got.fill_px)
