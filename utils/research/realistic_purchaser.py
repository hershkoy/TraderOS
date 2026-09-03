"""Close-confirm 15m fill: work the next same-session bar, fill at mid.

Signal exists at the close of bar T (IB labels bars at period start, so that
clock is T+15m = the open of bar T+1). The live window is that next 15m bar,
including the last RTH bar (15:45-16:00 ET). A confirming close at 16:00 has
no following RTH 15m and is cancelled.

Fill is the execution bar mid, not the signal-bar low. Cancel when
(mid-low)/mid exceeds ``max_low_to_mid_pct`` (default 0.5%; half the range,
about 1% high-low). Optional ``max_chase_pct`` caps mid vs the signal close
(off by default). Not wired into the backtester yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Mapping, Optional, Union
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
