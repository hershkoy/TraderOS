"""Realistic fills for 15m close-confirm and daily signals that use 15m prints.

Default 15m fill (``signal-close``): buy at the close of the signal bar — the
bar that tagged the rail. That close is known when the bar completes, so the
BUY sits on the touch candle instead of the next bar's mid (which can be
mid-channel and never tagged support).

Kept 15m fill (``next-mid``): signal exists at the close of bar T (IB labels
bars at period start, so that clock is T+15m = the open of bar T+1). Fill is
the next bar's mid. The last RTH bar (15:45-16:00 ET) has no following RTH 15m
and is cancelled. Wild-bar gate: cancel when (mid-low)/mid exceeds
``max_low_to_mid_pct`` (default 0.5%).

1d default (``signal-close``): first RTH 15m that prints X, fill at that bar's
close. ``next-mid`` blends X with the next 15m mid (cancels on 15:45 prints).
``open-cross``: first RTH 15m whose **open** is already above resist; fill at
that same bar's **close** (known when the 15m completes). Last RTH bar is
allowed. Days with no 15m open above resist are skipped.

``next-open`` (1d): signal is the completed daily close; fill at the **next
session's open** (MOO after an EOD scan). No IB 15m join. Last bar of the
sample has no next open and is skipped. 15m ``next-open`` buys the next same-
session 15m **open** (last RTH cancelled, same as next-mid).

``next-open-mid`` (1d): same EOD signal, then buy the **mid** of the next
session's first RTH 15m (09:30 ET). Needs IB 15m. Names with no next-session
09:30 print are skipped. Live-executable: you know the daily close at 16:00
ET and work the next open bar's midpoint.

``hot-cross`` (1d buy-now): first RTH 15m whose **high** reaches resist after
H2, without waiting for that session's daily close. Fill heuristic ``lerp85``
is ``rail + 0.85 * (close - rail)`` clamped to the bar. ``rail`` / ``close``
are bounds. A 15m gap is ``open >= rail`` (not a daily gap).

``close-cross`` (1d close-confirm): first RTH 15m whose **close** is above
resist. Fill follows ``fill_mode``: ``signal-close`` buys that confirm bar's
close (last RTH allowed); ``next-mid`` / ``next-open`` buy the next same-session
15m mid/open (15:45 confirm cancels). Does not wait for the daily close.

Optional ``max_chase_pct`` is off by default. Hooked via ``--realistic-fill``
and ``--realistic-fill-mode`` in ``backtest_channel_touch_trades.py`` /
``backtest_channel_touch_h2_break.py``. ``--intraday-trigger hot-cross`` uses
``purchase_hot_cross_15m``; ``close-cross`` uses ``purchase_close_cross_15m``.
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
FILL_MODE_SIGNAL_CLOSE = "signal-close"
FILL_MODE_NEXT_MID = "next-mid"
FILL_MODE_OPEN_CROSS = "open-cross"
FILL_MODE_NEXT_OPEN = "next-open"
FILL_MODE_NEXT_OPEN_MID = "next-open-mid"
DEFAULT_FILL_MODE = FILL_MODE_SIGNAL_CLOSE
FILL_MODES_1D = (
    FILL_MODE_SIGNAL_CLOSE,
    FILL_MODE_NEXT_MID,
    FILL_MODE_OPEN_CROSS,
    FILL_MODE_NEXT_OPEN,
    FILL_MODE_NEXT_OPEN_MID,
)
HOT_CROSS_FILL_LERP85 = "lerp85"
HOT_CROSS_FILL_RAIL = "rail"
HOT_CROSS_FILL_CLOSE = "close"
HOT_CROSS_FILLS = (
    HOT_CROSS_FILL_LERP85,
    HOT_CROSS_FILL_RAIL,
    HOT_CROSS_FILL_CLOSE,
)
DEFAULT_HOT_CROSS_FILL = HOT_CROSS_FILL_LERP85
DEFAULT_HOT_CROSS_LERP = 0.85
INTRADAY_TRIGGER_HOT_CROSS = "hot-cross"
INTRADAY_TRIGGER_CLOSE_CROSS = "close-cross"
INTRADAY_TRIGGERS = (INTRADAY_TRIGGER_HOT_CROSS, INTRADAY_TRIGGER_CLOSE_CROSS)


def normalize_fill_mode(raw: Optional[str]) -> str:
    text = str(raw or DEFAULT_FILL_MODE).strip().lower().replace("_", "-")
    if text in (FILL_MODE_NEXT_MID, "mid", "next-bar-mid", "nextmid"):
        return FILL_MODE_NEXT_MID
    if text in (FILL_MODE_SIGNAL_CLOSE, "close", "signalclose", "bar-close", "current-close"):
        return FILL_MODE_SIGNAL_CLOSE
    if text in (FILL_MODE_OPEN_CROSS, "opencross", "open-confirm", "openconfirm"):
        return FILL_MODE_OPEN_CROSS
    if text in (
        FILL_MODE_NEXT_OPEN_MID,
        "nextopenmid",
        "next-open-15m-mid",
        "next-session-open-mid",
    ):
        return FILL_MODE_NEXT_OPEN_MID
    if text in (
        FILL_MODE_NEXT_OPEN,
        "nextopen",
        "next-session-open",
        "next-day-open",
        "moo",
    ):
        return FILL_MODE_NEXT_OPEN
    raise ValueError("unknown realistic fill mode: %s" % raw)


def normalize_hot_cross_fill(raw: Optional[str]) -> str:
    text = str(raw or DEFAULT_HOT_CROSS_FILL).strip().lower().replace("_", "-")
    if text in (HOT_CROSS_FILL_LERP85, "lerp", "0.85", "85"):
        return HOT_CROSS_FILL_LERP85
    if text in (HOT_CROSS_FILL_RAIL, "resist", "stop"):
        return HOT_CROSS_FILL_RAIL
    if text in (HOT_CROSS_FILL_CLOSE, "signal-close", "bar-close"):
        return HOT_CROSS_FILL_CLOSE
    raise ValueError("unknown hot-cross fill: %s" % raw)


def needs_15m_purchase_panels(
    timeframe: str,
    *,
    realistic_fill: bool,
    fill_mode: str,
    intraday_trigger: str = "",
) -> bool:
    """True when a 1d book must join IB 15m to price the fill."""
    if str(timeframe or "").strip().lower() not in ("1d", "d", "daily"):
        return False
    trig = str(intraday_trigger or "").strip().lower().replace("_", "-")
    if trig in INTRADAY_TRIGGERS:
        return True
    if not realistic_fill:
        return False
    return normalize_fill_mode(fill_mode) != FILL_MODE_NEXT_OPEN

REASON_FILLED = "filled"
REASON_NO_NEXT_BAR = "no_next_bar"
REASON_END_OF_SESSION = "end_of_session"
REASON_OVERNIGHT = "overnight"
REASON_WILD_RANGE = "wild_low_to_mid"
REASON_CHASE = "chase"
REASON_BAD_OHLC = "bad_ohlc"
REASON_BAD_SIGNAL = "bad_signal"
REASON_PRICE_NOT_PRINTED = "price_not_printed"
REASON_NO_OPEN_CROSS = "no_open_cross"
REASON_NO_NEXT_OPEN = "no_next_open"
REASON_NO_HOT_CROSS = "no_hot_cross"
REASON_NO_CLOSE_CROSS = "no_close_cross"

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
    gap_15m: Optional[bool] = None
    bar_open: Optional[float] = None
    bar_high: Optional[float] = None
    bar_low: Optional[float] = None
    bar_close: Optional[float] = None


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
    gap_15m: Optional[bool] = None,
    bar_open: Optional[float] = None,
    bar_high: Optional[float] = None,
    bar_low: Optional[float] = None,
    bar_close: Optional[float] = None,
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
        gap_15m=gap_15m,
        bar_open=bar_open,
        bar_high=bar_high,
        bar_low=bar_low,
        bar_close=bar_close,
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


def purchase_at_signal_close(
    signal_bar: BarLike,
    *,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    """Fill at the close of the signal bar (known when that 15m bar completes).

    Last RTH bar is allowed — there is no next-bar requirement.
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
    start = _floor_15m(as_et(sig_ts, naive_tz=naive_tz))
    mid = bar_mid(sig_h, sig_l)
    l2m = low_to_mid_pct(sig_h, sig_l)
    return _result(
        REASON_FILLED,
        filled=True,
        fill_px=float(sig_c),
        signal_time=sig_clock,
        signal_px=sig_c,
        exec_bar_ts=start,
        mid=mid,
        low_to_mid=l2m,
        chase=0.0,
        hit_bar_ts=start,
    )


def purchase_next_bar_open(
    signal_bar: BarLike,
    next_bar: Optional[BarLike],
    *,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    """Fill at the next same-session 15m **open** (known at the signal close).

    Last RTH bar has no following print and is cancelled. No wild-bar gate:
    the open is the executable MOO-style price, not a mid.
    """
    inner = purchase_after_close_signal(
        signal_bar,
        next_bar,
        max_low_to_mid_pct=None,
        max_chase_pct=None,
        naive_tz=naive_tz,
    )
    if not inner.filled:
        return inner
    opened = _px(next_bar, "open") if next_bar is not None else None
    if opened is None or opened != opened or float(opened) <= 0:
        return _result(
            REASON_BAD_OHLC,
            signal_time=inner.signal_time,
            signal_px=inner.signal_px,
            exec_bar_ts=inner.exec_bar_ts,
        )
    chase = None
    if inner.signal_px:
        chase = (float(opened) - float(inner.signal_px)) / float(inner.signal_px)
    return _result(
        REASON_FILLED,
        filled=True,
        fill_px=float(opened),
        signal_time=inner.signal_time,
        signal_px=inner.signal_px,
        exec_bar_ts=inner.exec_bar_ts,
        mid=inner.mid,
        low_to_mid=inner.low_to_mid_pct,
        chase=chase,
        hit_bar_ts=inner.exec_bar_ts,
    )


def purchase_next_daily_open(
    df: pd.DataFrame,
    signal_i: int,
) -> PurchaseResult:
    """1d: buy the next session's **open** after a completed close signal.

    Nightly-honest: EOD scan on bar T, MOO on T+1. No 15m join. The last daily
    bar in the sample has no next open.
    """
    if df is None or df.empty or signal_i < 0 or signal_i >= len(df):
        return _result(REASON_BAD_SIGNAL)
    nxt = int(signal_i) + 1
    if nxt >= len(df):
        return _result(REASON_NO_NEXT_OPEN)
    if "open" not in df.columns:
        return _result(REASON_BAD_OHLC)
    try:
        px = float(df["open"].iloc[nxt])
    except (TypeError, ValueError):
        return _result(REASON_BAD_OHLC)
    if px != px or px <= 0:
        return _result(REASON_BAD_OHLC)
    sig_c = None
    if "close" in df.columns:
        try:
            sig_c = float(df["close"].iloc[signal_i])
        except (TypeError, ValueError):
            sig_c = None
    ts = df.index[nxt]
    exec_ts = pd.Timestamp(ts).to_pydatetime()
    return _result(
        REASON_FILLED,
        filled=True,
        fill_px=px,
        signal_px=sig_c,
        exec_bar_ts=exec_ts,
        hit_bar_ts=exec_ts,
    )


def _rth_open_15m_bar(bars: Sequence[BarLike]) -> Optional[dict]:
    """First 09:30 ET RTH 15m, else the first RTH bar of the session."""
    if not bars:
        return None
    for bar in bars:
        ts = bar.get("ts")
        if ts is None:
            continue
        et = as_et(ts)
        if et.hour == 9 and et.minute == 30:
            return dict(bar)
    return dict(bars[0])


def purchase_next_session_open_mid(
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    signal_session_date: Any = None,
    naive_tz: str = "UTC",
    session_index: Optional[dict] = None,
) -> PurchaseResult:
    """1d EOD close-confirm: buy mid of the next session's 09:30 ET 15m.

    Signal date is the daily bar's US cash session. Fill is the next RTH
    session's opening 15m midpoint. No wild-bar cancel: the open bar is the
    executable window. Skip when there is no later RTH 15m session.
    """
    day = _as_session_date(signal_session_date, naive_tz=naive_tz)
    if day is None:
        return _result(REASON_BAD_SIGNAL)
    indexed = session_index if session_index is not None else index_rth_15m_by_session(
        bars_15m, naive_tz=naive_tz
    )
    later = [d for d in indexed.keys() if d > day]
    if not later:
        return _result(REASON_NO_NEXT_OPEN)
    nxt = min(later)
    hit = _rth_open_15m_bar(indexed.get(nxt) or [])
    if hit is None:
        return _result(REASON_NO_NEXT_OPEN)
    high = _px(hit, "high")
    low = _px(hit, "low")
    if not _hl_ok(high, low):
        return _result(REASON_BAD_OHLC)
    mid = bar_mid(float(high), float(low))
    if mid != mid or mid <= 0:
        return _result(REASON_BAD_OHLC)
    hit_ts = hit.get("ts")
    start = _floor_15m(as_et(hit_ts, naive_tz=naive_tz)) if hit_ts is not None else None
    return _result(
        REASON_FILLED,
        filled=True,
        fill_px=float(mid),
        exec_bar_ts=start,
        mid=mid,
        low_to_mid=low_to_mid_pct(float(high), float(low)),
        hit_bar_ts=start,
        bar_open=_px(hit, "open"),
        bar_high=float(high),
        bar_low=float(low),
        bar_close=_px(hit, "close"),
    )


def _rth_last_15m_bar(bars: Sequence[BarLike]) -> Optional[dict]:
    """15:45 ET last RTH 15m, else the last RTH bar of the session."""
    if not bars:
        return None
    for bar in reversed(list(bars)):
        ts = bar.get("ts")
        if ts is None:
            continue
        et = as_et(ts)
        if et.hour == 15 and et.minute == 45:
            return dict(bar)
    return dict(bars[-1])


def purchase_last_rth_open_above_mid(
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    signal_session_date: Any = None,
    rail: Any = None,
    naive_tz: str = "UTC",
    session_index: Optional[dict] = None,
) -> PurchaseResult:
    """Same-session last RTH 15m: fill mid only if that bar **opened** above rail.

    Open is known at 15:45 ET; mid is known at 16:00 with the daily close.
    Skip when open <= rail or the session has no RTH 15m.
    """
    day = _as_session_date(signal_session_date, naive_tz=naive_tz)
    try:
        lvl = float(rail)
    except (TypeError, ValueError):
        lvl = float("nan")
    if day is None or lvl != lvl or lvl <= 0:
        return _result(REASON_BAD_SIGNAL)
    indexed = session_index if session_index is not None else index_rth_15m_by_session(
        bars_15m, naive_tz=naive_tz
    )
    hit = _rth_last_15m_bar(indexed.get(day) or [])
    if hit is None:
        return _result(REASON_END_OF_SESSION, signal_px=lvl)
    opened = _px(hit, "open")
    high = _px(hit, "high")
    low = _px(hit, "low")
    if opened is None or not _hl_ok(high, low):
        return _result(REASON_BAD_OHLC, signal_px=lvl)
    hit_ts = hit.get("ts")
    start = _floor_15m(as_et(hit_ts, naive_tz=naive_tz)) if hit_ts is not None else None
    if float(opened) <= lvl:
        return _result(
            REASON_NO_OPEN_CROSS,
            signal_px=lvl,
            exec_bar_ts=start,
            hit_bar_ts=start,
            bar_open=float(opened),
            bar_high=float(high),
            bar_low=float(low),
            bar_close=_px(hit, "close"),
        )
    mid = bar_mid(float(high), float(low))
    if mid != mid or mid <= 0:
        return _result(REASON_BAD_OHLC, signal_px=lvl, bar_open=float(opened))
    return _result(
        REASON_FILLED,
        filled=True,
        fill_px=float(mid),
        signal_px=lvl,
        exec_bar_ts=start,
        mid=mid,
        low_to_mid=low_to_mid_pct(float(high), float(low)),
        hit_bar_ts=start,
        bar_open=float(opened),
        bar_high=float(high),
        bar_low=float(low),
        bar_close=_px(hit, "close"),
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
    fill_mode: str = DEFAULT_FILL_MODE,
) -> PurchaseResult:
    """Fill using a 15m OHLCV frame (DatetimeIndex or timestamp column)."""
    if df is None or df.empty or signal_i < 0 or signal_i >= len(df):
        return _result(REASON_BAD_SIGNAL)
    signal_bar = _row_bar(df, signal_i, naive_tz=naive_tz)
    mode = normalize_fill_mode(fill_mode)
    if mode in (FILL_MODE_SIGNAL_CLOSE, FILL_MODE_OPEN_CROSS):
        return purchase_at_signal_close(signal_bar, naive_tz=naive_tz)
    next_bar = None
    if signal_i + 1 < len(df):
        next_bar = _row_bar(df, signal_i + 1, naive_tz=naive_tz)
    if mode == FILL_MODE_NEXT_OPEN:
        return purchase_next_bar_open(signal_bar, next_bar, naive_tz=naive_tz)
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
    """US cash session date for a daily bar.

    Daily OHLCV here is keyed by the session calendar date at 00:00 (naive or
    UTC). Converting that instant to America/New_York turns Monday 00:00 UTC
    into Sunday evening, so ``--realistic-fill`` looks up Sunday 15m and drops
    every Monday (and fills Tue-Fri from the prior session).
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime) or isinstance(value, pd.Timestamp):
        ts = pd.Timestamp(value)
        clock = ts.time()
        if clock.hour == 0 and clock.minute == 0 and clock.second == 0 and clock.microsecond == 0:
            return ts.date()
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
    fill_mode: str = FILL_MODE_NEXT_MID,
) -> PurchaseResult:
    """Daily signal at price X: first 15m print of X that session.

    ``next-mid`` (this function's default, kept): entry = (X + next_15m_mid) / 2.
    Wild/EOD rules match the 15m next-mid purchaser (15:45 print has no window).

    ``signal-close``: fill at that print bar's close (last RTH print is allowed).
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
    hit_ts = hit["ts"]
    mode = normalize_fill_mode(fill_mode)
    if mode == FILL_MODE_SIGNAL_CLOSE:
        hit_c = _px(hit, "close")
        hit_h = _px(hit, "high")
        hit_l = _px(hit, "low")
        if hit_c is None or not _hl_ok(hit_h, hit_l):
            return _result(REASON_BAD_OHLC, signal_px=px, hit_bar_ts=hit_ts)
        chase = (float(hit_c) - px) / px
        if max_chase_pct is not None and chase > float(max_chase_pct):
            return _result(
                REASON_CHASE,
                signal_px=px,
                exec_bar_ts=hit_ts,
                mid=bar_mid(hit_h, hit_l),
                chase=chase,
                hit_bar_ts=hit_ts,
            )
        inner_close = purchase_at_signal_close(hit, naive_tz=naive_tz)
        if not inner_close.filled:
            return _result(
                inner_close.reason,
                signal_px=px,
                exec_bar_ts=inner_close.exec_bar_ts,
                hit_bar_ts=hit_ts,
            )
        return _result(
            REASON_FILLED,
            filled=True,
            fill_px=float(hit_c),
            signal_time=inner_close.signal_time,
            signal_px=px,
            exec_bar_ts=inner_close.exec_bar_ts,
            mid=inner_close.mid,
            low_to_mid=inner_close.low_to_mid_pct,
            chase=chase,
            hit_bar_ts=hit_ts,
        )
    nxt = bars[hit_i + 1] if hit_i + 1 < len(bars) else None
    inner = purchase_after_close_signal(
        hit,
        nxt,
        max_low_to_mid_pct=max_low_to_mid_pct,
        max_chase_pct=None,
        naive_tz=naive_tz,
    )
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


def purchase_open_cross_15m(
    resist: float,
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    session_date: Any = None,
    error_pct: float = 0.0,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    """First RTH 15m that **opens** above resist; fill at that bar's close.

    Detection is the open (already through the rail at the print). The close is
    the executable price once that 15m completes. Last RTH bar is allowed.
    """
    try:
        lvl = float(resist)
    except (TypeError, ValueError):
        return _result(REASON_BAD_SIGNAL)
    if lvl != lvl or lvl <= 0:
        return _result(REASON_BAD_SIGNAL)
    floor = lvl * (1.0 + max(0.0, float(error_pct)) / 100.0)
    day = _as_session_date(session_date, naive_tz=naive_tz)
    bars = _coerce_15m_bars(bars_15m, naive_tz=naive_tz)
    if day is not None:
        bars = [b for b in bars if b["ts"].date() == day]
    if not bars:
        return _result(REASON_BAD_SIGNAL, signal_px=lvl)

    for bar in bars:
        opened = _px(bar, "open")
        if opened is None:
            continue
        if opened <= floor:
            continue
        inner = purchase_at_signal_close(bar, naive_tz=naive_tz)
        hit_ts = bar.get("ts")
        if not inner.filled:
            return _result(
                inner.reason,
                signal_px=lvl,
                exec_bar_ts=inner.exec_bar_ts,
                hit_bar_ts=hit_ts,
            )
        return _result(
            REASON_FILLED,
            filled=True,
            fill_px=float(inner.fill_px) if inner.fill_px is not None else None,
            signal_time=inner.signal_time,
            signal_px=lvl,
            exec_bar_ts=inner.exec_bar_ts,
            mid=inner.mid,
            low_to_mid=inner.low_to_mid_pct,
            chase=inner.chase_pct,
            hit_bar_ts=hit_ts,
        )
    return _result(REASON_NO_OPEN_CROSS, signal_px=lvl)


def hot_cross_fill_price(
    rail: float,
    opened: float,
    high: float,
    low: float,
    close: float,
    *,
    fill_mode: str = DEFAULT_HOT_CROSS_FILL,
    lerp: float = DEFAULT_HOT_CROSS_LERP,
) -> tuple:
    """Fill on the 15m bar that first traded through ``rail``.

    Returns ``(fill_px, gap_15m)``. ``gap_15m`` is ``open >= rail``.
    ``lerp85``: ``rail + lerp * (close - rail)`` when close >= rail, else rail,
    then clamp to ``[low, high]``. Does not lift a large-gap lerp up to open
    beyond that clamp (a gap with no lower wick already clamps to open).
    ``rail``: rail, or open if gapped. ``close``: that 15m close.
    """
    mode = normalize_hot_cross_fill(fill_mode)
    rail_f = float(rail)
    opened_f = float(opened)
    high_f = float(high)
    low_f = float(low)
    close_f = float(close)
    lo = min(low_f, high_f)
    hi = max(low_f, high_f)
    gap = opened_f >= rail_f
    if mode == HOT_CROSS_FILL_CLOSE:
        px = close_f
    elif mode == HOT_CROSS_FILL_RAIL:
        px = opened_f if gap else rail_f
    else:
        if close_f >= rail_f:
            px = rail_f + float(lerp) * (close_f - rail_f)
        else:
            px = rail_f
    px = min(max(px, lo), hi)
    return float(px), bool(gap)


def index_rth_15m_by_session(
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    naive_tz: str = "UTC",
) -> dict:
    """RTH 15m bars grouped by ET session date (coerce once per symbol)."""
    keyed = _coerce_15m_bars(bars_15m, naive_tz=naive_tz)
    out: dict = {}
    for bar in keyed:
        out.setdefault(bar["ts"].date(), []).append(bar)
    return out


def _purchase_hot_cross_session_bars(
    lvl: float,
    bars: Sequence[BarLike],
    *,
    fill_mode: str,
    lerp: float,
    naive_tz: str,
) -> PurchaseResult:
    mode = normalize_hot_cross_fill(fill_mode)
    if not bars:
        return _result(REASON_BAD_SIGNAL, signal_px=lvl)
    for bar in bars:
        opened = _px(bar, "open")
        high = _px(bar, "high")
        low = _px(bar, "low")
        close_px = _px(bar, "close")
        if opened is None or close_px is None or not _hl_ok(high, low):
            continue
        if float(high) < lvl:
            continue
        fill_px, gap = hot_cross_fill_price(
            lvl,
            float(opened),
            float(high),
            float(low),
            float(close_px),
            fill_mode=mode,
            lerp=lerp,
        )
        hit_ts = bar.get("ts")
        start = _floor_15m(as_et(hit_ts, naive_tz=naive_tz)) if hit_ts is not None else None
        return _result(
            REASON_FILLED,
            filled=True,
            fill_px=float(fill_px),
            signal_px=lvl,
            exec_bar_ts=start,
            mid=bar_mid(float(high), float(low)),
            hit_bar_ts=start,
            gap_15m=bool(gap),
            bar_open=float(opened),
            bar_high=float(high),
            bar_low=float(low),
            bar_close=float(close_px),
        )
    return _result(REASON_NO_HOT_CROSS, signal_px=lvl)


def purchase_hot_cross_15m(
    resist: float,
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    session_date: Any = None,
    fill_mode: str = DEFAULT_HOT_CROSS_FILL,
    lerp: float = DEFAULT_HOT_CROSS_LERP,
    naive_tz: str = "UTC",
    session_index: Optional[dict] = None,
) -> PurchaseResult:
    """First RTH 15m whose **high** reaches resist; buy-now fill on that bar.

    Does not require a daily close above resist. Last RTH bar is allowed.
    Pass ``session_index`` from ``index_rth_15m_by_session`` to avoid recoercing.
    """
    try:
        lvl = float(resist)
    except (TypeError, ValueError):
        return _result(REASON_BAD_SIGNAL)
    if lvl != lvl or lvl <= 0:
        return _result(REASON_BAD_SIGNAL)
    day = _as_session_date(session_date, naive_tz=naive_tz)
    if session_index is not None:
        bars = session_index.get(day, []) if day is not None else []
        if day is None:
            bars = [b for rows in session_index.values() for b in rows]
    else:
        bars = _coerce_15m_bars(bars_15m, naive_tz=naive_tz)
        if day is not None:
            bars = [b for b in bars if b["ts"].date() == day]
    return _purchase_hot_cross_session_bars(
        lvl,
        bars,
        fill_mode=fill_mode,
        lerp=lerp,
        naive_tz=naive_tz,
    )


def _session_bars(
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    session_date: Any = None,
    naive_tz: str = "UTC",
    session_index: Optional[dict] = None,
) -> List[dict]:
    day = _as_session_date(session_date, naive_tz=naive_tz)
    if session_index is not None:
        bars = session_index.get(day, []) if day is not None else []
        if day is None:
            bars = [b for rows in session_index.values() for b in rows]
        return list(bars)
    bars = _coerce_15m_bars(bars_15m, naive_tz=naive_tz)
    if day is not None:
        bars = [b for b in bars if b["ts"].date() == day]
    return bars


def purchase_close_cross_15m(
    resist: float,
    bars_15m: Union[pd.DataFrame, Sequence[BarLike], None],
    *,
    session_date: Any = None,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
    session_index: Optional[dict] = None,
    fill_mode: str = FILL_MODE_NEXT_MID,
) -> PurchaseResult:
    """First RTH 15m whose **close** is above resist; fill by ``fill_mode``.

    Does not wait for a daily close. A high through the rail with close still
    below is not a signal (unlike hot-cross).

    ``signal-close``: buy the confirm bar close (last RTH allowed).
    ``next-mid`` / ``next-open``: buy the next same-session mid/open; 15:45
    confirm cancels (``end_of_session``). Default remains ``next-mid``.
    """
    try:
        lvl = float(resist)
    except (TypeError, ValueError):
        return _result(REASON_BAD_SIGNAL)
    if lvl != lvl or lvl <= 0:
        return _result(REASON_BAD_SIGNAL)
    mode = normalize_fill_mode(fill_mode)
    bars = _session_bars(
        bars_15m,
        session_date=session_date,
        naive_tz=naive_tz,
        session_index=session_index,
    )
    if not bars:
        return _result(REASON_BAD_SIGNAL, signal_px=lvl)
    for i, bar in enumerate(bars):
        close_px = _px(bar, "close")
        opened = _px(bar, "open")
        high = _px(bar, "high")
        low = _px(bar, "low")
        if close_px is None or opened is None or not _hl_ok(high, low):
            continue
        if float(close_px) <= lvl:
            continue
        next_bar = bars[i + 1] if i + 1 < len(bars) else None
        if mode in (FILL_MODE_SIGNAL_CLOSE, FILL_MODE_OPEN_CROSS):
            inner = purchase_at_signal_close(bar, naive_tz=naive_tz)
        elif mode == FILL_MODE_NEXT_OPEN:
            inner = purchase_next_bar_open(bar, next_bar, naive_tz=naive_tz)
        else:
            inner = purchase_after_close_signal(
                bar,
                next_bar,
                max_low_to_mid_pct=max_low_to_mid_pct,
                max_chase_pct=max_chase_pct,
                naive_tz=naive_tz,
            )
        hit_ts = bar.get("ts")
        start = _floor_15m(as_et(hit_ts, naive_tz=naive_tz)) if hit_ts is not None else None
        gap = float(opened) >= lvl
        if not inner.filled:
            return _result(
                inner.reason,
                signal_px=lvl,
                signal_time=inner.signal_time,
                exec_bar_ts=inner.exec_bar_ts,
                hit_bar_ts=start,
                gap_15m=bool(gap),
                bar_open=float(opened),
                bar_high=float(high),
                bar_low=float(low),
                bar_close=float(close_px),
            )
        return _result(
            REASON_FILLED,
            filled=True,
            fill_px=float(inner.fill_px) if inner.fill_px is not None else None,
            signal_time=inner.signal_time,
            signal_px=lvl,
            exec_bar_ts=inner.exec_bar_ts,
            mid=inner.mid,
            low_to_mid=inner.low_to_mid_pct,
            chase=inner.chase_pct,
            hit_bar_ts=start,
            gap_15m=bool(gap),
            bar_open=float(opened),
            bar_high=float(high),
            bar_low=float(low),
            bar_close=float(close_px),
        )
    return _result(REASON_NO_CLOSE_CROSS, signal_px=lvl)


def exec_fill_15m_after_signal(
    df: pd.DataFrame,
    signal_i: int,
    *,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
    fill_mode: str = DEFAULT_FILL_MODE,
) -> Optional[tuple]:
    """Realistic 15m fill at ``signal_i``, or None.

    Default ``signal-close`` stays on the signal bar at its close.
    ``next-mid`` returns (signal_i + 1, next bar mid) as before.
    ``next-open`` returns (signal_i + 1, next bar open).
    """
    mode = normalize_fill_mode(fill_mode)
    got = purchase_at_signal_index(
        df,
        signal_i,
        max_low_to_mid_pct=max_low_to_mid_pct,
        max_chase_pct=max_chase_pct,
        naive_tz=naive_tz,
        fill_mode=mode,
    )
    if not got.filled or got.fill_px is None:
        return None
    if mode in (FILL_MODE_SIGNAL_CLOSE, FILL_MODE_OPEN_CROSS):
        return int(signal_i), float(got.fill_px)
    return int(signal_i) + 1, float(got.fill_px)


def exec_fill_daily_with_15m(
    signal_buy_price: float,
    df_15m: Optional[pd.DataFrame],
    session_date: Any,
    *,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    naive_tz: str = "UTC",
    fill_mode: str = DEFAULT_FILL_MODE,
    resist: Optional[float] = None,
    open_cross_error_pct: float = 0.0,
) -> Optional[float]:
    """Daily X: default fill at the 15m close that printed X; ``next-mid`` blends.

    ``open-cross`` ignores X and fills at the close of the first 15m that
    opens above ``resist`` (falls back to X if resist is omitted).
    """
    if df_15m is None or df_15m.empty:
        return None
    mode = normalize_fill_mode(fill_mode)
    if mode == FILL_MODE_OPEN_CROSS:
        lvl = float(resist) if resist is not None else float(signal_buy_price)
        got = purchase_open_cross_15m(
            lvl,
            df_15m,
            session_date=session_date,
            error_pct=open_cross_error_pct,
            naive_tz=naive_tz,
        )
    else:
        got = purchase_after_daily_signal(
            signal_buy_price,
            df_15m,
            session_date=session_date,
            max_low_to_mid_pct=max_low_to_mid_pct,
            max_chase_pct=max_chase_pct,
            naive_tz=naive_tz,
            fill_mode=mode,
        )
    if not got.filled or got.fill_px is None:
        return None
    return float(got.fill_px)
