"""Realistic sells after a 1d last-RTH (or other 15m) fill.

Daily occupancy fills the stop on the same bar whose low tagged it. After a
16:00 ET buy you cannot go back and sell that stop on the next daily candle.
These walkers delay the fill until a price that exists after the decision:

- ``15m-next-mid``: track ATR k=2 + 10% trail on RTH 15m. Decision on bar N
  (low <= stop at that bar's close). Fill the **next** RTH 15m mid (overnight
  to the next session's 09:30 is allowed — you already have a position).
- ``daily-close-next-open-mid``: same stop on **session** high/low. Decision at
  that session's close (16:00 ET). Fill the next session's 09:30 ET 15m mid.

Peak starts at the entry price. The entry bar/session is not used for the
stop check (same idea as ``skip_entry_bar_stop``). Hard-stop distance is the
daily ATR k=2 clamp (1.5%–6%), not 15m ATR.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

from utils.research.realistic_purchaser import as_et, bar_mid
from utils.scanning.channel_touch_bought import (
    ATR_STOP_MULT,
    STOP_PCT,
    STOP_PCT_CEIL,
    STOP_PCT_FLOOR,
    TRAIL_PCT,
    current_stop_price,
    hard_stop_price,
)

EXIT_MODE_15M_NEXT_MID = "15m-next-mid"
EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID = "daily-close-next-open-mid"
EXIT_MODES = (EXIT_MODE_15M_NEXT_MID, EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID)

REASON_FILLED = "filled"
REASON_NO_15M = "no_15m"
REASON_NO_ENTRY_BAR = "no_entry_bar"
REASON_NO_NEXT_BAR = "no_next_bar"
REASON_EOD = "eod"

EXIT_HARD_STOP = "hard_stop"
EXIT_TRAIL_STOP = "trail_stop"
EXIT_EOD = "eod"


@dataclass
class ExitResult:
    filled: bool
    reason: str
    sell_px: Optional[float] = None
    decision_ts: Optional[datetime] = None
    exec_bar_ts: Optional[datetime] = None
    exit_reason: Optional[str] = None
    peak_px: Optional[float] = None
    hard_stop: Optional[float] = None
    stop_at_decision: Optional[float] = None


def normalize_exit_mode(raw: Optional[str]) -> str:
    text = str(raw or EXIT_MODE_15M_NEXT_MID).strip().lower().replace("_", "-")
    if text in (
        EXIT_MODE_15M_NEXT_MID,
        "15m",
        "intraday",
        "next-mid",
        "bar-n-next-mid",
    ):
        return EXIT_MODE_15M_NEXT_MID
    if text in (
        EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID,
        "daily",
        "eod",
        "close-next-open-mid",
        "next-open-mid",
        "daily-close",
    ):
        return EXIT_MODE_DAILY_CLOSE_NEXT_OPEN_MID
    raise ValueError("unknown realistic exit mode: %s" % raw)


def flatten_rth_sessions(
    by_day: Dict[date, Sequence[dict]],
) -> List[dict]:
    """Chronological RTH 15m bars from a session-date index."""
    bars: List[dict] = []
    for day in sorted(by_day.keys()):
        day_bars = list(by_day.get(day) or [])
        day_bars.sort(key=lambda b: as_et(b["ts"]))
        bars.extend(day_bars)
    return bars


def find_bar_index(
    bars: Sequence[dict],
    ts: Any,
) -> Optional[int]:
    if ts is None or not bars:
        return None
    want = _minute_et(ts)
    if want is None:
        return None
    for i, bar in enumerate(bars):
        got = _minute_et(bar.get("ts"))
        if got is not None and got == want:
            return i
    return None


def atr_dollars(
    *,
    atr_pct: Any = None,
    ref_px: Any = None,
) -> Optional[float]:
    """ATR in dollars from a stored atr_pct (ATR / ref_px * 100)."""
    try:
        pct = float(atr_pct)
        px = float(ref_px)
    except (TypeError, ValueError):
        return None
    if pct != pct or px != px or pct <= 0 or px <= 0:
        return None
    return px * pct / 100.0


def simulate_exit(
    by_day: Dict[date, Sequence[dict]],
    *,
    entry_ts: Any,
    entry_px: float,
    mode: str,
    atr_at_entry: Optional[float] = None,
    trail_pct: float = TRAIL_PCT,
    atr_stop_mult: float = ATR_STOP_MULT,
    stop_pct: float = STOP_PCT,
    stop_pct_floor: float = STOP_PCT_FLOOR,
    stop_pct_ceil: float = STOP_PCT_CEIL,
) -> ExitResult:
    mode_n = normalize_exit_mode(mode)
    if mode_n == EXIT_MODE_15M_NEXT_MID:
        return simulate_exit_15m_next_mid(
            flatten_rth_sessions(by_day),
            entry_ts=entry_ts,
            entry_px=entry_px,
            atr_at_entry=atr_at_entry,
            trail_pct=trail_pct,
            atr_stop_mult=atr_stop_mult,
            stop_pct=stop_pct,
            stop_pct_floor=stop_pct_floor,
            stop_pct_ceil=stop_pct_ceil,
        )
    return simulate_exit_daily_close_next_open_mid(
        by_day,
        entry_ts=entry_ts,
        entry_px=entry_px,
        atr_at_entry=atr_at_entry,
        trail_pct=trail_pct,
        atr_stop_mult=atr_stop_mult,
        stop_pct=stop_pct,
        stop_pct_floor=stop_pct_floor,
        stop_pct_ceil=stop_pct_ceil,
    )


def simulate_exit_15m_next_mid(
    bars: Sequence[dict],
    *,
    entry_ts: Any,
    entry_px: float,
    atr_at_entry: Optional[float] = None,
    trail_pct: float = TRAIL_PCT,
    atr_stop_mult: float = ATR_STOP_MULT,
    stop_pct: float = STOP_PCT,
    stop_pct_floor: float = STOP_PCT_FLOOR,
    stop_pct_ceil: float = STOP_PCT_CEIL,
) -> ExitResult:
    """Decision on 15m bar N; fill mid of N+1 (overnight OK)."""
    px = float(entry_px)
    if not bars:
        return ExitResult(filled=False, reason=REASON_NO_15M)
    if px != px or px <= 0:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    entry_i = find_bar_index(bars, entry_ts)
    if entry_i is None:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    hard = hard_stop_price(
        px,
        atr_at_entry=atr_at_entry,
        atr_stop_mult=atr_stop_mult,
        stop_pct=stop_pct,
        stop_pct_floor=stop_pct_floor,
        stop_pct_ceil=stop_pct_ceil,
    )
    peak = px
    last_i = len(bars) - 1
    if entry_i >= last_i:
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            hard_stop=hard,
            peak_px=peak,
        )
    for i in range(entry_i + 1, len(bars)):
        hi = _ohlc(bars[i], "high")
        lo = _ohlc(bars[i], "low")
        if hi is not None:
            peak = max(peak, hi)
        stop = current_stop_price(hard, peak, trail_pct)
        if lo is None or lo > stop + 1e-12:
            continue
        return _fill_next_mid(
            bars,
            decision_i=i,
            peak=peak,
            hard=hard,
            stop=stop,
        )
    return _eod_last_close(bars, peak=peak, hard=hard)


def simulate_exit_daily_close_next_open_mid(
    by_day: Dict[date, Sequence[dict]],
    *,
    entry_ts: Any,
    entry_px: float,
    atr_at_entry: Optional[float] = None,
    trail_pct: float = TRAIL_PCT,
    atr_stop_mult: float = ATR_STOP_MULT,
    stop_pct: float = STOP_PCT,
    stop_pct_floor: float = STOP_PCT_FLOOR,
    stop_pct_ceil: float = STOP_PCT_CEIL,
) -> ExitResult:
    """Decision at session close; fill next session 09:30 15m mid."""
    px = float(entry_px)
    if not by_day:
        return ExitResult(filled=False, reason=REASON_NO_15M)
    if px != px or px <= 0:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    entry_et = _minute_et(entry_ts)
    if entry_et is None:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    entry_day = entry_et.date()
    days = sorted(d for d, bars in by_day.items() if bars)
    if entry_day not in days:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    hard = hard_stop_price(
        px,
        atr_at_entry=atr_at_entry,
        atr_stop_mult=atr_stop_mult,
        stop_pct=stop_pct,
        stop_pct_floor=stop_pct_floor,
        stop_pct_ceil=stop_pct_ceil,
    )
    peak = px
    after = [d for d in days if d > entry_day]
    if not after:
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            hard_stop=hard,
            peak_px=peak,
        )
    for i, day in enumerate(after):
        sess = _session_ohlc(by_day[day])
        if sess is None:
            continue
        sess_high, sess_low, last_bar = sess
        peak = max(peak, sess_high)
        stop = current_stop_price(hard, peak, trail_pct)
        if sess_low > stop + 1e-12:
            continue
        nxt_day = _next_session(after, i)
        if nxt_day is None:
            return ExitResult(
                filled=False,
                reason=REASON_NO_NEXT_BAR,
                decision_ts=last_bar.get("ts"),
                exit_reason=(
                    EXIT_TRAIL_STOP if stop > hard + 1e-9 else EXIT_HARD_STOP
                ),
                peak_px=peak,
                hard_stop=hard,
                stop_at_decision=stop,
            )
        fill_bar = _first_bar(by_day.get(nxt_day) or [])
        if fill_bar is None:
            return ExitResult(
                filled=False,
                reason=REASON_NO_NEXT_BAR,
                decision_ts=last_bar.get("ts"),
                peak_px=peak,
                hard_stop=hard,
                stop_at_decision=stop,
            )
        hi = _ohlc(fill_bar, "high")
        lo = _ohlc(fill_bar, "low")
        if hi is None or lo is None:
            return ExitResult(
                filled=False,
                reason=REASON_NO_NEXT_BAR,
                decision_ts=last_bar.get("ts"),
                peak_px=peak,
                hard_stop=hard,
            )
        mid = bar_mid(hi, lo)
        return ExitResult(
            filled=True,
            reason=REASON_FILLED,
            sell_px=float(mid),
            decision_ts=last_bar.get("ts"),
            exec_bar_ts=fill_bar.get("ts"),
            exit_reason=EXIT_TRAIL_STOP if stop > hard + 1e-9 else EXIT_HARD_STOP,
            peak_px=peak,
            hard_stop=hard,
            stop_at_decision=stop,
        )
    last_day = after[-1]
    last_sess = _session_ohlc(by_day[last_day])
    if last_sess is None:
        return ExitResult(
            filled=False,
            reason=REASON_EOD,
            peak_px=peak,
            hard_stop=hard,
        )
    _h, _l, last_bar = last_sess
    cl = _ohlc(last_bar, "close")
    if cl is None:
        return ExitResult(
            filled=False,
            reason=REASON_EOD,
            peak_px=peak,
            hard_stop=hard,
        )
    return ExitResult(
        filled=True,
        reason=REASON_EOD,
        sell_px=float(cl),
        decision_ts=last_bar.get("ts"),
        exec_bar_ts=last_bar.get("ts"),
        exit_reason=EXIT_EOD,
        peak_px=peak,
        hard_stop=hard,
        stop_at_decision=current_stop_price(hard, peak, trail_pct),
    )


def _fill_next_mid(
    bars: Sequence[dict],
    *,
    decision_i: int,
    peak: float,
    hard: float,
    stop: float,
) -> ExitResult:
    nxt_i = int(decision_i) + 1
    if nxt_i >= len(bars):
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            decision_ts=bars[decision_i].get("ts"),
            exit_reason=EXIT_TRAIL_STOP if stop > hard + 1e-9 else EXIT_HARD_STOP,
            peak_px=peak,
            hard_stop=hard,
            stop_at_decision=stop,
        )
    nxt = bars[nxt_i]
    hi = _ohlc(nxt, "high")
    lo = _ohlc(nxt, "low")
    if hi is None or lo is None:
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            decision_ts=bars[decision_i].get("ts"),
            peak_px=peak,
            hard_stop=hard,
            stop_at_decision=stop,
        )
    mid = bar_mid(hi, lo)
    return ExitResult(
        filled=True,
        reason=REASON_FILLED,
        sell_px=float(mid),
        decision_ts=bars[decision_i].get("ts"),
        exec_bar_ts=nxt.get("ts"),
        exit_reason=EXIT_TRAIL_STOP if stop > hard + 1e-9 else EXIT_HARD_STOP,
        peak_px=peak,
        hard_stop=hard,
        stop_at_decision=stop,
    )


def _eod_last_close(
    bars: Sequence[dict],
    *,
    peak: float,
    hard: float,
) -> ExitResult:
    last = bars[-1]
    cl = _ohlc(last, "close")
    if cl is None:
        return ExitResult(
            filled=False,
            reason=REASON_EOD,
            peak_px=peak,
            hard_stop=hard,
        )
    return ExitResult(
        filled=True,
        reason=REASON_EOD,
        sell_px=float(cl),
        decision_ts=last.get("ts"),
        exec_bar_ts=last.get("ts"),
        exit_reason=EXIT_EOD,
        peak_px=peak,
        hard_stop=hard,
        stop_at_decision=current_stop_price(hard, peak, TRAIL_PCT),
    )


def _minute_et(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        et = as_et(ts)
    except (TypeError, ValueError):
        return None
    return et.replace(second=0, microsecond=0)


def _ohlc(bar: dict, key: str) -> Optional[float]:
    try:
        val = float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def _session_ohlc(bars: Sequence[dict]) -> Optional[tuple]:
    highs: List[float] = []
    lows: List[float] = []
    last = None
    ordered = sorted(bars, key=lambda b: as_et(b["ts"]))
    for bar in ordered:
        hi = _ohlc(bar, "high")
        lo = _ohlc(bar, "low")
        if hi is not None:
            highs.append(hi)
        if lo is not None:
            lows.append(lo)
        last = bar
    if not highs or not lows or last is None:
        return None
    return max(highs), min(lows), last


def _first_bar(bars: Sequence[dict]) -> Optional[dict]:
    if not bars:
        return None
    return sorted(bars, key=lambda b: as_et(b["ts"]))[0]


def _next_session(days: Sequence[date], i: int) -> Optional[date]:
    nxt = int(i) + 1
    if nxt >= len(days):
        return None
    return days[nxt]
