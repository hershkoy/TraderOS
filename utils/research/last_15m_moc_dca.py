"""3-day MOC dollar-cost-average on the last-15m H2 book.

Signal stays last-RTH 15m open-above-rail (current_best 1d). Instead of buying
the full size at the signal mid, split the allocated notional into ``n_tranches``
equal parts and buy each at the session close (MOC):

  - Day 0 = signal session close, then the next n-1 RTH sessions.
  - Skip (pause) a tranche when that session's close is below the projected
    resist rail; unfilled cash stays idle (does not chase outside the window).
  - Once any shares are held, ATR k=2 hard-stop (from first fill) + 10% trail
    can close the position during the scale-in window (daily low tags stop;
    fill at the stop price). Remaining unfilled tranches are cancelled.
  - After the window (or early stop), remaining shares exit via 15m N+1 mid
    ATR/trail from the last fill timestamp (same walker as the baseline book).

P&L primary metric is return on **allocated** capital (idle cash = 0%), so a
1/3 fill that stops out is not inflated vs baseline full size. Occupancy is
not re-walked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from utils.research.last_15m_volume_geometry import (
    _num,
    _resist_at_offset,
    atr_dollars_for_row,
    fill_day_et,
)
from utils.research.realistic_exits import (
    ExitResult,
    REASON_EOD,
    REASON_NO_ENTRY_BAR,
    REASON_NO_NEXT_BAR,
    REASON_NO_15M,
    _eod_last_close,
    _fill_next_mid,
    _ohlc,
    find_bar_index,
    flatten_rth_sessions,
)
from utils.research.realistic_purchaser import as_et
from utils.scanning.channel_touch_bought import (
    ATR_STOP_MULT,
    TRAIL_PCT,
    current_stop_price,
    hard_stop_price,
)

ET = ZoneInfo("America/New_York")

N_TRANCHES_DEFAULT = 3
FRICTION_PCT = 0.25

REASON_OK = "ok"
REASON_NO_SIGNAL_DAY = "no_signal_day"
REASON_NO_DAILY = "no_daily"
REASON_NO_RAIL = "no_rail"
REASON_NO_FILLS = "no_fills"
REASON_NO_15M = "no_15m"
REASON_EXIT_FAIL = "exit_fail"

EXIT_SCALEIN_STOP = "scalein_stop"
EXIT_SCALEIN_TRAIL = "scalein_trail"


@dataclass
class TrancheFill:
    session_day: date
    px: float
    notional: float
    shares: float
    resist: float
    paused: bool = False


@dataclass
class MocDcaResult:
    filled: bool
    reason: str
    n_tranches_planned: int
    n_tranches_filled: int = 0
    n_paused: int = 0
    avg_cost: Optional[float] = None
    first_px: Optional[float] = None
    last_fill_day: Optional[date] = None
    last_fill_ts: Optional[datetime] = None
    entry_ts: Optional[datetime] = None
    sell_px: Optional[float] = None
    sell_ts: Optional[datetime] = None
    exit_reason: Optional[str] = None
    hard_stop: Optional[float] = None
    peak_px: Optional[float] = None
    gain_allocated_pct: Optional[float] = None
    gain_deployed_pct: Optional[float] = None
    capital_deployed: float = 0.0
    tranches: List[TrancheFill] = field(default_factory=list)


def resist_at_session(
    row: pd.Series,
    *,
    bars_after_fill: int,
) -> Optional[float]:
    """Projected resist at signal day + ``bars_after_fill`` daily bars."""
    signal_x = _num(row, "signal_x")
    slope_pct = _num(row, "slope_pct_per_bar")
    l1 = _num(row, "l1_price")
    slope = 0.0
    if slope_pct is not None and l1 is not None and l1 > 0:
        slope = (slope_pct / 100.0) * l1
    if signal_x is not None and signal_x > 0:
        return float(signal_x + slope * float(bars_after_fill))
    return _resist_at_offset(row, bars_after_fill=bars_after_fill)


def _session_ohlc(panel_1d: pd.DataFrame, day: date) -> Optional[Dict[str, float]]:
    if panel_1d is None or panel_1d.empty:
        return None
    idx = panel_1d.index
    # Daily bars are usually midnight UTC / naive date.
    for ts, row in panel_1d.iterrows():
        try:
            if hasattr(ts, "tz_convert"):
                d = ts.tz_convert("UTC").date() if ts.tzinfo is not None else ts.date()
            else:
                d = pd.Timestamp(ts).date()
        except (TypeError, ValueError, AttributeError):
            continue
        if d != day:
            continue
        try:
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
        except (KeyError, TypeError, ValueError):
            return None
        if min(o, h, l, c) != min(o, h, l, c):
            return None
        return {"open": o, "high": h, "low": l, "close": c}
    return None


def _daily_sessions_from(panel_1d: pd.DataFrame, start: date) -> List[date]:
    days: List[date] = []
    if panel_1d is None or panel_1d.empty:
        return days
    for ts in panel_1d.index:
        try:
            if hasattr(ts, "tz_convert") and ts.tzinfo is not None:
                d = ts.tz_convert("UTC").date()
            else:
                d = pd.Timestamp(ts).date()
        except (TypeError, ValueError, AttributeError):
            continue
        if d >= start:
            days.append(d)
    return sorted(set(days))


def _last_15m_ts(by_day: Dict[date, Sequence[dict]], day: date) -> Optional[datetime]:
    bars = list(by_day.get(day) or [])
    if not bars:
        return None
    best: Optional[datetime] = None
    for bar in bars:
        ts = bar.get("ts")
        if ts is None:
            continue
        try:
            et = as_et(ts)
        except (TypeError, ValueError):
            continue
        if best is None or et > best:
            best = et
    return best


def _moc_entry_ts(by_day: Dict[date, Sequence[dict]], day: date) -> datetime:
    """Prefer last RTH 15m bar; else synthesize 16:00 ET."""
    last = _last_15m_ts(by_day, day)
    if last is not None:
        return last
    return datetime.combine(day, time(16, 0), tzinfo=ET)


def _simulate_exit_15m_seeded(
    bars: Sequence[dict],
    *,
    entry_ts: Any,
    entry_px: float,
    atr_at_entry: Optional[float],
    initial_peak: Optional[float] = None,
    trail_pct: float = TRAIL_PCT,
    atr_stop_mult: float = ATR_STOP_MULT,
) -> ExitResult:
    """Same as 15m N+1 mid walker, but trail peak can start above entry."""
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
    )
    peak = max(px, float(initial_peak) if initial_peak is not None else px)
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


def apply_moc_dca(
    row: pd.Series,
    *,
    panel_1d: Optional[pd.DataFrame],
    by_day_15m: Optional[Dict[date, Sequence[dict]]],
    n_tranches: int = N_TRANCHES_DEFAULT,
    trail_pct: float = TRAIL_PCT,
    atr_stop_mult: float = ATR_STOP_MULT,
) -> MocDcaResult:
    """Scale into one trade with equal-dollar MOC tranches."""
    n = max(1, int(n_tranches))
    fill_day = fill_day_et(row)
    if fill_day is None:
        return MocDcaResult(filled=False, reason=REASON_NO_SIGNAL_DAY, n_tranches_planned=n)
    if panel_1d is None or panel_1d.empty:
        return MocDcaResult(filled=False, reason=REASON_NO_DAILY, n_tranches_planned=n)

    sessions = _daily_sessions_from(panel_1d, fill_day)
    if not sessions:
        return MocDcaResult(filled=False, reason=REASON_NO_DAILY, n_tranches_planned=n)

    window = sessions[:n]
    tranche_notional = 1.0 / float(n)
    allocated = 1.0

    atr = atr_dollars_for_row(row, float(_num(row, "buy_price") or 0.0) or 1.0)
    shares = 0.0
    cost = 0.0
    first_px: Optional[float] = None
    hard: Optional[float] = None
    peak: Optional[float] = None
    last_fill_day: Optional[date] = None
    last_fill_ts: Optional[datetime] = None
    tranches: List[TrancheFill] = []
    n_paused = 0
    by_day = by_day_15m or {}

    for offset, day in enumerate(window):
        resist = resist_at_session(row, bars_after_fill=offset)
        if resist is None:
            return MocDcaResult(
                filled=False,
                reason=REASON_NO_RAIL,
                n_tranches_planned=n,
                tranches=tranches,
            )
        ohlc = _session_ohlc(panel_1d, day)
        if ohlc is None:
            continue

        # Stop check on already-held shares before / on this session (intraday low).
        if shares > 0 and hard is not None and peak is not None:
            peak = max(peak, ohlc["high"])
            stop = current_stop_price(hard, peak, trail_pct)
            if ohlc["low"] <= stop + 1e-12:
                sell_px = float(stop)
                gain_alloc = (shares * sell_px - cost) / allocated * 100.0
                gain_dep = (shares * sell_px - cost) / cost * 100.0 if cost > 0 else None
                exit_reason = (
                    EXIT_SCALEIN_STOP
                    if stop <= hard + 1e-9
                    else EXIT_SCALEIN_TRAIL
                )
                return MocDcaResult(
                    filled=True,
                    reason=REASON_OK,
                    n_tranches_planned=n,
                    n_tranches_filled=len([t for t in tranches if not t.paused]),
                    n_paused=n_paused,
                    avg_cost=cost / shares if shares > 0 else None,
                    first_px=first_px,
                    last_fill_day=last_fill_day,
                    last_fill_ts=last_fill_ts,
                    entry_ts=last_fill_ts,
                    sell_px=sell_px,
                    sell_ts=_moc_entry_ts(by_day, day),
                    exit_reason=exit_reason,
                    hard_stop=hard,
                    peak_px=peak,
                    gain_allocated_pct=round(gain_alloc, 4),
                    gain_deployed_pct=None if gain_dep is None else round(gain_dep, 4),
                    capital_deployed=cost,
                    tranches=tranches,
                )

        close_px = ohlc["close"]
        if close_px < resist - 1e-12:
            n_paused += 1
            tranches.append(
                TrancheFill(
                    session_day=day,
                    px=close_px,
                    notional=0.0,
                    shares=0.0,
                    resist=float(resist),
                    paused=True,
                )
            )
            continue

        # MOC buy
        fill_shares = tranche_notional / close_px
        shares += fill_shares
        cost += tranche_notional
        entry_ts = _moc_entry_ts(by_day, day)
        if first_px is None:
            first_px = close_px
            hard = hard_stop_price(
                first_px,
                atr_at_entry=atr,
                atr_stop_mult=atr_stop_mult,
            )
            peak = max(close_px, ohlc["high"])
        else:
            peak = max(peak or close_px, ohlc["high"], close_px)
        last_fill_day = day
        last_fill_ts = entry_ts
        tranches.append(
            TrancheFill(
                session_day=day,
                px=close_px,
                notional=tranche_notional,
                shares=fill_shares,
                resist=float(resist),
                paused=False,
            )
        )

        # Same-session stop after the MOC is not actionable (close already printed).
        # Next session will check the low.

    n_filled = len([t for t in tranches if not t.paused and t.shares > 0])
    if shares <= 0 or first_px is None or last_fill_ts is None or hard is None:
        return MocDcaResult(
            filled=False,
            reason=REASON_NO_FILLS,
            n_tranches_planned=n,
            n_tranches_filled=0,
            n_paused=n_paused,
            tranches=tranches,
        )

    avg_cost = cost / shares
    if not by_day:
        return MocDcaResult(
            filled=False,
            reason=REASON_NO_15M,
            n_tranches_planned=n,
            n_tranches_filled=n_filled,
            n_paused=n_paused,
            avg_cost=avg_cost,
            first_px=first_px,
            last_fill_day=last_fill_day,
            last_fill_ts=last_fill_ts,
            capital_deployed=cost,
            tranches=tranches,
        )

    bars = flatten_rth_sessions(by_day)
    # Stop distance from first fill; trail peak continues from scale-in peak.
    exit_res = _simulate_exit_15m_seeded(
        bars,
        entry_ts=last_fill_ts,
        entry_px=float(first_px),
        atr_at_entry=atr,
        initial_peak=peak,
        trail_pct=trail_pct,
        atr_stop_mult=atr_stop_mult,
    )
    if not exit_res.filled or exit_res.sell_px is None:
        return MocDcaResult(
            filled=False,
            reason=REASON_EXIT_FAIL,
            n_tranches_planned=n,
            n_tranches_filled=n_filled,
            n_paused=n_paused,
            avg_cost=avg_cost,
            first_px=first_px,
            last_fill_day=last_fill_day,
            last_fill_ts=last_fill_ts,
            hard_stop=hard,
            peak_px=peak,
            capital_deployed=cost,
            tranches=tranches,
        )

    sell_px = float(exit_res.sell_px)
    # Carry forward scale-in peak into exit walker peak when available.
    gain_alloc = (shares * sell_px - cost) / allocated * 100.0
    gain_dep = (shares * sell_px - cost) / cost * 100.0
    return MocDcaResult(
        filled=True,
        reason=REASON_OK,
        n_tranches_planned=n,
        n_tranches_filled=n_filled,
        n_paused=n_paused,
        avg_cost=round(avg_cost, 6),
        first_px=first_px,
        last_fill_day=last_fill_day,
        last_fill_ts=last_fill_ts,
        entry_ts=last_fill_ts,
        sell_px=sell_px,
        sell_ts=exit_res.exec_bar_ts,
        exit_reason=exit_res.exit_reason,
        hard_stop=hard,
        peak_px=exit_res.peak_px if exit_res.peak_px is not None else peak,
        gain_allocated_pct=round(gain_alloc, 4),
        gain_deployed_pct=round(gain_dep, 4),
        capital_deployed=cost,
        tranches=tranches,
    )


def friction_gain(gain_pct: float, *, friction_pct: float = FRICTION_PCT) -> float:
    """Round-trip friction on deployed capital only (matches book overlays)."""
    return float(gain_pct) - 2.0 * float(friction_pct)
