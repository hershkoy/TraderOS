"""Next-session MOO on the last-15m book, without chasing an overnight gap.

Last-15m H2 is known at 15:45/16:00 ET. This overlay waits for the next RTH
09:30 ET open:

- No overnight gap-up vs the signal session's last 15m close, and the open
  still holds at/above the projected resist rail: buy that open (MOO).
- Overnight gap-up, or the open is already below the rail: do not chase.
  Same session only, wait for a pullback toward the rail then a positive 15m
  candle that holds (low and close at/above the rail). Fill that bar's close.

Occupancy is not re-walked. Exits stay 15m N+1 mid ATR k=2 + 10% trail from
the new entry. Gap-up is tick vs prior RTH close, not 15m ``gap_15m``
(open >= rail on a trigger bar).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.research.evening_doji_star import Candle
from utils.research.last_15m_volume_geometry import (
    _resist_at_offset,
    atr_dollars_for_row,
    fill_day_et,
)
from utils.research.morning_doji_star import (
    is_bullish_engulfing,
    is_dragonfly_doji,
    is_hammer,
)
from utils.research.realistic_exits import (
    flatten_rth_sessions,
    simulate_exit_15m_next_mid,
)
from utils.research.realistic_purchaser import as_et
from utils.scanning.channel_touch_bought import ATR_STOP_MULT

POLICY_CHASE_MOO = "chase_moo"
POLICY_SKIP_GAP = "skip_gap"
POLICY_PULLBACK = "pullback"
POLICIES = (POLICY_CHASE_MOO, POLICY_SKIP_GAP, POLICY_PULLBACK)

REASON_MOO = "moo"
REASON_PULLBACK = "pullback_confirm"
REASON_NO_NEXT = "no_next_session"
REASON_NO_OPEN = "no_open_bar"
REASON_NO_PRIOR_CLOSE = "no_prior_close"
REASON_NO_RAIL = "no_rail"
REASON_NO_CONFIRM = "no_confirm"
REASON_CHASE_SKIP = "gap_skipped"
REASON_OPEN_BELOW = "open_below_rail"
REASON_CLOSED_BELOW = "closed_below_rail"

PULLBACK_RETRACE = 0.50
RAIL_HOLD_EPS = 1e-12


@dataclass
class OpenNoChaseFill:
    filled: bool
    reason: str
    policy: str
    gap_up: bool
    open_holds: bool
    entry_px: Optional[float] = None
    entry_ts: Optional[datetime] = None
    pattern: Optional[str] = None
    next_day: Optional[date] = None
    open_px: Optional[float] = None
    open_ts: Optional[datetime] = None
    prior_close: Optional[float] = None
    resist: Optional[float] = None
    gap_pct: Optional[float] = None
    sell_px: Optional[float] = None
    sell_ts: Optional[datetime] = None
    exit_reason: Optional[str] = None
    gain_pct: Optional[float] = None


def _bar_px(bar: dict, key: str) -> Optional[float]:
    try:
        val = float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def _sorted_bars(bars: Sequence[dict]) -> List[dict]:
    keyed: List[Tuple[datetime, dict]] = []
    for bar in bars:
        ts = bar.get("ts")
        if ts is None:
            continue
        try:
            et = as_et(ts)
        except (TypeError, ValueError):
            continue
        keyed.append((et, bar))
    keyed.sort(key=lambda item: item[0])
    return [bar for _et, bar in keyed]


def _candle(bar: dict, session_day: date) -> Optional[Candle]:
    opened = _bar_px(bar, "open")
    high = _bar_px(bar, "high")
    low = _bar_px(bar, "low")
    close = _bar_px(bar, "close")
    if opened is None or high is None or low is None or close is None:
        return None
    return Candle(
        session_date=session_day,
        open=opened,
        high=high,
        low=low,
        close=close,
    )


def next_rth_day(by_day: Dict[date, Sequence[dict]], fill_day: date) -> Optional[date]:
    later = [d for d, bars in by_day.items() if d > fill_day and bars]
    if not later:
        return None
    return min(later)


def last_session_close(bars: Sequence[dict]) -> Optional[float]:
    ordered = _sorted_bars(bars)
    if not ordered:
        return None
    return _bar_px(ordered[-1], "close")


def positive_15m_holding(
    curr: Candle,
    prior: Optional[Candle],
    rail: float,
) -> Optional[str]:
    """Green / hammer / engulfing whose low and close hold at/above the rail."""
    if float(curr.close) + RAIL_HOLD_EPS < float(rail):
        return None
    if float(curr.low) + RAIL_HOLD_EPS < float(rail):
        return None
    if is_hammer(curr) or is_dragonfly_doji(curr):
        return "hammer"
    if prior is not None and is_bullish_engulfing(prior, curr):
        return "engulfing"
    if float(curr.close) > float(curr.open) + RAIL_HOLD_EPS:
        return "green"
    return None


def _pullback_armed(
    *,
    low: float,
    rail: float,
    open_px: float,
    retrace: float = PULLBACK_RETRACE,
) -> bool:
    if open_px <= rail + RAIL_HOLD_EPS:
        return True
    span = open_px - rail
    level = rail + float(retrace) * span
    return low <= level + RAIL_HOLD_EPS


def _wait_pullback_fill(
    bars: Sequence[dict],
    *,
    session_day: date,
    rail: float,
    open_px: float,
    opened_below: bool,
) -> Tuple[Optional[dict], Optional[str], Optional[Candle], str]:
    """Walk 09:45..15:45. Returns (bar, pattern, candle, reason)."""
    ordered = _sorted_bars(bars)
    if len(ordered) < 2:
        return None, None, None, REASON_NO_CONFIRM
    armed = bool(opened_below)
    prior_c: Optional[Candle] = _candle(ordered[0], session_day)
    for bar in ordered[1:]:
        curr = _candle(bar, session_day)
        if curr is None:
            continue
        if (not opened_below) and float(curr.close) + RAIL_HOLD_EPS < float(rail):
            return None, None, None, REASON_CLOSED_BELOW
        if not armed:
            armed = _pullback_armed(low=float(curr.low), rail=rail, open_px=open_px)
        if armed:
            pat = positive_15m_holding(curr, prior_c, rail)
            if pat is not None:
                return bar, pat, curr, REASON_PULLBACK
        prior_c = curr
    return None, None, None, REASON_NO_CONFIRM


def plan_next_open_fill(
    by_day: Dict[date, Sequence[dict]],
    *,
    fill_day: date,
    resist: float,
    policy: str,
) -> OpenNoChaseFill:
    """Pick a next-session fill under ``policy``. Does not walk the exit."""
    pol = str(policy or POLICY_PULLBACK).strip().lower()
    if pol not in POLICIES:
        raise ValueError("unknown open-no-chase policy: %s" % policy)
    empty = OpenNoChaseFill(
        filled=False,
        reason=REASON_NO_NEXT,
        policy=pol,
        gap_up=False,
        open_holds=False,
        resist=float(resist) if resist == resist else None,
    )
    if not by_day or fill_day is None:
        return empty
    if resist != resist or resist <= 0:
        empty.reason = REASON_NO_RAIL
        empty.resist = None
        return empty
    nxt = next_rth_day(by_day, fill_day)
    if nxt is None:
        return empty
    prior_bars = list(by_day.get(fill_day) or [])
    prior_close = last_session_close(prior_bars)
    if prior_close is None or prior_close <= 0:
        empty.reason = REASON_NO_PRIOR_CLOSE
        empty.next_day = nxt
        return empty
    nxt_bars = list(by_day.get(nxt) or [])
    ordered = _sorted_bars(nxt_bars)
    if not ordered:
        empty.reason = REASON_NO_OPEN
        empty.next_day = nxt
        empty.prior_close = float(prior_close)
        return empty
    open_bar = ordered[0]
    open_px = _bar_px(open_bar, "open")
    open_ts = open_bar.get("ts")
    if open_px is None or open_px <= 0:
        empty.reason = REASON_NO_OPEN
        empty.next_day = nxt
        empty.prior_close = float(prior_close)
        return empty
    gap_up = float(open_px) > float(prior_close) + RAIL_HOLD_EPS
    open_holds = float(open_px) + RAIL_HOLD_EPS >= float(resist)
    gap_pct = (float(open_px) / float(prior_close) - 1.0) * 100.0
    base = OpenNoChaseFill(
        filled=False,
        reason=REASON_NO_CONFIRM,
        policy=pol,
        gap_up=gap_up,
        open_holds=open_holds,
        next_day=nxt,
        open_px=float(open_px),
        open_ts=open_ts if isinstance(open_ts, datetime) else None,
        prior_close=float(prior_close),
        resist=float(resist),
        gap_pct=round(float(gap_pct), 4),
    )
    want_moo = (not gap_up) and open_holds
    if pol == POLICY_CHASE_MOO:
        want_moo = True
    if want_moo:
        base.filled = True
        base.reason = REASON_MOO
        base.entry_px = float(open_px)
        base.entry_ts = base.open_ts
        base.pattern = "moo"
        return base
    if pol == POLICY_SKIP_GAP:
        base.reason = REASON_CHASE_SKIP if gap_up else REASON_OPEN_BELOW
        return base
    opened_below = not open_holds
    bar, pat, candle, why = _wait_pullback_fill(
        ordered,
        session_day=nxt,
        rail=float(resist),
        open_px=float(open_px),
        opened_below=opened_below,
    )
    base.reason = why
    if bar is None or candle is None or pat is None:
        return base
    ts = bar.get("ts")
    base.filled = True
    base.reason = REASON_PULLBACK
    base.pattern = pat
    base.entry_px = float(candle.close)
    base.entry_ts = ts if isinstance(ts, datetime) else None
    return base


def apply_open_no_chase(
    row: Any,
    by_day: Dict[date, Sequence[dict]],
    *,
    policy: str,
) -> OpenNoChaseFill:
    """Plan the fill and reprice the 15m N+1 mid exit from the new entry."""
    fill_day = fill_day_et(row)
    resist = _resist_at_offset(row, bars_after_fill=1)
    if fill_day is None:
        return OpenNoChaseFill(
            filled=False,
            reason=REASON_NO_NEXT,
            policy=str(policy),
            gap_up=False,
            open_holds=False,
        )
    if resist is None:
        return OpenNoChaseFill(
            filled=False,
            reason=REASON_NO_RAIL,
            policy=str(policy),
            gap_up=False,
            open_holds=False,
        )
    planned = plan_next_open_fill(
        by_day,
        fill_day=fill_day,
        resist=float(resist),
        policy=policy,
    )
    if not planned.filled or planned.entry_px is None or planned.entry_ts is None:
        return planned
    bars = flatten_rth_sessions(by_day)
    atr = atr_dollars_for_row(row, float(planned.entry_px))
    got = simulate_exit_15m_next_mid(
        bars,
        entry_ts=planned.entry_ts,
        entry_px=float(planned.entry_px),
        atr_at_entry=atr,
        atr_stop_mult=ATR_STOP_MULT,
    )
    if not got.filled or got.sell_px is None:
        planned.filled = False
        planned.reason = str(got.reason or "no_exit")
        planned.exit_reason = got.exit_reason
        return planned
    planned.sell_px = float(got.sell_px)
    planned.sell_ts = got.exec_bar_ts
    planned.exit_reason = got.exit_reason
    planned.gain_pct = (float(got.sell_px) / float(planned.entry_px) - 1.0) * 100.0
    return planned
