"""Earnings flatten/skip overlays on last-15m buy-write and wheel books.

Shared policy (max_sessions=10, matching stock hygiene):

* ahead == 0 (AMC tonight): entry skip
* ahead 1..max_sessions: early exit before print when long stock
  - buy-write: stock @ last RTH mid + BTC short call
  - wheel: if still in CSP and print on/before put expiry -> entry skip
    (no put path walk); if assigned and print before share exit ->
    liquidate shares @ mid + BTC call, keep put+call credits already taken

Occupancy is not re-walked. Gain columns stay strategy-native
(buy-write = stock notional %; wheel = cash-secured %).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from utils.research.last_15m_buy_write import (
    EXIT_EARNINGS_FLATTEN as BW_EXIT_FLATTEN,
    buy_write_pnl_per_share,
    stock_notional_pct,
)
from utils.research.last_15m_earnings_event_filters import (
    EXIT_EARNINGS_FLATTEN,
    EXIT_EARNINGS_SKIP,
    EarningsEvent,
    FlattenResult,
    count_sessions_ahead,
    flatten_session_for_event,
    last_rth_mid,
    next_earnings_after_fill,
    parse_session_date,
)
from utils.research.last_15m_short_put_bs import _as_date
from utils.research.last_15m_wheel import cash_secured_pct


def _num(row: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        if name not in row:
            continue
        try:
            v = float(row[name])
        except (TypeError, ValueError):
            continue
        if v == v:
            return v
    return None


@dataclass
class OverlayOutcome:
    keep: bool
    gain_pct: Optional[float]
    reason: str
    sessions_ahead: Optional[int] = None
    flatten_session: Optional[date] = None
    used: bool = False
    skipped: bool = False
    sell_px: Optional[float] = None
    call_close_debit: Optional[float] = None


def _session_days_from_index(by_day: Mapping[date, Any]) -> List[date]:
    return sorted(by_day.keys()) if by_day else []


def _snap_bmo_flatten(
    ev: EarningsEvent,
    by_day: Mapping[date, Any],
) -> date:
    flat_day = flatten_session_for_event(ev)
    if (not ev.amc) and flat_day not in by_day:
        prior = sorted(d for d in by_day.keys() if d < ev.earn_date)
        if prior:
            return prior[-1]
    return flat_day


def earnings_window_for_row(
    row: Mapping[str, Any],
    events: Sequence[EarningsEvent],
    by_day: Mapping[date, Any],
    *,
    max_sessions: int,
) -> Optional[Tuple[EarningsEvent, date, int]]:
    """Return (event, flatten_day, ahead) if print is inside the window."""
    buy_day = parse_session_date(row.get("buy_time") or row.get("buy_date"))
    if buy_day is None:
        return None
    ev = next_earnings_after_fill(events, buy_day)
    if ev is None:
        return None
    flat_day = _snap_bmo_flatten(ev, by_day)
    ahead = count_sessions_ahead(buy_day, flat_day, _session_days_from_index(by_day))
    if ahead is None or ahead > int(max_sessions):
        return None
    return ev, flat_day, int(ahead)


def apply_buy_write_earnings(
    row: Mapping[str, Any],
    events: Sequence[EarningsEvent],
    by_day: Mapping[date, Any],
    *,
    max_sessions: int = 10,
    call_close_debit: Optional[float] = None,
) -> OverlayOutcome:
    """Overlay flatten/skip on a filled buy-write row (gain_pct already set)."""
    base = _num(row, "gain_pct")
    if base is None:
        return OverlayOutcome(keep=False, gain_pct=None, reason="no_gain")
    entry_px = _num(row, "buy_price")
    if entry_px is None or entry_px <= 0:
        return OverlayOutcome(keep=False, gain_pct=None, reason="no_entry")

    win = earnings_window_for_row(
        row, events, by_day, max_sessions=int(max_sessions)
    )
    if win is None:
        return OverlayOutcome(
            keep=True, gain_pct=float(base), reason="no_earnings_window"
        )
    _ev, flat_day, ahead = win
    exit_day = parse_session_date(
        row.get("bw_exit_date") or row.get("wheel_exit_date") or row.get("call_expiry")
    )
    if exit_day is not None and exit_day < flat_day:
        return OverlayOutcome(
            keep=True,
            gain_pct=float(base),
            reason="already_out",
            sessions_ahead=ahead,
            flatten_session=flat_day,
        )

    if ahead == 0:
        return OverlayOutcome(
            keep=False,
            gain_pct=None,
            reason=EXIT_EARNINGS_SKIP,
            sessions_ahead=ahead,
            flatten_session=flat_day,
            used=True,
            skipped=True,
        )

    mid = last_rth_mid(by_day, flat_day)
    if mid is None:
        return OverlayOutcome(
            keep=True,
            gain_pct=float(base),
            reason="no_15m_mid",
            sessions_ahead=ahead,
            flatten_session=flat_day,
        )
    exit_px, _ts = mid
    call_credit = float(_num(row, "call_credit") or 0.0)
    debit = float(call_close_debit) if call_close_debit is not None else 0.0
    # Intrinsic floor if debit missing and ITM vs flatten mid.
    strike = _num(row, "call_strike")
    if call_close_debit is None and strike is not None:
        debit = max(0.0, float(exit_px) - float(strike))

    pnl = buy_write_pnl_per_share(
        entry_px=float(entry_px),
        call_credit=call_credit,
        called_away=False,
        share_exit_px=float(exit_px),
        call_close_debit=debit,
    )
    gain = stock_notional_pct(pnl, float(entry_px))
    return OverlayOutcome(
        keep=True,
        gain_pct=gain,
        reason=BW_EXIT_FLATTEN,
        sessions_ahead=ahead,
        flatten_session=flat_day,
        used=True,
        skipped=False,
        sell_px=float(exit_px),
        call_close_debit=debit,
    )


def apply_wheel_earnings(
    row: Mapping[str, Any],
    events: Sequence[EarningsEvent],
    by_day: Mapping[date, Any],
    *,
    max_sessions: int = 10,
    call_close_debit: Optional[float] = None,
) -> OverlayOutcome:
    """Overlay flatten/skip on a filled wheel row (cash-secured gain_pct)."""
    if not bool(row.get("wheel_ok")):
        return OverlayOutcome(keep=False, gain_pct=None, reason="not_filled")
    base = _num(row, "gain_pct")
    if base is None:
        return OverlayOutcome(keep=False, gain_pct=None, reason="no_gain")

    put_strike = _num(row, "put_strike")
    put_credit = _num(row, "put_credit")
    if put_strike is None or put_credit is None or put_strike <= 0:
        return OverlayOutcome(keep=False, gain_pct=None, reason="no_put")

    win = earnings_window_for_row(
        row, events, by_day, max_sessions=int(max_sessions)
    )
    if win is None:
        return OverlayOutcome(
            keep=True, gain_pct=float(base), reason="no_earnings_window"
        )
    _ev, flat_day, ahead = win

    put_exp = _as_date(row.get("put_expiry"))
    exit_day = parse_session_date(
        row.get("wheel_exit_date") or row.get("call_expiry") or row.get("put_expiry")
    )
    assigned = bool(row.get("wheel_assigned"))

    if ahead == 0:
        return OverlayOutcome(
            keep=False,
            gain_pct=None,
            reason=EXIT_EARNINGS_SKIP,
            sessions_ahead=ahead,
            flatten_session=flat_day,
            used=True,
            skipped=True,
        )

    # Print before put expiry while still short the put: skip entry (no BTC path).
    if put_exp is not None and flat_day <= put_exp and not assigned:
        return OverlayOutcome(
            keep=False,
            gain_pct=None,
            reason="skip_csp_into_print",
            sessions_ahead=ahead,
            flatten_session=flat_day,
            used=True,
            skipped=True,
        )
    if put_exp is not None and flat_day <= put_exp and assigned:
        # Assigned on put_exp; flatten on/before that day means skip the cycle.
        if flat_day < put_exp:
            return OverlayOutcome(
                keep=False,
                gain_pct=None,
                reason="skip_csp_into_print",
                sessions_ahead=ahead,
                flatten_session=flat_day,
                used=True,
                skipped=True,
            )

    if exit_day is not None and exit_day < flat_day:
        return OverlayOutcome(
            keep=True,
            gain_pct=float(base),
            reason="already_out",
            sessions_ahead=ahead,
            flatten_session=flat_day,
        )

    # Share-leg flatten (assigned, print after put expiry).
    if not assigned:
        # OTM put expire already closed before a later print outside put window.
        return OverlayOutcome(
            keep=True,
            gain_pct=float(base),
            reason="put_otm_before_print",
            sessions_ahead=ahead,
            flatten_session=flat_day,
        )

    mid = last_rth_mid(by_day, flat_day)
    if mid is None:
        return OverlayOutcome(
            keep=True,
            gain_pct=float(base),
            reason="no_15m_mid",
            sessions_ahead=ahead,
            flatten_session=flat_day,
        )
    exit_px, _ts = mid
    call_credit = float(_num(row, "call_credit") or 0.0)
    call_strike = _num(row, "call_strike")
    debit = float(call_close_debit) if call_close_debit is not None else 0.0
    if call_close_debit is None and call_strike is not None:
        debit = max(0.0, float(exit_px) - float(call_strike))

    # PnL: put credit - put_strike + call credit - call BTC + share exit
    pnl = (
        float(put_credit)
        - float(put_strike)
        + float(call_credit)
        - float(debit)
        + float(exit_px)
    )
    gain = cash_secured_pct(pnl, float(put_strike))
    return OverlayOutcome(
        keep=True,
        gain_pct=gain,
        reason=EXIT_EARNINGS_FLATTEN,
        sessions_ahead=ahead,
        flatten_session=flat_day,
        used=True,
        skipped=False,
        sell_px=float(exit_px),
        call_close_debit=debit,
    )


def stock_flatten_as_reference(
    row: Mapping[str, Any],
    events: Sequence[EarningsEvent],
    by_day: Mapping[date, Any],
    *,
    max_sessions: int = 10,
) -> FlattenResult:
    """Reuse stock-book flatten helper (same mid exit math)."""
    from utils.research.last_15m_earnings_event_filters import (
        apply_flatten_before_earnings,
    )

    # Stock trail book uses gain_pct_stock when wheel CSV overwrote gain_pct.
    work = dict(row)
    if work.get("gain_pct_stock") is not None and pd.notna(work.get("gain_pct_stock")):
        work["gain_pct"] = float(work["gain_pct_stock"])
    return apply_flatten_before_earnings(
        work, events, by_day, max_sessions=int(max_sessions)
    )


__all__ = [
    "OverlayOutcome",
    "apply_buy_write_earnings",
    "apply_wheel_earnings",
    "earnings_window_for_row",
    "stock_flatten_as_reference",
]
