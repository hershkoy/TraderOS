"""Next-session MOO without chasing an overnight gap (last-15m book)."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from utils.research.evening_doji_star import Candle
from utils.research.last_15m_open_no_chase import (
    POLICY_CHASE_MOO,
    POLICY_PULLBACK,
    POLICY_SKIP_GAP,
    REASON_CHASE_SKIP,
    REASON_CLOSED_BELOW,
    REASON_MOO,
    REASON_NO_CONFIRM,
    REASON_OPEN_BELOW,
    REASON_PULLBACK,
    plan_next_open_fill,
    positive_15m_holding,
)
from utils.research.morning_doji_star import is_hammer

ET = ZoneInfo("America/New_York")


def _bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 100.0}


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _signal_and_next(*, nxt_bars, prior_close=10.00):
    fill_day = date(2024, 2, 8)
    nxt = date(2024, 2, 9)
    by_day = {
        fill_day: [
            _bar(_et(2024, 2, 8, 9, 30), 9.80, 9.90, 9.70, 9.85),
            _bar(_et(2024, 2, 8, 15, 45), 9.90, 10.10, 9.88, prior_close),
        ],
        nxt: nxt_bars,
    }
    return by_day, fill_day


def test_positive_15m_holding_green_and_hammer():
    rail = 10.0
    green = Candle(session_date=date(2024, 2, 9), open=10.05, high=10.20, low=10.02, close=10.18)
    assert positive_15m_holding(green, None, rail) == "green"
    wick_below = Candle(session_date=date(2024, 2, 9), open=10.10, high=10.20, low=9.90, close=10.15)
    assert positive_15m_holding(wick_below, None, rail) is None
    hammer = Candle(session_date=date(2024, 2, 9), open=10.12, high=10.15, low=10.00, close=10.13)
    assert is_hammer(hammer)
    assert positive_15m_holding(hammer, None, rail) == "hammer"
    red_hold = Candle(session_date=date(2024, 2, 9), open=10.20, high=10.22, low=10.05, close=10.08)
    assert positive_15m_holding(red_hold, None, rail) is None


def test_no_gap_open_above_rail_is_moo():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 10.00, 10.20, 9.95, 10.10),
        _bar(_et(2024, 2, 9, 9, 45), 10.10, 10.30, 10.05, 10.25),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=9.90, policy=POLICY_PULLBACK)
    assert got.filled is True
    assert got.reason == REASON_MOO
    assert got.gap_up is False
    assert got.entry_px == 10.00


def test_gap_up_skip_policy_does_not_fill():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 10.40, 10.50, 10.30, 10.45),
        _bar(_et(2024, 2, 9, 9, 45), 10.45, 10.48, 10.05, 10.10),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=9.90, policy=POLICY_SKIP_GAP)
    assert got.filled is False
    assert got.reason == REASON_CHASE_SKIP
    assert got.gap_up is True


def test_chase_moo_buys_gapped_open():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 10.40, 10.50, 10.30, 10.45),
        _bar(_et(2024, 2, 9, 9, 45), 10.45, 10.48, 10.05, 10.10),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=9.90, policy=POLICY_CHASE_MOO)
    assert got.filled is True
    assert got.reason == REASON_MOO
    assert got.entry_px == 10.40


def test_gap_up_pullback_green_holding_fills_close():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 10.40, 10.50, 10.30, 10.45),
        _bar(_et(2024, 2, 9, 9, 45), 10.40, 10.42, 10.05, 10.08),
        _bar(_et(2024, 2, 9, 10, 0), 10.08, 10.25, 10.02, 10.22),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=10.00, policy=POLICY_PULLBACK)
    assert got.filled is True
    assert got.reason == REASON_PULLBACK
    assert got.pattern == "green"
    assert got.entry_px == 10.22
    assert got.entry_ts == _et(2024, 2, 9, 10, 0)


def test_gap_up_no_pullback_skips():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 10.40, 10.50, 10.30, 10.45),
        _bar(_et(2024, 2, 9, 9, 45), 10.45, 10.60, 10.35, 10.55),
        _bar(_et(2024, 2, 9, 10, 0), 10.55, 10.70, 10.50, 10.65),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=10.00, policy=POLICY_PULLBACK)
    assert got.filled is False
    assert got.reason == REASON_NO_CONFIRM
    assert got.gap_up is True


def test_gap_up_close_below_rail_cancels():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 10.40, 10.50, 10.30, 10.45),
        _bar(_et(2024, 2, 9, 9, 45), 10.30, 10.35, 9.80, 9.90),
        _bar(_et(2024, 2, 9, 10, 0), 9.90, 10.30, 10.00, 10.25),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=10.00, policy=POLICY_PULLBACK)
    assert got.filled is False
    assert got.reason == REASON_CLOSED_BELOW


def test_skip_gap_open_below_rail():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 9.80, 9.90, 9.70, 9.85),
        _bar(_et(2024, 2, 9, 9, 45), 9.90, 10.20, 9.88, 10.15),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=10.00, policy=POLICY_SKIP_GAP)
    assert got.filled is False
    assert got.gap_up is False
    assert got.reason == REASON_OPEN_BELOW


def test_open_below_rail_reclaim_green_holding():
    nxt_bars = [
        _bar(_et(2024, 2, 9, 9, 30), 9.80, 9.90, 9.70, 9.85),
        _bar(_et(2024, 2, 9, 9, 45), 9.90, 10.20, 9.88, 10.15),
        _bar(_et(2024, 2, 9, 10, 0), 10.10, 10.30, 10.05, 10.28),
    ]
    by_day, fill_day = _signal_and_next(nxt_bars=nxt_bars, prior_close=10.00)
    got = plan_next_open_fill(by_day, fill_day=fill_day, resist=10.00, policy=POLICY_PULLBACK)
    assert got.gap_up is False
    assert got.open_holds is False
    assert got.filled is True
    assert got.reason == REASON_PULLBACK
    assert got.pattern == "green"
    assert got.entry_px == 10.28
