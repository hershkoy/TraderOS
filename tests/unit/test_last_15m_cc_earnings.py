"""Unit tests for buy-write / wheel earnings overlays."""
from __future__ import annotations

from datetime import date, datetime

from utils.research.last_15m_cc_earnings import (
    apply_buy_write_earnings,
    apply_wheel_earnings,
)
from utils.research.last_15m_earnings_event_filters import EarningsEvent


def _bars(day: date, mid: float):
    ts = datetime(day.year, day.month, day.day, 15, 45)
    # last_rth_mid uses high/low for mid
    return [{"ts": ts, "high": mid + 0.1, "low": mid - 0.1}]


def test_buy_write_entry_skip_amc_same_day():
    buy = date(2024, 2, 9)
    row = {
        "buy_date": buy,
        "buy_time": "2024-02-09 15:45",
        "buy_price": 10.0,
        "gain_pct": 5.0,
        "call_credit": 0.4,
        "call_strike": 11.0,
        "bw_exit_date": "2024-03-08",
    }
    by_day = {buy: _bars(buy, 10.0)}
    ev = [EarningsEvent(earn_date=buy, amc=True)]
    got = apply_buy_write_earnings(row, ev, by_day, max_sessions=10)
    assert got.skipped
    assert not got.keep


def test_buy_write_flatten_reprices():
    buy = date(2024, 2, 1)
    flat = date(2024, 2, 8)
    row = {
        "buy_date": buy,
        "buy_time": "2024-02-01 15:45",
        "buy_price": 100.0,
        "gain_pct": 8.0,
        "call_credit": 2.0,
        "call_strike": 110.0,
        "bw_exit_date": "2024-03-01",
    }
    days = [buy, flat]
    by_day = {d: _bars(d, 100.0 if d == buy else 97.0) for d in days}
    # Also need intervening sessions for count_sessions_ahead
    for i in range(2, 8):
        d = date(2024, 2, i)
        if d.weekday() < 5:
            by_day[d] = _bars(d, 99.0)
    ev = [EarningsEvent(earn_date=date(2024, 2, 8), amc=True)]
    got = apply_buy_write_earnings(
        row, ev, by_day, max_sessions=10, call_close_debit=0.5
    )
    assert got.keep and got.used and not got.skipped
    # pnl = -100 + 2 - 0.5 + 97 = -1.5 -> -1.5%
    assert got.gain_pct is not None
    assert abs(got.gain_pct - (-1.5)) < 1e-6


def test_wheel_skip_csp_into_print():
    buy = date(2024, 1, 2)
    row = {
        "wheel_ok": True,
        "buy_date": buy,
        "buy_time": "2024-01-02 15:45",
        "gain_pct": 1.2,
        "put_strike": 50.0,
        "put_credit": 1.0,
        "put_expiry": "2024-02-02",
        "wheel_assigned": False,
        "wheel_exit_date": "2024-02-02",
    }
    by_day = {}
    for i in range(2, 20):
        d = date(2024, 1, i)
        if d.weekday() < 5:
            by_day[d] = _bars(d, 51.0)
    ev = [EarningsEvent(earn_date=date(2024, 1, 12), amc=True)]
    got = apply_wheel_earnings(row, ev, by_day, max_sessions=10)
    assert got.skipped
    assert got.reason == "skip_csp_into_print"


def test_wheel_share_leg_flatten():
    # Assign early so a print within 10 sessions of buy still hits the share leg.
    buy = date(2024, 1, 2)
    put_exp = date(2024, 1, 5)
    flat = date(2024, 1, 12)
    row = {
        "wheel_ok": True,
        "buy_date": buy,
        "buy_time": "2024-01-02 15:45",
        "gain_pct": -5.0,
        "put_strike": 100.0,
        "put_credit": 2.0,
        "put_expiry": put_exp.isoformat(),
        "call_credit": 1.5,
        "call_strike": 110.0,
        "wheel_assigned": True,
        "wheel_exit_date": "2024-02-15",
    }
    by_day = {}
    for day in range(2, 20):
        d = date(2024, 1, day)
        if d.weekday() < 5:
            by_day[d] = _bars(d, 90.0 if d == flat else 95.0)
    ev = [EarningsEvent(earn_date=flat, amc=True)]
    got = apply_wheel_earnings(
        row, ev, by_day, max_sessions=10, call_close_debit=0.0
    )
    assert got.keep and got.used
    # pnl = 2 - 100 + 1.5 - 0 + 90 = -6.5 -> -6.5%
    assert got.gain_pct is not None
    assert abs(got.gain_pct - (-6.5)) < 1e-6
