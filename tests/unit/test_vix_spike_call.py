"""Unit tests for the VIX spike call credit spread (no Theta)."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from utils.research.vix_spike_call import (
    book_stats,
    expected_vix_monthly,
    filter_monthly_expirations,
    implied_vix_forward,
    is_entry_signal,
    leg_is_usable,
    pick_monthly_expiry,
    round_to_step,
    run_vix_spike_book,
    third_friday,
    walk_call_spread_exit,
)
from utils.research.spx_bull_put import trade_pnl, vertical_credit


def test_stuck_ten_ask_is_not_a_usable_leg():
    assert not leg_is_usable({"bid": 0.25, "ask": 10.0, "mid": 5.125})
    assert leg_is_usable({"bid": 0.60, "ask": 1.10, "mid": 0.85})
    assert not leg_is_usable({"bid": 0.05, "ask": 0.0, "mid": 0.0})


def test_august_2024_monthly_is_the_wednesday_before_september_spx():
    assert third_friday(2024, 9) == date(2024, 9, 20)
    assert expected_vix_monthly(2024, 8) == date(2024, 8, 21)
    listed = [
        date(2024, 8, 14),
        date(2024, 8, 21),
        date(2024, 9, 18),
    ]
    assert filter_monthly_expirations(listed) == [date(2024, 8, 21), date(2024, 9, 18)]


def test_pick_nearest_monthly_inside_30_to_60_dte():
    monthlies = [date(2024, 8, 21), date(2024, 9, 18), date(2024, 10, 16)]
    # 2024-08-05 -> Aug 21 is 16 DTE, Sep 18 is 44, Oct 16 is 72
    assert pick_monthly_expiry(monthlies, date(2024, 8, 5)) == date(2024, 9, 18)


def test_round_half_up_onto_five_point_grid():
    assert round_to_step(42.4, 5) == 40.0
    assert round_to_step(42.5, 5) == 45.0
    assert round_to_step(42.6, 5) == 45.0


def test_cross_requires_a_prior_close_under_the_trigger():
    assert is_entry_signal(31.0, 33.0, 32.0, "cross")
    assert not is_entry_signal(33.0, 34.0, 32.0, "cross")
    assert not is_entry_signal(None, 40.0, 32.0, "cross")
    assert is_entry_signal(33.0, 34.0, 32.0, "elevated")
    assert not is_entry_signal(20.0, 31.9, 32.0, "elevated")


def test_parity_forward_sits_between_spot_and_the_strike():
    call_q = {"bid": 5.8, "ask": 6.2, "mid": 6.0}
    put_q = {"bid": 1.9, "ask": 2.1, "mid": 2.0}
    forward = implied_vix_forward(30.0, call_q, put_q, dte=45, rate=0.0)
    assert forward is not None
    assert abs(forward - 34.0) < 1e-9
    assert implied_vix_forward(30.0, {"bid": 0.0, "ask": 1.0, "mid": 0.5}, put_q, 45) is None


def test_profit_target_beats_a_later_expiry_week():
    exp = date(2024, 9, 18)
    days = [date(2024, 8, 6), date(2024, 8, 7), date(2024, 9, 16)]

    def debit_on(day):
        return {date(2024, 8, 6): 2.0, date(2024, 8, 7): 1.0, date(2024, 9, 16): 0.2}[day]

    # credit 3.00, remain 40% => tp debit 1.20. Aug 7 hits it.
    got = walk_call_spread_exit(
        session_days=days, debit_on=debit_on, expiration=exp, credit=3.0, tp_remain=0.40
    )
    assert got == (date(2024, 8, 7), 1.0, "tp_60")


def test_no_stop_holds_a_doubled_debit_into_expiry_week():
    exp = date(2024, 9, 18)
    days = [date(2024, 8, 6), date(2024, 9, 16)]

    def debit_on(day):
        return 9.0 if day == date(2024, 8, 6) else 4.0

    got = walk_call_spread_exit(
        session_days=days, debit_on=debit_on, expiration=exp, credit=3.0, tp_remain=0.40
    )
    assert got[0] == date(2024, 9, 16)
    assert got[2] == "expiry_week"


def test_sample_end_is_not_labeled_as_settlement():
    exp = date(2024, 10, 16)
    days = [date(2024, 8, 6), date(2024, 8, 7)]

    def debit_on(_day):
        return 2.0

    got = walk_call_spread_exit(
        session_days=days, debit_on=debit_on, expiration=exp, credit=3.0, tp_remain=0.40
    )
    assert got[2] == "sample_end"


def test_book_takes_the_cross_and_sizes_pnl_off_defined_risk():
    idx = pd.bdate_range("2024-08-01", "2024-09-20")
    closes = []
    for i, _ts in enumerate(idx):
        closes.append(28.0 if i < 2 else (38.0 if i == 2 else 22.0))
    spots = pd.DataFrame({"close": closes}, index=idx)
    exp = date(2024, 9, 18)
    strikes = [float(k) for k in range(20, 80, 5)]

    def mark_fn(_exp, strike, _right, start, end):
        out = {}
        for ts in idx:
            day = ts.date()
            if day < start or day > end:
                continue
            # Short 50 is richer than long 60. Debit collapses after two sessions.
            if strike == 50:
                mid = 4.0 if day <= date(2024, 8, 7) else 0.8
            else:
                mid = 1.0 if day <= date(2024, 8, 7) else 0.2
            out[day] = {"bid": mid - 0.05, "ask": mid + 0.05, "mid": mid}
        return out

    book = run_vix_spike_book(
        spots,
        [date(2024, 8, 14), exp],
        start=date(2024, 8, 1),
        end=date(2024, 9, 20),
        mark_fn=mark_fn,
        strike_fn=lambda _exp: strikes,
        trigger=32,
        package_slip=0.10,
        arm_mode="cross",
        ref_mode="spot",
        max_entries=5,
    )
    assert len(book.trades) == 1
    trade = book.trades[0]
    assert trade.entry_date == idx[2].date()
    assert trade.short_k == 50.0
    assert trade.long_k == 60.0
    assert trade.exit_reason == "tp_60"
    credit = vertical_credit(
        {"bid": 3.95, "ask": 4.05, "mid": 4.0},
        {"bid": 0.95, "ask": 1.05, "mid": 1.0},
        "mid",
        0.10,
    )
    assert abs(trade.credit - credit) < 1e-9
    pnl, ror, _risk = trade_pnl(trade.credit, trade.exit_debit, 10.0)
    assert abs(trade.pnl - pnl) < 1e-6
    assert abs(trade.ror_pct - ror) < 1e-6
    stats = book_stats(book.trades, risk_budget=15_000)
    assert stats["n"] == 1
    assert stats["pf"] is None or stats["total_pnl"] > 0
    # One lot of a ~$7 risk is well under $15k, so the budget sleeve uses many lots.
    assert stats["budget_pnl"] > stats["total_pnl"]


def test_second_day_above_the_trigger_does_not_reenter_on_a_cross_book():
    idx = pd.bdate_range("2024-08-05", periods=6)
    spots = pd.DataFrame({"close": [20, 40, 42, 18, 18, 18]}, index=idx)
    exp = idx[-1].date() + timedelta(days=40)

    def mark_fn(_exp, strike, _right, start, end):
        out = {}
        day = start
        while day <= end:
            mid = 3.0 if strike < 60 else 0.4
            out[day] = {"bid": mid - 0.05, "ask": mid + 0.05, "mid": mid}
            day = day + timedelta(days=1)
        return out

    book = run_vix_spike_book(
        spots,
        [exp],
        start=idx[0].date(),
        end=idx[-1].date(),
        mark_fn=mark_fn,
        strike_fn=lambda _e: [float(k) for k in range(20, 90, 5)],
        trigger=32,
        arm_mode="cross",
        ref_mode="spot",
        min_credit=0.5,
        monthly_only=False,
    )
    assert len(book.trades) == 1
    assert book.skipped == [] or all(s["reason"] != "position_open" for s in book.skipped)
