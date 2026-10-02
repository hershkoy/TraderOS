"""Unit tests for the 56 DTE 2x1 put-ratio payoff and exits."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from utils.research.mes_put_ratio import (
    expiry_package_value,
    package_value,
    pick_friday_expiry,
    pick_ratio_strikes,
    simulate_trade,
    summarize_trades,
)


def _mark(day, long_bid, long_ask, short_bid, short_ask):
    return {
        "day": day,
        "long_bid": long_bid,
        "long_ask": long_ask,
        "short_bid": short_bid,
        "short_ask": short_ask,
    }


def test_expiry_payoff_regions():
    # Credit 9.75 from the 2024-01-19 sample shape: long 22.05, short 15.90.
    credit = package_value(22.05, 15.90)
    assert abs(credit - 9.75) < 1e-9
    # Above the long strike: both worthless, keep the credit.
    assert abs(expiry_package_value(4900, 4570, 4480) - 0.0) < 1e-9
    # Between the strikes: long is intrinsic, shorts are not. We are paid the long.
    assert abs(expiry_package_value(4500, 4570, 4480) - (-70.0)) < 1e-9
    # Through the short strike: naked extra put. 4000 is a large debit.
    # short intrinsic 480, long 570 -> pay 2*480 - 570 = 390
    assert abs(expiry_package_value(4000, 4570, 4480) - 390.0) < 1e-9


def test_hold_to_expiry_winner_and_crash():
    entry = date(2024, 1, 19)
    exp = date(2024, 3, 15)
    marks = [_mark(entry, 21.7, 22.4, 15.5, 16.3)]
    winner = simulate_trade(
        entry_date=entry,
        expiration=exp,
        spot_entry=4839.81,
        long_k=4570,
        short_k=4480,
        long_delta=-0.14,
        short_delta=-0.10,
        marks=marks,
        settle_spot=4900,
        comm_per_leg=0.0,
    )
    assert not winner.skipped
    assert winner.exit_reason == "expiry"
    assert winner.pnl_pts > 0
    # mid credit = 2*15.9 - 22.05 = 9.75
    assert abs(winner.credit_pts - 9.75) < 1e-9
    assert abs(winner.pnl_dollars - 9.75 * 100) < 1e-6

    loser = simulate_trade(
        entry_date=entry,
        expiration=exp,
        spot_entry=4839.81,
        long_k=4570,
        short_k=4480,
        long_delta=-0.14,
        short_delta=-0.10,
        marks=marks,
        settle_spot=4000,
        comm_per_leg=0.0,
    )
    assert loser.pnl_pts < 0
    assert abs(loser.pnl_pts - (9.75 - 390.0)) < 1e-9


def test_tp50_exits_before_expiry():
    entry = date(2024, 1, 19)
    exp = entry + timedelta(days=56)
    day2 = entry + timedelta(days=7)
    marks = [
        _mark(entry, 21.7, 22.4, 15.5, 16.3),
        # Package mid well under half of 9.75.
        _mark(day2, 4.0, 4.2, 3.0, 3.2),
    ]
    trade = simulate_trade(
        entry_date=entry,
        expiration=exp,
        spot_entry=4800,
        long_k=4570,
        short_k=4480,
        long_delta=-0.14,
        short_delta=-0.10,
        marks=marks,
        settle_spot=4000,
        tp_frac=0.5,
        comm_per_leg=0.0,
    )
    assert trade.exit_reason == "tp"
    assert trade.exit_date == day2
    assert trade.pnl_pts > 0


def test_dte21_exit():
    entry = date(2024, 1, 19)
    exp = date(2024, 3, 15)
    late = date(2024, 2, 23)  # 21 DTE
    marks = [
        _mark(entry, 21.7, 22.4, 15.5, 16.3),
        _mark(late, 10.0, 10.4, 8.0, 8.4),
    ]
    trade = simulate_trade(
        entry_date=entry,
        expiration=exp,
        spot_entry=4800,
        long_k=4570,
        short_k=4480,
        long_delta=-0.14,
        short_delta=-0.10,
        marks=marks,
        settle_spot=4900,
        exit_dte=21,
        comm_per_leg=0.0,
    )
    assert trade.exit_reason == "dte"
    assert trade.exit_date == late
    assert trade.dte_exit == 21


def test_pick_ratio_strikes_orders_wings():
    chain = pd.DataFrame(
        [
            {"strike": 4480, "bid": 15.5, "ask": 16.3, "delta": -0.099, "underlying_price": 4840},
            {"strike": 4570, "bid": 21.7, "ask": 22.4, "delta": -0.140, "underlying_price": 4840},
            {"strike": 4700, "bid": 40.0, "ask": 41.0, "delta": -0.25, "underlying_price": 4840},
            {"strike": 4900, "bid": 80.0, "ask": 82.0, "delta": -0.55, "underlying_price": 4840},
        ]
    )
    picked = pick_ratio_strikes(chain, long_delta=0.14, short_delta=0.10)
    assert picked is not None
    assert picked["short_k"] == 4480
    assert picked["long_k"] == 4570
    assert picked["long_k"] > picked["short_k"]


def test_pick_friday_expiry_prefers_56():
    start = date(2024, 1, 19)
    exps = [start + timedelta(days=d) for d in (49, 56, 63)]
    assert pick_friday_expiry(exps, start, target_dte=56, tol=10) == start + timedelta(days=56)


def test_summarize_counts_2022_losses():
    entry = date(2022, 8, 19)
    exp = date(2022, 10, 14)
    marks = [_mark(entry, 20, 21, 14, 15)]
    loser = simulate_trade(
        entry_date=entry,
        expiration=exp,
        spot_entry=4300,
        long_k=4000,
        short_k=3900,
        long_delta=-0.14,
        short_delta=-0.10,
        marks=marks,
        settle_spot=3500,
        comm_per_leg=0.0,
    )
    stats = summarize_trades([loser])
    assert stats["losses"] == 1
    assert stats["losses_entry_2022"] == 1
    assert stats["losses_overlapping_2022"] == 1
    assert stats["by_entry_year"]["2022"]["losses"] == 1
