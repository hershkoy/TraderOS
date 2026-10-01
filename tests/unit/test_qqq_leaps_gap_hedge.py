"""Unit tests for QQQ LEAPS gap hedge helpers (no live Theta)."""
from __future__ import annotations

from datetime import date, timedelta

from utils.research.qqq_leaps_gap import LeapTrade
from utils.research.qqq_leaps_gap_hedge import (
    ATTACH_CONTINUOUS,
    ATTACH_ENTRY_DAY,
    ATTACH_WHILE_OPEN,
    H1,
    H2,
    H3,
    HedgeTrade,
    apply_max_open,
    combine_leaps_hedge_stats,
    hedge_stats,
    open_leaps_on_day,
    pick_expiration,
    snap_strike,
    target_hedge_units,
)


def _leap(entry: date, exit_: date, contracts: int = 1, pnl: float = 100.0) -> LeapTrade:
    return LeapTrade(
        mode="strat1",
        entry_date=entry,
        exit_date=exit_,
        contracts=contracts,
        entry_px=100.0,
        strike=100.0,
        entry_premium=10.0,
        exit_premium=15.0,
        entry_delta=0.6,
        entry_iv=0.2,
        dte=365,
        hold_days=(exit_ - entry).days,
        exit_reason="tp50",
        pnl_cash=pnl,
        gain_pct=50.0,
        skipped=False,
    )


def test_snap_strike_nearest():
    assert snap_strike([10.0, 18.0, 25.0, 30.0], 17.5) == 18.0
    assert snap_strike([10.0, 18.0, 25.0], 24.0) == 25.0


def test_pick_expiration_near_45_dte():
    day = date(2021, 6, 1)
    exps = [
        day + timedelta(days=10),
        day + timedelta(days=44),
        day + timedelta(days=90),
        day + timedelta(days=52),
    ]
    got = pick_expiration(exps, day)
    assert got == day + timedelta(days=44)


def test_target_while_open_scales_with_lots():
    assert target_hedge_units(H1, ATTACH_WHILE_OPEN, 0, 12.0) == 0
    assert target_hedge_units(H1, ATTACH_WHILE_OPEN, 5, 12.0) == 5
    assert target_hedge_units(H1, ATTACH_WHILE_OPEN, 5, 16.0) == 0
    assert target_hedge_units(H2, ATTACH_WHILE_OPEN, 5, 13.0) == 3
    assert target_hedge_units(H2, ATTACH_WHILE_OPEN, 1, 13.0) == 1


def test_target_continuous_holds_one_with_zero_leaps():
    assert target_hedge_units(H1, ATTACH_CONTINUOUS, 0, 12.0) == 1
    assert target_hedge_units(H2, ATTACH_CONTINUOUS, 0, 12.0) == 1
    assert target_hedge_units(H1, ATTACH_CONTINUOUS, 4, 12.0) == 4
    assert target_hedge_units(H3, ATTACH_CONTINUOUS, 0, 16.0) == 0


def test_target_entry_day_unused_returns_zero():
    assert target_hedge_units(H1, ATTACH_ENTRY_DAY, 5, 12.0) == 0


def test_open_leaps_on_day_half_open():
    t1 = _leap(date(2021, 1, 4), date(2021, 2, 1))
    t2 = _leap(date(2021, 1, 10), date(2021, 3, 1))
    assert len(open_leaps_on_day([t1, t2], date(2021, 1, 5))) == 1
    assert len(open_leaps_on_day([t1, t2], date(2021, 1, 15))) == 2
    assert len(open_leaps_on_day([t1, t2], date(2021, 2, 1))) == 1


def test_apply_max_open_fcfs():
    trades = [
        _leap(date(2021, 1, 4), date(2021, 6, 1), pnl=100),
        _leap(date(2021, 1, 5), date(2021, 6, 2), pnl=200),
        _leap(date(2021, 1, 6), date(2021, 6, 3), pnl=300),
    ]
    kept = apply_max_open(trades, 2)
    assert len(kept) == 2
    assert kept[0].entry_date == date(2021, 1, 4)
    assert kept[1].entry_date == date(2021, 1, 5)


def test_hedge_stats_and_combine():
    leaps = [_leap(date(2021, 1, 4), date(2021, 2, 1), pnl=1000.0)]
    hedges = [
        HedgeTrade(
            structure=H1,
            attach=ATTACH_ENTRY_DAY,
            mark_mode="mid_mid",
            entry_date=date(2021, 1, 4),
            exit_date=date(2021, 2, 1),
            contracts=1,
            symbol="QQQ",
            expiration=date(2021, 2, 19),
            long_strike=95.0,
            short_strike=90.0,
            long_right="put",
            short_right="put",
            entry_debit=1.0,
            exit_credit=0.4,
            exit_reason="leaps_exit",
            pnl_cash=-60.0,
            hold_days=28,
            skipped=False,
        )
    ]
    hs = hedge_stats(hedges)
    assert hs["n"] == 1
    assert hs["total_pnl"] == -60.0
    comb = combine_leaps_hedge_stats(leaps, hedges)
    assert comb["combined_pnl"] == 940.0
    assert comb["leaps_pnl"] == 1000.0


def test_h2_arm_tighter_than_h3():
    assert target_hedge_units(H3, ATTACH_WHILE_OPEN, 2, 14.5) == 2
    assert target_hedge_units(H2, ATTACH_WHILE_OPEN, 2, 14.5) == 0
