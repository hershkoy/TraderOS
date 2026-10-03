"""Unit tests for the NVDA earnings expected-move premium sale (no Theta)."""
from __future__ import annotations

from datetime import date

from utils.research.nvda_earnings_em import (
    unadjusted_price,
    amc_sessions,
    book_stats,
    buy_price,
    call_intrinsic,
    classify_open,
    defined_risk,
    held_iron_condor,
    held_short_put,
    mode_key,
    nearest_strike,
    package_pnl,
    pick_expiry_after,
    put_intrinsic,
    sell_price,
    spread_intrinsic,
    strike_just_above,
    strike_just_below,
    synthetic_forward,
    target_wing,
    usable_quote,
    wing_strike,
)


STRIKES = [170.0, 172.5, 175.0, 177.5, 180.0, 182.5, 185.0, 187.5, 190.0, 192.5, 200.0, 202.5, 205.0, 207.5]


def _q(bid, ask):
    return {"bid": bid, "ask": ask, "mid": 0.5 * (bid + ask), "spread": ask - bid}


def test_pre_split_yahoo_close_is_scaled_onto_the_listed_strikes():
    # 2024-05-22 Yahoo close 94.95 was 949.50 before the 10-for-1.
    assert abs(unadjusted_price(date(2024, 5, 22), 94.949997) - 949.49997) < 1e-6
    assert abs(unadjusted_price(date(2025, 11, 19), 186.52) - 186.52) < 1e-9
    assert abs(unadjusted_price(date(2024, 8, 28), 125.61) - 125.61) < 1e-9


def test_expiry_is_the_first_listed_date_after_the_report():
    exps = [date(2026, 2, 25), date(2026, 2, 27), date(2026, 3, 6)]
    assert pick_expiry_after(exps, date(2026, 2, 25)) == date(2026, 2, 27)
    assert pick_expiry_after(exps, date(2026, 2, 27)) == date(2026, 3, 6)


def test_video_example_strikes_sit_just_outside_179_and_200():
    assert strike_just_below(STRIKES, 179.0) == 177.5
    assert strike_just_above(STRIKES, 200.0) == 202.5
    assert nearest_strike(STRIKES, 190.2) == 190.0
    assert wing_strike(STRIKES, 177.5, 5.0, "put") == 172.5
    assert wing_strike(STRIKES, 202.5, 5.0, "call") == 207.5


def test_five_points_on_190_scales_with_spot():
    assert abs(target_wing(190.0) - 5.0) < 1e-9
    assert abs(target_wing(950.0) - 25.0) < 1e-9


def test_wing_snaps_when_the_exact_strike_is_missing():
    coarse = [170.0, 175.0, 180.0]
    # 5 points under 177.5 is 172.5; 170 is closer than 175 and is still below.
    assert wing_strike(coarse, 177.5, 5.0, "put") == 170.0


def test_parity_forward_and_rejected_quotes():
    assert abs(synthetic_forward(190.0, _q(5.8, 6.2), _q(4.9, 5.1)) - 191.0) < 1e-9
    assert usable_quote({"bid": 0.0, "ask": 1.0}) is None
    assert usable_quote({"bid": 1.2, "ask": 1.0}) is None
    got = usable_quote({"bid": 1.0, "ask": 1.2})
    assert got is not None and abs(got["mid"] - 1.1) < 1e-9


def test_natural_entry_sells_the_bid_and_buys_the_ask():
    q = _q(1.00, 1.20)
    assert sell_price(q, "natural", 0.0) == 1.00
    assert buy_price(q, "natural", 0.0) == 1.20
    assert abs(sell_price(q, "mid", 0.05) - 1.05) < 1e-9
    assert abs(buy_price(q, "mid", 0.05) - 1.15) < 1e-9


def test_credit_spread_keeps_the_crush_and_natural_can_lose_it():
    legs = [
        {"role": "short", "entry": _q(1.90, 2.00), "exit": _q(0.40, 0.50)},
        {"role": "long", "entry": _q(0.90, 1.00), "exit": _q(0.15, 0.25)},
    ]
    mid = package_pnl(legs, "mid", 0.0, comm_per_leg=0.0)
    # entry credit (1.95 - 0.95) = 1.00; exit debit (0.45 - 0.20) = 0.25; gross 0.75
    assert abs(mid["credit"] - 1.00) < 1e-9
    assert abs(mid["net"] - 75.0) < 1e-9
    natural = package_pnl(legs, "natural", 0.0, comm_per_leg=0.65)
    # entry 1.90 - 1.00 = 0.90; exit 0.50 - 0.15 = 0.35; gross 0.55; fees 2.60
    assert abs(natural["credit"] - 0.90) < 1e-9
    assert abs(natural["net"] - (55.0 - 2.60)) < 1e-9
    assert defined_risk(5.0, mid["credit"]) == 4.0


def test_open_classes_match_the_video_claims():
    flat = classify_open(prior_close=190.0, next_open=190.0, spot=190.0, em=10.8)
    assert flat["opened_green"] and flat["inside_em"] and not flat["broke_down"]
    down = classify_open(prior_close=190.0, next_open=170.0, spot=190.0, em=10.8)
    assert not down["opened_green"] and down["broke_down"] and not down["inside_em"]
    up = classify_open(prior_close=190.0, next_open=210.0, spot=190.0, em=10.8)
    assert up["broke_up"] and not up["inside_em"]
    assert held_short_put(179.0, 177.5)
    assert not held_short_put(177.0, 177.5)
    assert held_iron_condor(190.0, 177.5, 202.5)
    assert not held_iron_condor(210.0, 177.5, 202.5)


def test_intrinsic_spread_is_the_width_once_both_strikes_are_through():
    assert put_intrinsic(180.0, 190.0) == 0.0
    assert abs(spread_intrinsic(180.0, 175.0, 170.0, "put") - 5.0) < 1e-9
    assert call_intrinsic(200.0, 190.0) == 0.0
    assert abs(spread_intrinsic(200.0, 205.0, 210.0, "call") - 5.0) < 1e-9


def test_amc_exit_is_the_next_session():
    days = [date(2026, 2, 25), date(2026, 2, 26), date(2026, 2, 27)]
    assert amc_sessions(date(2026, 2, 25), days) == (date(2026, 2, 25), date(2026, 2, 26))


def test_book_stats_blank_pf_when_nothing_lost():
    rows = [{"net_mid": 10.0, "ror_mid": 5.0}, {"net_mid": 20.0, "ror_mid": 8.0}]
    stats = book_stats(rows, "net_mid")
    assert stats["n"] == 2
    assert stats["win_rate_pct"] == 100.0
    assert stats["pf"] is None
    assert mode_key("mid", 0.10) == "mid_slip0.1"
    assert mode_key("natural", 0.0) == "natural"
