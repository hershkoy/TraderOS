"""Unit tests for the QQQ 0-DTE 9:45 short put (no Theta)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from utils.research.qqq_0dte_put import (
    CLAIMED,
    FILL_MID,
    FILL_NATURAL,
    FILL_TARGET,
    approx_put_margin,
    claimed_one_lot_pnl,
    compound_book,
    max_losing_streak,
    select_put,
    simulate_short_put,
    summarize_pnls,
)


def _chain(rows):
    recs = []
    for strike, bid, ask, delta in rows:
        recs.append(
            {
                "timestamp": "2024-06-03T09:45:00.000",
                "strike": strike,
                "right": "PUT",
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "underlying_price": 400.0,
                "implied_vol": 0.20,
                "iv_error": 0.0,
            }
        )
    return pd.DataFrame(recs)


def _path(rows):
    recs = []
    for hh, mm, bid, ask in rows:
        recs.append(
            {
                "timestamp": f"2024-06-03T{hh:02d}:{mm:02d}:00",
                "bid": bid,
                "ask": ask,
            }
        )
    return pd.DataFrame(recs)


def test_claimed_averages_are_one_lot_not_the_account():
    # 782 * 85 - 285 * 174. The $556k account is a different, sized book.
    assert abs(claimed_one_lot_pnl() - 16880.0) < 1e-6
    assert CLAIMED["winners"] / CLAIMED["n"] == CLAIMED["winners"] / 1067


def test_select_nearest_30_delta_and_ignore_bad_iv():
    chain = _chain(
        [
            (397, 1.20, 1.30, -0.40),
            (398, 0.90, 1.00, -0.31),
            (399, 0.70, 0.80, -0.22),
            (390, 0.01, 0.02, -0.01),
        ]
    )
    bad = _chain([(396, 0.10, 0.11, -0.30)])
    bad["iv_error"] = 100.0
    chain = pd.concat([chain, bad], ignore_index=True)
    picked, reason = select_put(chain)
    assert reason == "ok"
    assert picked["strike"] == 398.0
    assert abs(picked["delta"] + 0.30) < 0.02
    points, reason_p = select_put(chain, points_below=2.5)
    assert reason_p == "ok"
    # 400 - 2.5 is equidistant from 397 and 398; the lower strike wins the tie.
    assert points["strike"] == 397.0


def test_select_rejects_far_delta():
    chain = _chain([(380, 0.05, 0.06, -0.05)])
    picked, reason = select_put(chain)
    assert picked is None
    assert reason == "no_delta"


def test_target_take_profit_keeps_95_percent():
    path = _path(
        [
            (9, 45, 0.90, 1.10),
            (10, 0, 0.80, 0.90),
            (11, 0, 0.02, 0.04),
            (15, 45, 0.01, 0.02),
        ]
    )
    out = simulate_short_put(path, strike=400.0, mode=FILL_TARGET)
    assert out["exit_reason"] == "tp"
    assert abs(out["credit"] - 1.0) < 1e-9
    assert abs(out["exit_px"] - 0.05) < 1e-9
    assert abs(out["pnl_1lot"] - 95.0) < 1e-6


def test_target_stop_is_twice_the_credit():
    path = _path(
        [
            (9, 45, 0.90, 1.10),
            (10, 30, 2.90, 3.10),
        ]
    )
    out = simulate_short_put(path, strike=400.0, mode=FILL_TARGET)
    assert out["exit_reason"] == "sl"
    assert abs(out["exit_px"] - 3.0) < 1e-9
    assert abs(out["pnl_1lot"] + 200.0) < 1e-6


def test_mid_gap_through_stop_fills_worse_than_the_bracket():
    path = _path(
        [
            (9, 45, 0.90, 1.10),
            (10, 30, 4.80, 5.20),
        ]
    )
    gap = simulate_short_put(path, strike=400.0, mode=FILL_MID)
    target = simulate_short_put(path, strike=400.0, mode=FILL_TARGET)
    assert gap["exit_reason"] == "sl"
    assert abs(gap["exit_px"] - 5.0) < 1e-9
    assert gap["pnl_1lot"] < target["pnl_1lot"]
    assert abs(target["pnl_1lot"] + 200.0) < 1e-6


def test_natural_wide_ask_misses_the_95_percent_limit():
    path = _path(
        [
            (9, 45, 1.00, 1.10),
            (12, 0, 0.01, 0.08),
            (15, 45, 0.10, 0.20),
        ]
    )
    natural = simulate_short_put(path, strike=400.0, mode=FILL_NATURAL)
    target = simulate_short_put(path, strike=400.0, mode=FILL_TARGET)
    assert target["exit_reason"] == "tp"
    assert natural["exit_reason"] == "flat"
    assert abs(natural["pnl_1lot"] - 80.0) < 1e-6


def test_flat_at_1545_and_early_close():
    path = _path(
        [
            (9, 45, 0.90, 1.10),
            (12, 0, 0.40, 0.50),
            (15, 45, 0.30, 0.40),
        ]
    )
    out = simulate_short_put(path, strike=400.0, mode=FILL_TARGET)
    assert out["exit_reason"] == "flat"
    assert abs(out["exit_px"] - 0.35) < 1e-9
    early = _path(
        [
            (9, 45, 0.90, 1.10),
            (12, 55, 0.20, 0.30),
        ]
    )
    early_out = simulate_short_put(early, strike=400.0, mode=FILL_TARGET)
    assert early_out["exit_reason"] == "flat"
    assert abs(early_out["exit_px"] - 0.25) < 1e-9


def test_entry_bar_does_not_trigger_the_stop():
    path = _path(
        [
            (9, 45, 3.50, 3.70),
            (10, 0, 0.90, 1.10),
            (15, 45, 0.40, 0.50),
        ]
    )
    out = simulate_short_put(path, strike=400.0, mode=FILL_TARGET)
    assert out["exit_reason"] == "flat"
    assert out["pnl_1lot"] > 0


def test_compound_flat_bp_and_reg_t():
    trades = [
        {
            "date": "2024-01-02",
            "pnl_1lot": 100.0,
            "spot": 400.0,
            "strike": 397.0,
            "credit": 1.0,
        },
        {
            "date": "2024-01-03",
            "pnl_1lot": -50.0,
            "spot": 400.0,
            "strike": 397.0,
            "credit": 1.0,
        },
    ]
    sized = compound_book(trades, start=100_000.0, bp=10_000.0, mode="flat_bp")
    assert list(sized["curve"]["contracts"]) == [10, 10]
    assert abs(sized["end"] - 100_500.0) < 1e-6
    assert abs(approx_put_margin(400.0, 397.0, 1.0) - 7800.0) < 1e-6
    reg = compound_book(trades, start=100_000.0, mode="reg_t")
    assert int(reg["curve"].iloc[0]["contracts"]) == 12
    stats = summarize_pnls([100.0, -50.0, 25.0])
    assert stats["n"] == 3
    assert stats["winners"] == 2
    assert abs(stats["avg_loss"] - 50.0) < 1e-9
    assert max_losing_streak([10, -1, -2, 5, -3]) == 2
