"""Unit tests for iron-fly path MTM (no Theta)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from utils.research.spxw_0dte_iron_fly_path import (
    merge_package_paths,
    mtm_series,
    path_stats,
    summarize_path_book,
)


def _leg(rows, bid_key="bid", ask_key="ask"):
    return pd.DataFrame(rows)


def test_merge_and_mtm_mae_mfe():
    # Entry credit 8.0. Close mid starts at 8, dips to 12 (adverse), rises to 4 (good).
    ts = [
        "2024-06-03T13:45:00-04:00",
        "2024-06-03T14:00:00-04:00",
        "2024-06-03T15:00:00-04:00",
    ]
    # Package close = sp+sc - lp - lc. Build so close_mid = 8, 12, 4
    # Use lp=lc=1 always; sp+sc = 10, 14, 6
    sp = _leg(
        [
            {"timestamp": ts[0], "bid": 4.0, "ask": 4.2},
            {"timestamp": ts[1], "bid": 6.0, "ask": 6.2},
            {"timestamp": ts[2], "bid": 2.0, "ask": 2.2},
        ]
    )
    sc = _leg(
        [
            {"timestamp": ts[0], "bid": 5.8, "ask": 6.0},
            {"timestamp": ts[1], "bid": 7.8, "ask": 8.0},
            {"timestamp": ts[2], "bid": 3.8, "ask": 4.0},
        ]
    )
    wing = _leg(
        [
            {"timestamp": ts[0], "bid": 0.9, "ask": 1.1},
            {"timestamp": ts[1], "bid": 0.9, "ask": 1.1},
            {"timestamp": ts[2], "bid": 0.9, "ask": 1.1},
        ]
    )
    pkg = merge_package_paths(sp, sc, wing, wing)
    assert len(pkg) == 3
    mtm = mtm_series(pkg, credit=8.0, mark="mid")
    # close mids ~ (4.1+5.9 - 1 - 1)=8, (6.1+7.9-1-1)=12, (2.1+3.9-1-1)=4
    assert abs(mtm.iloc[0]["pnl"] - 0.0) < 5  # near flat at entry
    stats = path_stats(mtm, credit=8.0, settle_pnl=400.0)  # settle at close mid 4 -> +400
    assert stats["mae"] < 0
    assert stats["mfe"] > 0
    assert stats["tp_hit"] is True  # half credit = $400
    assert stats["tp_pnl"] == 400.0
    assert stats["exit_mode"] == "tp50"


def test_underwater_winner_flag():
    ts = [
        "2024-06-03T13:45:00-04:00",
        "2024-06-03T14:30:00-04:00",
        "2024-06-03T15:45:00-04:00",
    ]
    # close costs 8, 14, 3 with credit 8 -> pnl 0, -600, +500
    sp = _leg(
        [
            {"timestamp": ts[0], "bid": 4.0, "ask": 4.0},
            {"timestamp": ts[1], "bid": 7.0, "ask": 7.0},
            {"timestamp": ts[2], "bid": 1.5, "ask": 1.5},
        ]
    )
    sc = _leg(
        [
            {"timestamp": ts[0], "bid": 6.0, "ask": 6.0},
            {"timestamp": ts[1], "bid": 9.0, "ask": 9.0},
            {"timestamp": ts[2], "bid": 3.5, "ask": 3.5},
        ]
    )
    wing = _leg(
        [
            {"timestamp": ts[0], "bid": 1.0, "ask": 1.0},
            {"timestamp": ts[1], "bid": 1.0, "ask": 1.0},
            {"timestamp": ts[2], "bid": 1.0, "ask": 1.0},
        ]
    )
    pkg = merge_package_paths(sp, sc, wing, wing)
    mtm = mtm_series(pkg, credit=8.0)
    stats = path_stats(mtm, credit=8.0, settle_pnl=500.0)
    assert stats["underwater_half"] is True
    assert stats["winner_was_underwater_half"] is True


def test_summarize_path_book():
    rows = [
        {
            "mae": -200.0,
            "mfe": 400.0,
            "settle_pnl": 300.0,
            "tp_pnl": 250.0,
            "tp_hit": True,
            "underwater_half": False,
            "winner_was_underwater_half": False,
            "entry_slip_pts": 0.20,
            "exit_mid_pnl": 300.0,
            "exit_nat_pnl": 250.0,
        },
        {
            "mae": -500.0,
            "mfe": 100.0,
            "settle_pnl": -400.0,
            "tp_pnl": -400.0,
            "tp_hit": False,
            "underwater_half": True,
            "winner_was_underwater_half": False,
            "entry_slip_pts": 0.40,
            "exit_mid_pnl": -400.0,
            "exit_nat_pnl": -450.0,
        },
    ]
    s = summarize_path_book(rows)
    assert s["n"] == 2
    assert abs(s["entry_slip_pts_median"] - 0.30) < 1e-9
