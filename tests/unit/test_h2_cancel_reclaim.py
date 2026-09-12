"""Unit tests for H2 support-cancel then later resist-break walk."""
from __future__ import annotations

import numpy as np

from utils.research.h2_cancel_reclaim import (
    OUTCOME_CANCELLED,
    OUTCOME_EXPIRED,
    OUTCOME_FILLED,
    close_broke_support,
    count_bucket,
    setup_row_ok,
    walk_h2_cancel_reclaim,
)


def test_tars_like_shallow_poke_then_reclaim():
    # support=10, width=2, error 1.2% cancel below 9.88. Shallow cap 0.25w=0.50.
    n = 12
    close = np.full(n, 10.5)
    close[0] = 11.5  # H2
    close[3] = 9.70  # cancel, undershoot 0.30 / 2 = 0.15
    close[4] = 10.2  # recovered
    close[8] = 12.40  # resist 12 * 1.012 = 12.144
    low = close - 0.1
    low[3] = 9.60
    got = walk_h2_cancel_reclaim(
        close,
        low,
        h2_i=0,
        support_x0=0,
        support_y0=10.0,
        support_slope=0.0,
        width=2.0,
        min_wait=6,
        max_wait=20,
    )
    assert got["outcome"] == OUTCOME_CANCELLED
    assert got["cancel_i"] == 3
    assert abs(got["cancel_undershoot_close_width"] - 0.15) < 1e-9
    assert got["shallow"] is True
    assert got["recovered_next"] is True
    assert got["tars_like"] is True
    assert got["reclaim_i"] == 8
    assert got["reclaim_within_wait"] is True


def test_deep_cancel_still_can_reclaim():
    close = np.array([11.0, 10.5, 8.0, 10.2, 10.3, 10.4, 10.5, 12.5])
    low = close - 0.2
    got = walk_h2_cancel_reclaim(
        close,
        low,
        h2_i=0,
        support_x0=0,
        support_y0=10.0,
        support_slope=0.0,
        width=2.0,
        min_wait=6,
        max_wait=20,
    )
    assert got["outcome"] == OUTCOME_CANCELLED
    assert got["cancel_i"] == 2
    assert got["shallow"] is False
    assert got["tars_like"] is False
    assert got["reclaim_i"] == 7


def test_fill_before_cancel_is_not_reclaim_sleeve():
    close = np.array([11.0, 10.8, 10.9, 11.0, 11.1, 11.2, 12.5, 8.0])
    low = close - 0.1
    got = walk_h2_cancel_reclaim(
        close,
        low,
        h2_i=0,
        support_x0=0,
        support_y0=10.0,
        support_slope=0.0,
        width=2.0,
        min_wait=6,
        max_wait=20,
    )
    assert got["outcome"] == OUTCOME_FILLED
    assert got["fill_i"] == 6
    assert got["cancel_i"] is None
    assert got["reclaim_i"] is None


def test_cancel_without_later_breakout_expires_reclaim():
    close = np.full(15, 10.4)
    close[2] = 9.70
    close[3] = 10.4
    low = close - 0.05
    got = walk_h2_cancel_reclaim(
        close,
        low,
        h2_i=0,
        support_x0=0,
        support_y0=10.0,
        support_slope=0.0,
        width=2.0,
        min_wait=6,
        max_wait=10,
    )
    assert got["outcome"] == OUTCOME_CANCELLED
    assert got["tars_like"] is True
    assert got["reclaim_i"] is None
    assert got["reclaim_within_wait"] is False


def test_expired_never_fills_or_cancels():
    close = np.full(10, 10.5)
    low = close - 0.1
    got = walk_h2_cancel_reclaim(
        close,
        low,
        h2_i=0,
        support_x0=0,
        support_y0=10.0,
        support_slope=0.0,
        width=2.0,
        min_wait=6,
        max_wait=8,
    )
    assert got["outcome"] == OUTCOME_EXPIRED


def test_close_broke_support_matches_error_pct():
    assert close_broke_support(15.85, 16.22, error_pct=1.2) is True
    # 16.22 * 0.988 = 16.025; 16.10 would not cancel
    assert close_broke_support(16.10, 16.22, error_pct=1.2) is False


def test_setup_row_span365():
    ok = {
        "h2_date": "2023-11-15",
        "start_date": "2023-04-14",
        "channel_width": 3.28,
        "support_y0": 11.57,
        "support_slope": 0.03,
        "h2_idx": 775,
        "support_x0": 626,
    }
    assert setup_row_ok(ok) is True
    wide = dict(ok)
    wide["start_date"] = "2022-01-01"
    assert setup_row_ok(wide) is False


def test_count_bucket():
    rows = [
        {"outcome": OUTCOME_FILLED},
        {"outcome": OUTCOME_CANCELLED, "shallow": True, "tars_like": True, "recovered_next": True, "reclaim_within_wait": True},
        {"outcome": OUTCOME_CANCELLED, "shallow": False, "tars_like": False, "recovered_next": False, "reclaim_within_wait": True},
        {"outcome": OUTCOME_EXPIRED},
    ]
    c = count_bucket(rows)
    assert c["n_setups"] == 4
    assert c["n_filled_first"] == 1
    assert c["n_cancelled"] == 2
    assert c["n_tars_like"] == 1
    assert c["n_tars_like_reclaim"] == 1
    assert c["n_cancel_reclaim"] == 2
