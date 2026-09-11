"""Unit tests for channel_touch_robustness diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from channel_touch_robustness import (  # noqa: E402
    apply_max_open,
    drop_top_n_winners,
    robust_score,
    summarize_gains,
    tail_dependency_ratio,
    winsorize_gains,
)


def test_drop_top_n_removes_largest():
    g = np.array([1.0, 10.0, 2.0, 50.0, -3.0])
    out = drop_top_n_winners(g, 1)
    assert 50.0 not in out
    assert len(out) == 4


def test_winsorize_caps():
    g = np.array([1.0, 100.0, -2.0])
    out = winsorize_gains(g, cap_pct=20.0)
    assert out.max() == 20.0
    assert out[2] == -2.0


def test_tail_dependency():
    g = np.array([100.0, 50.0, 50.0, 10.0, -5.0])
    t = tail_dependency_ratio(g, top_n=3)
    assert t["tail_dependency_ratio"] == round(200.0 / 210.0, 4)


def test_summarize_positive():
    s = summarize_gains(np.array([1.0, 2.0, -1.0]), "x")
    assert s["n"] == 3
    assert s["expectancy_pct"] == round(2.0 / 3.0, 4)


def test_robust_score_blends_pooled_and_drop_top3():
    g = np.array([2.0, 2.0, 2.0, -1.0, -1.0])
    s = robust_score(g)
    assert s["n"] == 5
    assert s["n_drop"] == 2
    assert s["expectancy_pct"] == 0.8
    assert s["profit_factor"] == 3.0
    assert s["expectancy_drop_pct"] == -1.0
    assert s["profit_factor_drop"] == 0.0
    assert s["e_rob"] == -0.1
    assert s["pf_rob"] == 0.0
    assert s["r"] == round(-0.1 * (0.0 - 1.0), 4)


def test_robust_score_empty():
    s = robust_score(np.array([]))
    assert s["n"] == 0
    assert s["r"] is None


def test_apply_max_open():
    df = pd.DataFrame(
        [
            {"buy_date": "2024-01-01", "sell_date": "2024-01-20", "stock": "A", "rs_spy_126d": 1},
            {"buy_date": "2024-01-05", "sell_date": "2024-01-25", "stock": "B", "rs_spy_126d": 2},
            {"buy_date": "2024-01-10", "sell_date": "2024-01-30", "stock": "C", "rs_spy_126d": 3},
        ]
    )
    kept = apply_max_open(df, 2)
    assert len(kept) == 2


def test_n_open_at_entry_ignores_later_buys():
    from channel_touch_robustness import n_open_at_entry

    df = pd.DataFrame(
        {
            "buy_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01"]),
            "sell_date": pd.to_datetime(["2024-01-31", "2024-01-20", "2024-02-10"]),
            "stock": ["A", "B", "C"],
        }
    )
    n = n_open_at_entry(df)
    assert list(n) == [0, 1, 0]


def test_skip_crowded_days_drops_busy_session():
    from channel_touch_robustness import cap_same_day, skip_crowded_days

    df = pd.DataFrame(
        {
            "buy_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"]),
            "sell_date": pd.to_datetime(["2024-01-10"] * 4),
            "stock": ["A", "B", "C", "D"],
            "wait_bars": [40, 10, 20, 5],
        }
    )
    quiet = skip_crowded_days(df, max_names=2)
    assert set(quiet["stock"]) == {"D"}
    cap = cap_same_day(df, 2, tie_break="wait")
    jan2 = cap.loc[pd.to_datetime(cap["buy_date"]) == pd.Timestamp("2024-01-02")]
    assert list(jan2["stock"]) == ["A", "C"]
