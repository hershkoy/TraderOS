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
