"""Unit tests for L3 wait-12 occupancy helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from overlay_l3_wait12_occupancy import _pooled, _to_occ_rows  # noqa: E402


def test_pooled_and_occ_rows():
    df = pd.DataFrame(
        {
            "stock": ["AAA", "BBB"],
            "buy_date": ["2022-01-03", "2022-01-04"],
            "sell_date": ["2022-01-05", "2022-01-06"],
            "gain_pct": [1.0, -0.5],
            "gain_pct_net": [0.8, -0.7],
        }
    )
    p = _pooled(df)
    assert p["n"] == 2
    assert abs(p["E"] - 0.05) < 1e-6
    rows = _to_occ_rows(df)
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["gain_pct"] == 0.8
    assert rows[1]["entry_date"] == "2022-01-04"
