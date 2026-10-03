"""Spot CSV parse for the SPX BWB / bull-put loaders."""
from __future__ import annotations

import pandas as pd

from utils.research.spx_put_bwb import theta_csv_to_ohlc


def test_theta_csv_keeps_close_when_index_changes():
    raw = pd.DataFrame(
        {
            "created": ["2024-03-01T17:20:13.401", "2024-03-04T17:20:56.749"],
            "open": [5098.51, 5130.99],
            "high": [5140.33, 5149.67],
            "low": [5094.16, 5127.18],
            "close": [5137.08, 5130.95],
        }
    )
    out = theta_csv_to_ohlc(raw)
    assert len(out) == 2
    assert abs(float(out.iloc[0]["close"]) - 5137.08) < 1e-6
    assert out.index[0] == pd.Timestamp("2024-03-01")
    assert out.index[1] == pd.Timestamp("2024-03-04")
