"""Unit tests for low-vol basket."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.research.low_vol import simulate_low_vol_basket


def test_low_vol_prefers_calm_names() -> None:
    idx = pd.bdate_range("2019-01-02", periods=300)
    rng = np.random.default_rng(0)
    calm = 100 * np.cumprod(1.0 + rng.normal(0.0003, 0.005, size=300))
    wild = 100 * np.cumprod(1.0 + rng.normal(0.0003, 0.04, size=300))
    close = pd.DataFrame(
        {
            "C1": calm,
            "C2": calm * 1.01,
            "C3": calm * 0.99,
            "W1": wild,
            "W2": wild * 1.02,
            "W3": wild * 0.98,
        },
        index=idx,
    )
    vol = pd.DataFrame(2_000_000.0, index=idx, columns=close.columns)
    eq_g, eq_n, inv, notes = simulate_low_vol_basket(
        close,
        vol,
        eval_start="2019-06-01",
        eval_end="2020-01-31",
        liquid_n=6,
        hold_n=2,
        vol_window=40,
        min_price=1.0,
        min_adv=1.0,
        cost_bps_rt=10.0,
    )
    assert "low_vol" in notes
    assert float(inv.mean()) > 0.3
    assert len(eq_n) > 50
