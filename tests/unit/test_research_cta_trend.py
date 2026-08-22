"""Unit tests for CTA trend sleeve."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.research.cta_trend import blend_spy_cta, simulate_cta_sleeve


def test_cta_goes_short_in_downtrend() -> None:
    idx = pd.bdate_range("2018-01-02", periods=400)
    # Asset A rising, B falling
    a = pd.Series(np.linspace(100, 200, 400), index=idx)
    b = pd.Series(np.linspace(200, 50, 400), index=idx)
    closes = pd.DataFrame({"A": a, "B": b})
    eq_g, eq_n, exp, notes = simulate_cta_sleeve(
        closes,
        eval_start="2019-01-01",
        eval_end="2019-06-30",
        method="sma200",
        sma_period=100,
        cost_bps_rt=10.0,
        rebalance="monthly",
    )
    assert "CTA" in notes
    assert len(eq_n) > 20
    assert float(exp.mean()) > 0


def test_blend_spy_cta() -> None:
    idx = pd.bdate_range("2020-01-02", periods=10)
    spy = pd.Series(np.linspace(1.0, 1.1, 10), index=idx)
    cta = pd.Series(np.linspace(1.0, 1.2, 10), index=idx)
    blend = blend_spy_cta(spy, cta, spy_weight=0.7)
    assert abs(float(blend.iloc[0]) - 1.0) < 1e-12
    # End ~ 0.7*1.1 + 0.3*1.2 = 1.13
    assert abs(float(blend.iloc[-1]) - 1.13) < 1e-9
