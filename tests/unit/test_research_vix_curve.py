"""Unit tests for VIX term-structure overlay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.research.vix_curve import curve_ratio, risk_on_weights, simulate_vix_curve_overlay


def test_curve_ratio_contango_vs_backwardation() -> None:
    idx = pd.bdate_range("2020-01-02", periods=5)
    vix = pd.Series([12.0, 14.0, 25.0, 30.0, 18.0], index=idx)
    vix3m = pd.Series([15.0, 16.0, 20.0, 22.0, 19.0], index=idx)
    r = curve_ratio(vix, vix3m)
    assert float(r.iloc[0]) < 1.0
    assert float(r.iloc[2]) > 1.0


def test_binary_weights_flip_on_backwardation() -> None:
    idx = pd.bdate_range("2020-01-02", periods=4)
    ratio = pd.Series([0.8, 0.95, 1.05, 1.2], index=idx)
    w = risk_on_weights(ratio, mode="binary", back_thresh=1.0)
    assert list(w.values) == [1.0, 1.0, 0.0, 0.0]


def test_soft_weights_between_0_and_1() -> None:
    idx = pd.bdate_range("2020-01-02", periods=3)
    ratio = pd.Series([0.5, 1.0, 2.0], index=idx)
    w = risk_on_weights(ratio, mode="soft")
    assert float(w.iloc[0]) == 1.0
    assert abs(float(w.iloc[1]) - 1.0) < 1e-9
    assert abs(float(w.iloc[2]) - 0.5) < 1e-9


def test_simulate_exits_in_backwardation() -> None:
    idx = pd.bdate_range("2018-01-02", periods=120)
    rng = np.random.default_rng(7)
    spy_rets = rng.normal(0.0005, 0.01, size=120)
    spy = pd.Series(100 * np.cumprod(1.0 + spy_rets), index=idx)
    # Contango first half, backwardation second half
    vix3m = pd.Series(18.0, index=idx)
    vix = pd.Series(np.where(np.arange(120) < 60, 14.0, 22.0), index=idx)

    eq_g, eq_n, inv, notes = simulate_vix_curve_overlay(
        spy,
        vix,
        vix3m,
        None,
        eval_start="2018-02-01",
        eval_end="2018-06-15",
        mode="binary",
        cost_bps_rt=5.0,
    )
    assert "vix_curve" in notes
    assert len(eq_n) > 40
    # After lag, second regime should be mostly out of SPY
    late = inv.iloc[len(inv) // 2 :]
    assert float(late.mean()) < 0.25
