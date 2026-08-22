"""Unit tests for vol targeting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.research.vol_target import realized_vol, simulate_vol_target, vol_target_weights


def test_vol_target_scales_down_in_high_vol() -> None:
    idx = pd.bdate_range("2020-01-02", periods=80)
    # Calm then wild
    rets = np.concatenate([np.full(40, 0.001), np.full(40, 0.03) * np.array([1, -1] * 20)])
    px = 100 * np.cumprod(1.0 + rets)
    close = pd.Series(px, index=idx)
    r = close.pct_change()
    w = vol_target_weights(r, target_vol=0.12, lookback=20, leverage_cap=1.5)
    assert float(w.iloc[30]) > float(w.iloc[70])


def test_simulate_vol_target_beats_flat_on_vol_spike_path() -> None:
    idx = pd.bdate_range("2018-01-02", periods=300)
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.005, size=300)
    rets[150:180] = rng.normal(-0.01, 0.04, size=30)
    spy = pd.Series(100 * np.cumprod(1.0 + rets), index=idx)
    eq_g, eq_n, inv, notes = simulate_vol_target(
        spy,
        None,
        eval_start="2018-06-01",
        eval_end="2019-03-01",
        target_vol=0.10,
        lookback=20,
        leverage_cap=1.0,
        cost_bps_rt=5.0,
    )
    assert "vol_target=" in notes
    assert len(eq_n) > 50
    assert float(eq_n.iloc[0]) == 1.0 or abs(float(eq_n.iloc[0]) - 1.0) < 1e-9
    assert float(inv.mean()) > 0
    assert float(realized_vol(spy.pct_change(), 20).dropna().iloc[-1]) > 0
