"""Unit tests for utils.research.costs."""
from __future__ import annotations

import pandas as pd

from utils.research.costs import apply_turnover_costs, bps_to_frac, one_way_from_rt_bps, turnover_cost_series


def test_bps_helpers() -> None:
    assert abs(bps_to_frac(10) - 0.001) < 1e-12
    assert abs(one_way_from_rt_bps(10) - 0.0005) < 1e-12


def test_turnover_cost_on_full_rotation() -> None:
    idx = pd.bdate_range("2020-01-02", periods=3)
    w = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        index=idx,
        columns=["A", "B"],
    )
    cost = turnover_cost_series(w, cost_bps_rt=10.0)
    assert abs(float(cost.iloc[1]) - 0.001) < 1e-12
    assert abs(float(cost.iloc[2])) < 1e-12


def test_apply_turnover_costs_lags_weights() -> None:
    idx = pd.bdate_range("2020-01-02", periods=4)
    w = pd.DataFrame(0.0, index=idx, columns=["A"])
    w.iloc[:] = 1.0
    r = pd.DataFrame(0.01, index=idx, columns=["A"])
    gross, net, inv = apply_turnover_costs(w, r, cost_bps_rt=10.0, lag_weights=True)
    assert abs(float(gross.iloc[0])) < 1e-12
    assert float(inv.iloc[0]) == 0.0
    assert float(inv.iloc[1]) == 1.0
    assert float(net.iloc[1]) <= float(gross.iloc[1])
