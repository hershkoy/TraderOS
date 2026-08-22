"""Transaction cost helpers (basis points on turnover / notional)."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def bps_to_frac(bps: float) -> float:
    return float(bps) / 10000.0


def one_way_from_rt_bps(cost_bps_rt: float) -> float:
    """Split a round-trip bps cost equally across entry and exit."""
    return bps_to_frac(cost_bps_rt) / 2.0


def turnover_cost_series(weights: pd.DataFrame, cost_bps_rt: float = 10.0) -> pd.Series:
    """
    Daily cost as a fraction of equity.

    Full L1 weight change / 2 = one-way book traded; RT bps applied on that.
    """
    w = weights.fillna(0.0)
    w_change = w.diff().abs().sum(axis=1).fillna(0.0)
    return (w_change / 2.0) * bps_to_frac(cost_bps_rt)


def apply_turnover_costs(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps_rt: float = 10.0,
    lag_weights: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Vectorized long-only book: portfolio returns gross and net of RT costs.

    Returns (port_ret_gross, port_ret_net, invested_fraction).
    If lag_weights is True, today's weights earn tomorrow's close-to-close return
    (signal on close t, implement t+1).
    """
    w = weights.fillna(0.0)
    if lag_weights:
        w_use = w.shift(1).fillna(0.0)
    else:
        w_use = w
    r = asset_returns.reindex(index=w_use.index, columns=w_use.columns).fillna(0.0)
    port_gross = (w_use * r).sum(axis=1)
    cost = turnover_cost_series(w_use, cost_bps_rt=cost_bps_rt)
    port_net = port_gross - cost
    invested = (w_use.sum(axis=1) > 1e-12).astype(float)
    return port_gross, port_net, invested
