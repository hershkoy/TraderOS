"""Low-volatility cross-sectional basket (realized vol only; no fundamentals)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.research.costs import apply_turnover_costs
from utils.research.metrics import TRADING_DAYS, equity_from_returns
from utils.research.panel import adv_dollar, month_ends, top_n_liquid_mask


def simulate_low_vol_basket(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    eval_start: str,
    eval_end: str,
    liquid_n: int = 500,
    hold_n: int = 50,
    vol_window: int = 60,
    cost_bps_rt: float = 10.0,
    min_price: float = 10.0,
    min_adv: float = 1_000_000.0,
    inverse_vol_weight: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, str]:
    """
    Monthly rebalance into lowest realized-vol names among PIT liquid universe.

    Weight: equal or inverse-vol. No ROE/FCF (not in market_data).
    """
    close = close.astype(float).sort_index()
    volume = volume.astype(float).reindex_like(close)
    rets = close.pct_change(fill_method=None)
    vol = rets.rolling(int(vol_window), min_periods=max(10, vol_window // 2)).std(ddof=0) * np.sqrt(
        TRADING_DAYS
    )
    liquid = top_n_liquid_mask(
        adv_dollar(close, volume, window=30),
        top_n=liquid_n,
        min_adv=min_adv,
        close=close,
        min_price=min_price,
    )

    me = month_ends(close.index)
    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    all_days = close.index[(close.index >= s) & (close.index <= e)]
    weight = pd.DataFrame(0.0, index=all_days, columns=close.columns)

    holdings = {}
    for dt in me:
        if dt < s or dt > e:
            continue
        if dt not in liquid.index or dt not in vol.index:
            continue
        elig = liquid.loc[dt].fillna(False).astype(bool)
        names = elig.index[elig].tolist()
        if len(names) < hold_n:
            continue
        v = vol.loc[dt, names].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < hold_n:
            continue
        picks = v.nsmallest(hold_n)
        holdings[dt] = picks

    me_sorted = sorted(holdings.keys())
    for i, dt in enumerate(me_sorted):
        start_hold = all_days[all_days > dt]
        if start_hold.empty:
            continue
        start_d = start_hold[0]
        if i + 1 < len(me_sorted):
            end_hold = all_days[(all_days >= start_d) & (all_days <= me_sorted[i + 1])]
        else:
            end_hold = all_days[all_days >= start_d]
        picks = holdings[dt]
        if inverse_vol_weight:
            inv = 1.0 / picks.clip(lower=1e-6)
            w = inv / inv.sum()
        else:
            w = pd.Series(1.0 / len(picks), index=picks.index)
        for sym, wv in w.items():
            if sym in weight.columns:
                weight.loc[end_hold, sym] = float(wv)

    daily_ret = rets.reindex(weight.index).fillna(0.0)
    port_g, port_n, invested = apply_turnover_costs(
        weight, daily_ret, cost_bps_rt=cost_bps_rt, lag_weights=True
    )
    notes = (
        f"low_vol{vol_window}d hold_n={hold_n} liquid={liquid_n} "
        f"inv_vol={inverse_vol_weight} cost={cost_bps_rt:.0f}bps RT; "
        f"rebalances={len(holdings)}"
    )
    return equity_from_returns(port_g), equity_from_returns(port_n), invested, notes
