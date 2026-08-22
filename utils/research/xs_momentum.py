"""12-1 cross-sectional momentum with PIT liquidity and SPY SMA200 filter."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.research.costs import apply_turnover_costs
from utils.research.metrics import equity_from_returns
from utils.research.panel import adv_dollar, align_spy_regime, month_ends, top_n_liquid_mask


def simulate_xs_momentum(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    spy_close: pd.Series,
    *,
    eval_start: str,
    eval_end: str,
    top_n: int = 20,
    liquid_n: int = 500,
    lookback_months: int = 12,
    skip_months: int = 1,
    cost_bps_rt: float = 10.0,
    min_price: float = 5.0,
    min_adv: float = 1_000_000.0,
    sma_period: int = 200,
    adv_window: int = 30,
) -> Tuple[pd.Series, pd.Series, pd.Series, str]:
    """
    Monthly 12-1 equal-weight top_n among PIT top liquid_n ADV names.
    Invest only when SPY close > SMA(sma_period). Cash earns 0%.

    Returns (eq_gross, eq_net, invested, notes).
    """
    close = close.astype(float).sort_index()
    volume = volume.astype(float).reindex_like(close)

    adv = adv_dollar(close, volume, window=adv_window)
    liquid = top_n_liquid_mask(
        adv, top_n=liquid_n, min_adv=min_adv, close=close, min_price=min_price
    )
    _, _, risk_on = align_spy_regime(spy_close, close.index, sma_period=sma_period)

    skip_days = int(skip_months) * 21
    form_days = int(lookback_months) * 21
    me_idx = month_ends(close.index)

    holdings = {}
    for dt in me_idx:
        loc = close.index.get_loc(dt)
        if isinstance(loc, slice):
            continue
        if loc < form_days + skip_days:
            continue
        if dt not in liquid.index:
            continue
        on = False
        if dt in risk_on.index and pd.notna(risk_on.loc[dt]):
            on = bool(risk_on.loc[dt])
        if not on:
            holdings[dt] = []
            continue
        elig = liquid.loc[dt]
        names = elig.index[elig.fillna(False).astype(bool)].tolist()
        if len(names) < top_n:
            continue
        end_px = close.iloc[loc - skip_days][names]
        start_px = close.iloc[loc - form_days - skip_days][names]
        mom = (end_px / start_px) - 1.0
        mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
        if len(mom) < top_n:
            continue
        holdings[dt] = mom.nlargest(top_n).index.tolist()

    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    all_days = close.index[(close.index >= s) & (close.index <= e)]
    weight = pd.DataFrame(0.0, index=all_days, columns=close.columns)
    me_sorted = [dt for dt in sorted(holdings.keys()) if dt <= e]
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
        if not picks:
            continue
        w = 1.0 / len(picks)
        for sym in picks:
            if sym in weight.columns:
                weight.loc[end_hold, sym] = w

    daily_ret = close.pct_change(fill_method=None)
    port_gross, port_net, invested = apply_turnover_costs(
        weight, daily_ret, cost_bps_rt=cost_bps_rt, lag_weights=True
    )
    eq_gross = equity_from_returns(port_gross)
    eq_net = equity_from_returns(port_net)
    n_on = sum(1 for v in holdings.values() if v)
    notes = (
        f"PIT top{liquid_n} ADV{adv_window}d min_px={min_price:g} min_adv={min_adv:.0f}; "
        f"12-{skip_months}m top_n={top_n}; SPY>SMA{sma_period}; cost={cost_bps_rt:.0f}bps RT; "
        f"rebalances_on={n_on}/{len(holdings)}"
    )
    return eq_gross, eq_net, invested, notes
