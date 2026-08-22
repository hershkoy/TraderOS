"""Volatility-managed equity (Moreira & Muir style) on SPY."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.research.costs import bps_to_frac
from utils.research.metrics import TRADING_DAYS, equity_from_returns


def realized_vol(
    returns: pd.Series,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """Annualized realized vol from daily returns."""
    mp = min_periods if min_periods is not None else max(5, window // 2)
    return returns.rolling(int(window), min_periods=mp).std(ddof=0) * np.sqrt(TRADING_DAYS)


def vol_target_weights(
    returns: pd.Series,
    *,
    target_vol: float = 0.12,
    lookback: int = 20,
    leverage_cap: float = 1.5,
) -> pd.Series:
    """
    w_t = min(cap, target / sigma_hat_t). Signal on close t; apply at t+1.
    """
    sig = realized_vol(returns, window=lookback)
    w = (float(target_vol) / sig.replace(0.0, np.nan)).clip(upper=float(leverage_cap))
    return w.fillna(0.0)


def simulate_vol_target(
    spy_close: pd.Series,
    cash_close: pd.Series | None = None,
    *,
    eval_start: str,
    eval_end: str,
    target_vol: float = 0.12,
    lookback: int = 20,
    leverage_cap: float = 1.5,
    cost_bps_rt: float = 5.0,
    rebalance: str = "daily",
) -> Tuple[pd.Series, pd.Series, pd.Series, str]:
    """
    Scale SPY exposure to target vol; residual earns cash return (BIL) or 0%.

    rebalance: 'daily' or 'weekly' (W-FRI signal, hold until next Friday).
    Returns (eq_gross, eq_net, invested_spy_weight, notes).
    """
    spy = spy_close.astype(float).copy()
    spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    spy_ret = spy.pct_change(fill_method=None)

    if cash_close is not None and not cash_close.empty:
        cash = cash_close.astype(float).copy()
        cash.index = pd.DatetimeIndex(cash.index).tz_localize(None).normalize()
        cash = cash[~cash.index.duplicated(keep="last")].sort_index()
        cash = cash.reindex(spy.index).ffill()
        cash_ret = cash.pct_change(fill_method=None).fillna(0.0)
        cash_note = "BIL"
    else:
        cash_ret = pd.Series(0.0, index=spy.index)
        cash_note = "cash0"

    w_raw = vol_target_weights(
        spy_ret, target_vol=target_vol, lookback=lookback, leverage_cap=leverage_cap
    )
    if rebalance == "weekly":
        # Hold Friday signal until next Friday (forward-fill on weekdays)
        fri = w_raw.resample("W-FRI").last()
        w_sig = fri.reindex(spy.index, method="ffill")
    else:
        w_sig = w_raw

    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    mask = (spy.index >= s) & (spy.index <= e)
    w_sig = w_sig.loc[mask]
    spy_ret_e = spy_ret.reindex(w_sig.index).fillna(0.0)
    cash_ret_e = cash_ret.reindex(w_sig.index).fillna(0.0)

    # Implement t+1
    w = w_sig.shift(1).fillna(0.0)
    # Residual weight in cash (can be negative if leverage > 1: borrow at cash rate)
    w_cash = 1.0 - w
    port_gross = w * spy_ret_e + w_cash * cash_ret_e

    # Turnover cost on SPY weight changes only
    w_change = w.diff().abs().fillna(0.0)
    # L1 one-way fraction of book traded on SPY sleeve
    cost = (w_change / 2.0) * bps_to_frac(cost_bps_rt)
    port_net = port_gross - cost

    eq_gross = equity_from_returns(port_gross)
    eq_net = equity_from_returns(port_net)
    invested = w.clip(lower=0.0)  # report long equity exposure
    notes = (
        f"vol_target={target_vol:.0%} lookback={lookback}d cap={leverage_cap:g}x "
        f"rebal={rebalance} cash={cash_note} cost={cost_bps_rt:.0f}bps RT; "
        f"avg_w={float(w.mean()):.2f} max_w={float(w.max()):.2f}"
    )
    return eq_gross, eq_net, invested, notes
