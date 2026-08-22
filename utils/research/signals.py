"""Vectorized panel signals (RSI, SMA, N-day dump)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma_wide(close: pd.DataFrame, period: int) -> pd.DataFrame:
    return close.astype(float).rolling(int(period), min_periods=int(period)).mean()


def rsi_wide(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder-style RSI via rolling mean of gains/losses (matches indicators.momentum.RSI)."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0).rolling(int(period), min_periods=int(period)).mean()
    loss = (-delta.clip(upper=0.0)).rolling(int(period), min_periods=int(period)).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def n_day_return(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.astype(float).pct_change(int(n), fill_method=None)


def dump_signal(close: pd.DataFrame, n: int = 3, thresh: float = -0.05) -> pd.DataFrame:
    """True when N-day return is at or below thresh (e.g. -5%)."""
    return n_day_return(close, n) <= float(thresh)
