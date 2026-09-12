"""Equal-weight universe breadth: share of names above their own SMA20.

Prior-session only for last-15m fills (15:45 ET). Breadth on day t is the
fraction of names with a valid SMA20 whose close > SMA20. Missing SMA does
not count as below.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.research.causal_regime_hmm import lookup_asof
from utils.research.channel_touch_entry_features import completed_asof

SMA_LEN = 20
MEDIAN_MIN = 60


def breadth_above_sma(
    close: pd.DataFrame,
    *,
    sma_len: int = SMA_LEN,
) -> pd.Series:
    """Daily fraction of columns with close > SMA. Index = session dates."""
    if close is None or close.empty:
        return pd.Series(dtype=float)
    px = close.astype(float).copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is not None:
        px.index = px.index.tz_convert(None)
    px.index = pd.DatetimeIndex(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    sma = px.rolling(int(sma_len), min_periods=int(sma_len)).mean()
    valid = sma.notna() & px.notna()
    n = valid.sum(axis=1).replace(0, np.nan)
    above = ((px > sma) & valid).sum(axis=1)
    out = (above / n).astype(float)
    out.name = "breadth_sma20"
    return out


def breadth_expanding_median(breadth: pd.Series, *, min_periods: int = MEDIAN_MIN) -> pd.Series:
    """Median of breadth through t-1 (shift then expanding)."""
    b = pd.to_numeric(breadth, errors="coerce").sort_index()
    return b.shift(1).expanding(min_periods=int(min_periods)).median()


def attach_breadth(
    trades: pd.DataFrame,
    breadth: pd.Series,
    *,
    median: Optional[pd.Series] = None,
) -> pd.DataFrame:
    out = trades.copy()
    bvals = []
    mvals = []
    asofs = []
    for _, row in out.iterrows():
        asof = completed_asof(row, series_is_daily=True)
        asofs.append(asof)
        bvals.append(None if asof is None else lookup_asof(breadth, asof))
        if median is None or asof is None:
            mvals.append(None)
        else:
            mvals.append(lookup_asof(median, asof))
    out["regime_asof"] = asofs
    out["breadth_sma20"] = bvals
    if median is not None:
        out["breadth_sma20_med"] = mvals
    return out


def keep_if_breadth_at_least(trades: pd.DataFrame, *, min_breadth: float) -> pd.Series:
    """Keep when breadth missing or breadth >= min. Skip weak internals."""
    b = pd.to_numeric(trades["breadth_sma20"], errors="coerce")
    return b.isna() | (b >= float(min_breadth))


def keep_if_breadth_at_least_median(trades: pd.DataFrame) -> pd.Series:
    """Keep when either series missing or breadth >= expanding median through t-1."""
    b = pd.to_numeric(trades["breadth_sma20"], errors="coerce")
    m = pd.to_numeric(trades["breadth_sma20_med"], errors="coerce")
    return b.isna() | m.isna() | (b >= m)
