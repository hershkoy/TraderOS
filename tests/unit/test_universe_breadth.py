"""Universe SMA20 breadth helpers."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from utils.research.universe_breadth import (
    attach_breadth,
    breadth_above_sma,
    breadth_expanding_median,
    keep_if_breadth_at_least,
    keep_if_breadth_at_least_median,
)


def test_breadth_counts_only_names_with_sma():
    idx = pd.bdate_range("2020-01-02", periods=25)
    a = pd.Series(np.linspace(10, 14, 25), index=idx)
    b = pd.Series(np.linspace(20, 10, 25), index=idx)
    c = pd.Series([np.nan] * 25, index=idx)
    close = pd.DataFrame({"A": a, "B": b, "C": c})
    br = breadth_above_sma(close, sma_len=20)
    last = float(br.iloc[-1])
    assert last == 0.5


def test_breadth_median_is_causal():
    idx = pd.bdate_range("2020-01-02", periods=80)
    b = pd.Series(np.linspace(0.2, 0.8, 80), index=idx)
    med = breadth_expanding_median(b, min_periods=10)
    assert pd.isna(med.iloc[9])
    assert pd.notna(med.iloc[10])
    assert abs(float(med.iloc[-1]) - float(b.iloc[:-1].median())) < 1e-12


def test_last_15m_fill_joins_prior_session_breadth():
    idx = pd.bdate_range("2022-03-14", periods=3)
    breadth = pd.Series([0.35, 0.70, 0.80], index=idx)
    trades = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": "2022-03-15",
                "buy_time": datetime(2022, 3, 15, 15, 45),
                "gain_pct": 1.0,
            }
        ]
    )
    out = attach_breadth(trades, breadth)
    assert abs(float(out.loc[0, "breadth_sma20"]) - 0.35) < 1e-12
    keep = keep_if_breadth_at_least(out, min_breadth=0.50)
    assert bool(keep.iloc[0]) is False


def test_keep_breadth_missing_does_not_skip():
    trades = pd.DataFrame({"breadth_sma20": [np.nan, 0.6, 0.2]})
    keep = keep_if_breadth_at_least(trades, min_breadth=0.50)
    assert list(keep.astype(int)) == [1, 1, 0]


def test_keep_below_expanding_median():
    trades = pd.DataFrame(
        {
            "breadth_sma20": [0.40, 0.55],
            "breadth_sma20_med": [0.50, 0.50],
        }
    )
    keep = keep_if_breadth_at_least_median(trades)
    assert list(keep.astype(int)) == [0, 1]
