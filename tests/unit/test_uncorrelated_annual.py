"""Unit tests for utils.research.uncorrelated_annual."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.research.uncorrelated_annual import (
    correlation_matrix,
    greedy_uncorrelated,
    pick_for_lookback_year,
    simulate_uncorrelated_annual,
)


def _block_panel(n_days: int = 520) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two synthetic groups: A1/A2 highly correlated, B1/B2 highly correlated,
    groups nearly independent. Plus a fifth idiosyncratic name C.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-02", periods=n_days, freq="B")
    # Shared factors
    f_a = rng.normal(0, 0.01, size=n_days)
    f_b = rng.normal(0, 0.01, size=n_days)
    noise = {k: rng.normal(0, 0.002, size=n_days) for k in ("A1", "A2", "B1", "B2", "C")}
    rets = {
        "A1": f_a + noise["A1"],
        "A2": f_a + noise["A2"],
        "B1": f_b + noise["B1"],
        "B2": f_b + noise["B2"],
        "C": rng.normal(0, 0.01, size=n_days),
    }
    close = pd.DataFrame(
        {k: 100.0 * np.cumprod(1.0 + v) for k, v in rets.items()},
        index=idx,
    )
    volume = pd.DataFrame(1_000_000.0, index=idx, columns=close.columns)
    return close, volume


def test_greedy_picks_across_blocks():
    close, _ = _block_panel()
    # Use full 2018+2019 for a fat corr sample
    days = close.index[close.index.year == 2018]
    rets = close.loc[days].pct_change().iloc[1:]
    corr = rets.corr()
    picks, mean_pair = greedy_uncorrelated(corr, x=3)
    assert len(picks) == 3
    # Should not pick both A1 and A2 together before taking from B and C
    a_count = sum(1 for p in picks if p.startswith("A"))
    b_count = sum(1 for p in picks if p.startswith("B"))
    assert a_count <= 1
    assert b_count <= 1
    assert "C" in picks or a_count + b_count == 2
    assert mean_pair < 0.5


def test_correlation_matrix_year_slice():
    close, _ = _block_panel()
    corr = correlation_matrix(close, list(close.columns), 2018, min_periods=40)
    assert corr.shape[0] == 5
    assert corr.loc["A1", "A2"] > corr.loc["A1", "B1"]


def test_simulate_produces_equity():
    close, volume = _block_panel(n_days=780)
    # liquid_n large enough; all names pass ADV
    res = simulate_uncorrelated_annual(
        close,
        volume,
        eval_start="2019-01-01",
        eval_end="2020-12-31",
        x=3,
        liquid_n=5,
        min_adv=1.0,
        min_price=1.0,
        min_obs=100,
        cost_bps_rt=0.0,
    )
    assert len(res.equity_net) > 100
    assert float(res.invested.mean()) > 0.5
    assert any(p.symbols for p in res.picks)
    # Each hold year should have at most x names
    for p in res.picks:
        if p.symbols:
            assert len(p.symbols) <= 3


def test_pick_for_lookback_empty_year():
    close, volume = _block_panel(n_days=100)
    # Year with no data
    p = pick_for_lookback_year(
        close, volume, 2010, x=3, liquid_n=5, min_adv=1.0, min_price=1.0, min_obs=10
    )
    assert p.symbols == []
    assert p.hold_year == 2011
