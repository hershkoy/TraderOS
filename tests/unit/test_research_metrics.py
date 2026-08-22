"""Unit tests for utils.research.metrics."""
from __future__ import annotations

import pandas as pd

from utils.research.metrics import (
    annual_returns,
    clip_index,
    equity_from_returns,
    max_drawdown,
    passes_phase5_gates,
    perf_stats,
    window_equity,
)


def _eq_from_rets(rets) -> pd.Series:
    idx = pd.bdate_range("2020-01-02", periods=len(rets))
    return equity_from_returns(pd.Series(rets, index=idx))


def test_max_drawdown_and_sortino() -> None:
    eq = _eq_from_rets([0.01, 0.01, -0.20, 0.02, 0.02])
    mdd = max_drawdown(eq)
    assert mdd < -0.15
    stats = perf_stats("t", eq)
    assert stats.sortino != 0.0
    assert stats.sharpe != 0.0
    assert stats.max_drawdown == mdd


def test_phase5_gates_require_sharpe_and_mdd() -> None:
    idx = pd.bdate_range("2018-01-02", periods=260)
    spy_eq = pd.Series(1.0 + 0.001 * pd.Series(range(260)).values, index=idx)
    noisy = spy_eq.copy()
    noisy.iloc[100] = float(noisy.iloc[99]) * 0.5
    spy = perf_stats("SPY", spy_eq)
    cand = perf_stats("C", noisy)
    ok, msg = passes_phase5_gates(cand, spy)
    assert not ok
    assert "MDD" in msg or "Sharpe" in msg


def test_phase5_gates_pass() -> None:
    spy_s = perf_stats("SPY", pd.Series([1.0, 1.1], index=pd.bdate_range("2020-01-02", periods=2)))
    st = perf_stats("S", pd.Series([1.0, 1.2], index=pd.bdate_range("2020-01-02", periods=2)))
    st.sharpe = 1.2
    st.max_drawdown = -0.05
    spy_s.sharpe = 0.8
    spy_s.max_drawdown = -0.20
    ok, msg = passes_phase5_gates(st, spy_s)
    assert ok, msg


def test_clip_and_window_rebase() -> None:
    idx = pd.bdate_range("2018-01-02", periods=400)
    eq = pd.Series(range(1, 401), index=idx, dtype=float)
    inv = pd.Series(1.0, index=idx)
    clipped = clip_index(eq, "2018-06-01", "2018-12-31")
    assert not clipped.empty
    assert clipped.index[0] >= pd.Timestamp("2018-06-01")
    w, winv = window_equity(eq, inv, "2018-06-01", "2018-12-31")
    assert abs(float(w.iloc[0]) - 1.0) < 1e-12
    assert winv is not None and len(winv) == len(w)


def test_annual_returns() -> None:
    idx = pd.bdate_range("2020-01-02", "2021-12-31")
    eq = pd.Series(1.0, index=idx)
    first_2021 = idx[idx.year == 2021][0]
    eq.loc[eq.index >= first_2021] = 1.1
    ann = annual_returns(eq)
    assert 2020 in ann.index and 2021 in ann.index
    assert abs(float(ann.loc[2021]) - 0.10) < 0.02
