"""Unit tests for 12-1 XS momentum with PIT liquidity + SMA200 filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.research.xs_momentum import simulate_xs_momentum


def _panel(n_days: int = 420, n_syms: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(n_syms)]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    # S0 is the 12-1 winner: strong gain from t-12m to t-1m
    for i, c in enumerate(cols):
        drift = 0.0002 * (n_syms - i)
        noise = rng.normal(0.0, 0.002, size=n_days)
        close[c] = 50.0 + np.cumsum(drift + noise) + i
    close["S0"] = np.linspace(20.0, 200.0, n_days)
    vol = pd.DataFrame(1_000_000.0, index=idx, columns=cols)
    spy = pd.Series(np.linspace(200.0, 400.0, n_days), index=idx)
    return close, vol, spy


def test_xs_mom_goes_to_cash_when_spy_below_sma() -> None:
    close, vol, spy = _panel()
    spy[:] = 100.0
    spy.iloc[:50] = np.linspace(200.0, 100.0, 50)
    eq_g, eq_n, inv, notes = simulate_xs_momentum(
        close,
        vol,
        spy,
        eval_start="2019-01-01",
        eval_end="2019-12-31",
        top_n=2,
        liquid_n=6,
        min_price=1.0,
        min_adv=1.0,
        sma_period=20,
        cost_bps_rt=10.0,
    )
    assert "SPY>SMA" in notes
    assert float(inv.mean()) < 0.05
    assert abs(float(eq_n.iloc[-1]) - 1.0) < 0.02


def test_xs_mom_selects_top_names_when_risk_on() -> None:
    close, vol, spy = _panel()
    eq_g, eq_n, inv, notes = simulate_xs_momentum(
        close,
        vol,
        spy,
        eval_start="2019-06-01",
        eval_end="2019-12-31",
        top_n=2,
        liquid_n=6,
        min_price=1.0,
        min_adv=1.0,
        sma_period=50,
        cost_bps_rt=10.0,
    )
    assert float(inv.mean()) > 0.5
    assert float(eq_n.iloc[-1]) != float(eq_g.iloc[-1]) or float(eq_n.iloc[-1]) > 0.5
    assert "rebalances_on=" in notes
    assert len(eq_n) > 50
