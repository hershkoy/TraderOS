"""Unit tests for BS short-put overlay on last-15m signals."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from utils.options.greeks import black_scholes_put, calculate_greeks
from utils.research.last_15m_short_put_bs import (
    EXIT_EXPIRY,
    EXIT_HARD_STOP,
    EXIT_TP,
    apply_short_put_book,
    iv_from_atr_pct,
    round_strike,
    simulate_short_put,
    strike_for_put_delta,
)


def _panel(days, start, *, opens, highs, lows, closes):
    idx = pd.bdate_range(start=start, periods=days)
    n = len(idx)

    def _expand(v):
        if np.isscalar(v):
            return [v] * n
        return list(v)

    return pd.DataFrame(
        {
            "open": _expand(opens),
            "high": _expand(highs),
            "low": _expand(lows),
            "close": _expand(closes),
            "volume": [1_000_000] * n,
        },
        index=idx,
    )


def _row(**kwargs):
    base = {
        "stock": "TEST",
        "buy_date": "2024-01-02",
        "buy_price": 100.0,
        "hard_stop_price": 94.0,
        "atr_1d_pct": 2.0,
        "gain_pct": 1.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_round_strike_and_iv():
    assert round_strike(100.4) == 100.0
    assert round_strike(12.3) == 12.5
    iv = iv_from_atr_pct(2.0)
    assert 0.15 <= iv <= 0.90
    assert abs(iv - 2.0 / 100.0 * np.sqrt(252)) < 1e-9


def test_strike_for_put_delta_near_target():
    S, T, r, sigma = 100.0, 45 / 365.25, 0.05, 0.35
    k = strike_for_put_delta(S, T, r, sigma, target_delta=-0.30)
    d = calculate_greeks(S, k, T, r, sigma, "P")["delta"]
    assert -0.40 <= d <= -0.20


def test_tp50_on_grind_higher():
    start = date(2024, 1, 2)
    panel = _panel(40, start, opens=100, highs=110, lows=99, closes=105)
    row = _row(buy_date="2024-01-02", buy_price=100.0, atr_1d_pct=3.0)
    got = simulate_short_put(row, panel, dte=45, tp_frac=0.50, iv_override=0.40)
    assert got.filled
    assert got.exit_reason == EXIT_TP
    assert got.gain_cash_secured_pct is not None
    assert got.gain_cash_secured_pct > 0
    assert abs(got.exit_premium - 0.5 * got.credit) < 1e-6


def test_hard_stop_closes_put():
    start = date(2024, 1, 2)
    lows = [99.0] * 5 + [93.0] + [95.0] * 20
    closes = [100.0] * 5 + [94.0] + [96.0] * 20
    panel = _panel(len(lows), start, opens=100, highs=101, lows=lows, closes=closes)
    row = _row(buy_date="2024-01-02", buy_price=100.0, hard_stop_price=94.0, atr_1d_pct=2.0)
    got = simulate_short_put(row, panel, dte=45, iv_override=0.35)
    assert got.filled
    assert got.exit_reason == EXIT_HARD_STOP
    assert got.gain_cash_secured_pct is not None
    assert got.gain_cash_secured_pct < 0
    assert got.exit_premium > got.credit
    stressed = simulate_short_put(
        row, panel, dte=45, iv_override=0.35, stop_iv_mult=1.5
    )
    assert stressed.exit_premium > got.exit_premium
    assert stressed.gain_cash_secured_pct < got.gain_cash_secured_pct


def test_expiry_settlement_otm():
    start = date(2024, 1, 2)
    # Flat near entry; tp_frac=0 disables take-profit so we settle at expiry.
    panel = _panel(8, start, opens=100, highs=100.2, lows=99.8, closes=100.0)
    row = _row(buy_date="2024-01-02", buy_price=100.0, hard_stop_price=90.0, atr_1d_pct=2.0)
    got = simulate_short_put(row, panel, dte=5, iv_override=0.30, tp_frac=0.0)
    assert got.filled
    assert got.exit_reason == EXIT_EXPIRY
    assert got.exit_premium == 0.0
    assert got.gain_rop_pct is not None
    assert abs(got.gain_rop_pct - 100.0) < 1e-6


def test_apply_book_smoke():
    start = date(2024, 1, 2)
    panel = _panel(30, start, opens=100, highs=108, lows=99, closes=104)
    trades = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": "2024-01-02",
                "buy_price": 100.0,
                "hard_stop_price": 94.0,
                "atr_1d_pct": 2.5,
                "gain_pct": 5.0,
            }
        ]
    )
    out = apply_short_put_book(trades, {"AAA": panel}, dte=30, iv_override=0.35)
    assert len(out) == 1
    assert bool(out.iloc[0]["put_ok"]) is True
    assert out.iloc[0]["gain_pct"] is not None


def test_credit_matches_bs_atm():
    row = _row()
    start = date(2024, 1, 2)
    panel = _panel(5, start, opens=100, highs=101, lows=99, closes=100.5)
    credit = black_scholes_put(100.0, 100.0, 45 / 365.25, 0.05, 0.30)
    got = simulate_short_put(row, panel, dte=45, iv_override=0.30, strike_mode="atm")
    assert got.credit is not None
    assert abs(got.credit - credit) < 1e-6
