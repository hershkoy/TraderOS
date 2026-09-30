"""Unit tests for PCS open/delta spot-check helpers."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from utils.research.post_earnings_pcs_open_delta import (
    first_bar_on,
    imply_put_iv,
    pick_delta_short,
    put_delta_from_mid,
    strike_ladder,
    vertical_credits_from_legs,
)
from utils.options.greeks import black_scholes_put


def test_imply_put_iv_roundtrip():
    S, K, T, r, sig = 100.0, 95.0, 37 / 365.25, 0.05, 0.35
    mid = black_scholes_put(S, K, T, r, sig)
    iv = imply_put_iv(mid, S, K, T, r)
    assert iv is not None
    assert iv == pytest.approx(sig, rel=0.05)


def test_put_delta_from_mid_negative():
    S, K, T, r, sig = 100.0, 90.0, 37 / 365.25, 0.05, 0.40
    mid = black_scholes_put(S, K, T, r, sig)
    d, iv = put_delta_from_mid(mid, S, K, T, r)
    assert d is not None and iv is not None
    assert -1.0 < d < 0.0


def test_strike_ladder_otm():
    ks = strike_ladder(200.0, n=8)
    assert len(ks) <= 8
    assert all(k < 200 for k in ks)


def test_pick_delta_short():
    cands = [
        {"strike": 180, "delta": -0.10},
        {"strike": 185, "delta": -0.19},
        {"strike": 190, "delta": -0.35},
    ]
    p = pick_delta_short(cands, target_delta=-0.20)
    assert p["strike"] == 185


def test_vertical_open_close():
    c = vertical_credits_from_legs(1.5, 1.0, 0.4, 0.3)
    assert c["credit_open"] == pytest.approx(1.1)
    assert c["credit_close"] == pytest.approx(0.7)


def test_first_bar_on():
    bars = pd.DataFrame(
        {
            "date": [date(2025, 1, 2), date(2025, 1, 3)],
            "open": [1.0, 2.0],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1.1, 2.1],
            "volume": [1, 1],
        }
    )
    b = first_bar_on(bars, date(2025, 1, 3))
    assert b is not None
    assert b.open == 2.0
