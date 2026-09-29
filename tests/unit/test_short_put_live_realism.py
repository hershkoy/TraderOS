"""Unit tests for short_put_live_realism helpers."""
from __future__ import annotations

from datetime import date

from utils.research.short_put_live_realism import (
    cash_secured_gain,
    credit_after_spread,
    debit_after_spread,
    half_spread,
    occupancy_fcfs_put,
    pick_nearest_put_contract,
    resolve_listed_put,
    summarize_gains,
)


def test_half_spread_floor_and_frac():
    assert half_spread(0.10) == 0.05  # floor
    assert abs(half_spread(2.0) - 0.16) < 1e-9  # 8% of 2


def test_credit_debit_spread():
    assert abs(credit_after_spread(1.0) - 0.92) < 1e-9
    assert abs(debit_after_spread(1.0) - 1.08) < 1e-9


def test_cash_secured_gain():
    g = cash_secured_gain(1.0, 1.5, 50.0)
    assert g is not None
    assert abs(g - (-1.0)) < 1e-9


def test_pick_nearest_put():
    contracts = [
        {"ticker": "O:AAA250101P00045000", "strike_price": 45},
        {"ticker": "O:AAA250101P00050000", "strike_price": 50},
        {"ticker": "O:AAA250101P00055000", "strike_price": 55},
    ]
    p = pick_nearest_put_contract(contracts, 51.0)
    assert p is not None
    assert float(p["strike_price"]) == 50.0


def test_resolve_listed_put():
    listed = resolve_listed_put(
        underlying="XYZ",
        expiries=["2025-02-14", "2025-02-21", "2025-02-28"],
        contracts=[
            {"ticker": "O:XYZ250221P00100000", "strike_price": 100, "contract_type": "put"},
            {"ticker": "O:XYZ250221P00105000", "strike_price": 105, "contract_type": "put"},
        ],
        buy_date=date(2025, 1, 10),
        target_dte=45,
        target_strike=102.0,
    )
    assert listed is not None
    assert listed.expiry == "2025-02-21"
    assert listed.strike == 100.0
    assert listed.source == "contracts"


def test_resolve_listed_put_rounded_fallback():
    listed = resolve_listed_put(
        underlying="AVGO",
        expiries=["2025-04-11"],
        contracts=[],
        buy_date=date(2025, 2, 26),
        target_dte=45,
        target_strike=201.3,
    )
    assert listed is not None
    assert listed.source == "rounded"
    assert listed.strike == 200.0
    assert "AVGO" in listed.ticker


def test_nearby_fridays():
    from utils.research.short_put_live_realism import nearby_fridays

    # 2025-04-12 is a Saturday; nearest Fridays include 2025-04-11
    fr = nearby_fridays(date(2025, 4, 12), window_days=10)
    assert fr[0] == date(2025, 4, 11)
    assert all(d.weekday() == 4 for d in fr)


def test_round_and_candidates():
    from utils.research.short_put_live_realism import round_option_strike, strike_candidates

    assert round_option_strike(201.3) == 200.0
    c = strike_candidates(201.3, n_each_side=1)
    assert 200.0 in c
    assert 205.0 in c


def test_occupancy_fcfs_put():
    trades = [
        {"symbol": "A", "entry_date": "2025-01-02", "exit_date": "2025-01-10", "gain_pct": 1.0},
        {"symbol": "B", "entry_date": "2025-01-03", "exit_date": "2025-01-12", "gain_pct": -2.0},
        {"symbol": "C", "entry_date": "2025-01-04", "exit_date": "2025-01-20", "gain_pct": 0.5},
        {"symbol": "D", "entry_date": "2025-01-05", "exit_date": "2025-01-08", "gain_pct": 0.2},
    ]
    kept, meta = occupancy_fcfs_put(trades, max_open=2, unique_symbol=True)
    assert meta["n_kept"] == 2
    assert meta["n_skipped"] == 2
    assert len(kept) == 2


def test_summarize_gains():
    s = summarize_gains([1.0, -0.5, 2.0])
    assert s["n"] == 3
    assert s["E"] > 0
    assert s["PF"] > 1.0
