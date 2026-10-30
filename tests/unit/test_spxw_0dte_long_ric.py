"""Unit tests for the SPXW 0DTE long RIC (no Theta)."""
from __future__ import annotations

from datetime import date, time

import pandas as pd

from utils.research.spxw_0dte_long_ric import (
    CLAIMED,
    FILL_MID,
    FILL_NATURAL,
    claimed_one_lot_pnl,
    find_vix_trigger_eod_low,
    find_vix_trigger_hourly,
    select_long_ric,
    settle_long_ric,
    simulate_day,
)


def _chain(spot: float = 5000.0):
    rows = []
    put_quotes = {
        5000.0: (8.0, 8.4),
        4970.0: (2.0, 2.2),
        4990.0: (6.5, 6.9),
        4960.0: (1.5, 1.7),
        4980.0: (4.0, 4.3),
        4950.0: (0.8, 1.0),
    }
    call_quotes = {
        5000.0: (8.0, 8.4),
        5030.0: (2.0, 2.2),
        5010.0: (5.0, 5.3),
        5040.0: (1.0, 1.2),
        5020.0: (3.8, 4.1),
        5050.0: (0.9, 1.1),
    }
    for strike, (bid, ask) in put_quotes.items():
        rows.append(
            {
                "timestamp": "2024-06-04T10:30:00.000",
                "strike": strike,
                "right": "PUT",
                "bid": bid,
                "ask": ask,
                "delta": -0.3,
                "underlying_price": spot,
                "implied_vol": 0.12,
                "iv_error": 0.0,
            }
        )
    for strike, (bid, ask) in call_quotes.items():
        rows.append(
            {
                "timestamp": "2024-06-04T10:30:00.000",
                "strike": strike,
                "right": "CALL",
                "bid": bid,
                "ask": ask,
                "delta": 0.3,
                "underlying_price": spot,
                "implied_vol": 0.12,
                "iv_error": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_settle_both_wings_max():
    out = settle_long_ric(
        settle=4900.0,
        put_long=5000.0,
        put_short=4970.0,
        call_long=5010.0,
        call_short=5040.0,
        debit=9.0,
    )
    assert out["put_value"] == 30.0
    assert out["call_value"] == 0.0
    assert abs(out["pnl"] - (30.0 - 9.0) * 100) < 1e-6


def test_settle_call_side_pays():
    out = settle_long_ric(
        settle=5100.0,
        put_long=5000.0,
        put_short=4970.0,
        call_long=5010.0,
        call_short=5040.0,
        debit=9.0,
    )
    assert out["put_value"] == 0.0
    assert out["call_value"] == 30.0


def test_select_near_target_debits():
    picked, reason = select_long_ric(_chain(), clock=time(10, 30))
    assert reason == "ok"
    assert picked is not None
    assert abs(picked["put"]["long_strike"] - picked["put"]["short_strike"]) == 30
    assert abs(picked["call"]["short_strike"] - picked["call"]["long_strike"]) == 30
    assert picked["debit_mid"] > 0
    assert picked["debit_natural"] >= picked["debit_mid"] - 1e-9


def test_simulate_mid_vs_natural():
    mid, r1 = simulate_day(_chain(), settle=5100.0, fill=FILL_MID, clock=time(10, 30))
    nat, r2 = simulate_day(_chain(), settle=5100.0, fill=FILL_NATURAL, clock=time(10, 30))
    assert r1 == "ok" and r2 == "ok"
    assert mid is not None and nat is not None
    assert nat["debit"] >= mid["debit"]
    assert mid["won"] is True


def test_vix_hourly_trigger():
    hourly = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-06-04 09:30", tz="America/New_York"),
                "date": date(2024, 6, 4),
                "open": 14.0,
                "high": 14.1,
                "low": 13.95,
                "close": 14.0,
            },
            {
                "timestamp": pd.Timestamp("2024-06-04 10:00", tz="America/New_York"),
                "date": date(2024, 6, 4),
                "open": 13.9,
                "high": 13.95,
                "low": 13.85,
                "close": 13.88,
            },
        ]
    )
    entry, reason = find_vix_trigger_hourly(
        date(2024, 6, 4),
        prior_close=14.5,
        session_open=14.0,
        hourly=hourly,
        min_drop=0.005,
    )
    assert reason == "ok"
    assert entry == time(10, 0)


def test_vix_no_gap():
    entry, reason = find_vix_trigger_eod_low(
        prior_close=14.0,
        session_open=14.2,
        session_low=13.5,
    )
    assert entry is None and reason == "no_gap"


def test_claimed_positive():
    assert claimed_one_lot_pnl() > 0
    assert CLAIMED["n"] == 338
