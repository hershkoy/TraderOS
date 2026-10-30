"""Unit tests for the SPXW 0DTE short iron fly (no Theta)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from utils.research.spxw_0dte_iron_fly import (
    CLAIMED,
    FILL_MID,
    FILL_NATURAL,
    claimed_one_lot_pnl,
    eligible_days,
    select_iron_fly,
    settle_short_iron_fly,
    simulate_day,
    summarize_pnls,
)


def _chain(spot: float = 5000.0):
    """Synthetic 13:45 chain around spot with roughly target deltas."""
    rows = []
    # puts: 50-delta near spot, wings cheaper
    put_specs = [
        (4985.0, 1.0, 1.2, -0.15),
        (5000.0, 7.0, 7.4, -0.50),
        (5010.0, 10.0, 10.5, -0.65),
    ]
    call_specs = [
        (4990.0, 10.0, 10.5, 0.60),
        (5005.0, 5.5, 5.9, 0.40),
        (5020.0, 1.0, 1.2, 0.15),
    ]
    for strike, bid, ask, delta in put_specs:
        rows.append(
            {
                "timestamp": "2024-06-03T13:45:00.000",
                "strike": strike,
                "right": "PUT",
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "underlying_price": spot,
                "implied_vol": 0.12,
                "iv_error": 0.0,
            }
        )
    for strike, bid, ask, delta in call_specs:
        rows.append(
            {
                "timestamp": "2024-06-03T13:45:00.000",
                "strike": strike,
                "right": "CALL",
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "underlying_price": spot,
                "implied_vol": 0.12,
                "iv_error": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_settle_inside_body_keeps_credit():
    out = settle_short_iron_fly(
        settle=5002.0,
        short_put=5000.0,
        short_call=5005.0,
        wing=15.0,
        credit=7.5,
    )
    assert out["put_value"] == 0.0
    assert out["call_value"] == 0.0
    assert abs(out["pnl"] - 750.0) < 1e-6


def test_settle_blows_through_put_wing():
    out = settle_short_iron_fly(
        settle=4980.0,
        short_put=5000.0,
        short_call=5005.0,
        wing=15.0,
        credit=7.5,
    )
    # put spread maxes at 15; call worthless
    assert out["put_value"] == 15.0
    assert out["call_value"] == 0.0
    assert abs(out["pnl"] - (7.5 - 15.0) * 100) < 1e-6


def test_select_50d_put_40d_call_and_wings():
    picked, reason = select_iron_fly(_chain(), wing=15.0)
    assert reason == "ok"
    assert picked is not None
    assert picked["short_put"]["strike"] == 5000.0
    assert picked["short_call"]["strike"] == 5005.0
    assert picked["long_put"]["strike"] == 4985.0
    assert picked["long_call"]["strike"] == 5020.0
    assert picked["credit_mid"] > 0


def test_simulate_mid_vs_natural_credit():
    chain = _chain()
    mid, r1 = simulate_day(chain, settle=5002.0, fill=FILL_MID)
    nat, r2 = simulate_day(chain, settle=5002.0, fill=FILL_NATURAL)
    assert r1 == "ok" and r2 == "ok"
    assert mid is not None and nat is not None
    assert nat["credit"] <= mid["credit"]
    assert mid["won"] is True


def test_eligible_days_skip_wed_and_high_vix():
    exps = [
        date(2024, 6, 3),  # Mon
        date(2024, 6, 4),  # Tue
        date(2024, 6, 5),  # Wed
        date(2024, 6, 6),  # Thu
        date(2024, 6, 7),  # Fri
    ]
    vix = {
        date(2024, 6, 3): 15.0,
        date(2024, 6, 4): 21.0,  # too high
        date(2024, 6, 5): 14.0,
        date(2024, 6, 6): 16.0,
        date(2024, 6, 7): 18.0,
    }
    days = eligible_days(
        exps,
        start=date(2024, 6, 1),
        end=date(2024, 6, 10),
        vix_by_day=vix,
        vix_max=20.0,
    )
    assert days == [date(2024, 6, 3), date(2024, 6, 6), date(2024, 6, 7)]


def test_claimed_one_lot_approx_positive():
    assert claimed_one_lot_pnl() > 0
    assert CLAIMED["n"] == 299


def test_summarize_pnls_wr():
    s = summarize_pnls([100.0, -80.0, 50.0])
    assert s["n"] == 3
    assert abs(s["win_rate"] - 2 / 3) < 1e-9
    assert s["pf"] is not None and s["pf"] > 1.0


def test_load_index_eod_falls_back_when_theta_starts_late(monkeypatch):
    """Partial Theta (2024+) must not short-circuit a 2022 Yahoo window."""
    from utils.research import spxw_0dte_iron_fly as mod

    late = pd.DataFrame(
        {
            "created": ["2024-01-02", "2024-01-03"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
        }
    )

    def fake_theta(_client, _path, _params, timeout=60.0):  # noqa: ARG001
        return late

    yahoo = pd.DataFrame(
        {
            "date": [date(2022, 5, 11), date(2022, 5, 12), date(2024, 1, 2)],
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [4000.0, 4010.0, 4700.0],
        }
    )

    monkeypatch.setattr(mod, "_theta_csv", fake_theta)
    monkeypatch.setattr(mod, "_yahoo_index_eod", lambda *_a, **_k: yahoo)

    out = mod.load_index_eod(
        object(), "SPX", date(2022, 5, 11), date(2024, 1, 3)
    )
    assert out["date"].iloc[0] == date(2022, 5, 11)
    assert len(out) == 3
