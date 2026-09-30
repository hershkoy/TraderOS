"""Unit tests for post-earnings PCS live-realism helpers."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from utils.research.post_earnings_pcs_live import (
    align_leg_bars,
    book_stats_from_live,
    credit_vertical_mids,
    debit_vertical_mids,
    resolve_vertical_tickers,
    select_recent_pcs_sample,
    walk_vertical_tp_sl,
)


def test_credit_debit_haircut_widens_roundtrip():
    mid_s, mid_l = 1.20, 0.40
    mid_c = credit_vertical_mids(mid_s, mid_l, haircut=False)
    hs_c = credit_vertical_mids(mid_s, mid_l, haircut=True)
    assert mid_c == pytest.approx(0.80)
    assert hs_c < mid_c
    mid_d = debit_vertical_mids(mid_s, mid_l, haircut=False)
    hs_d = debit_vertical_mids(mid_s, mid_l, haircut=True)
    assert mid_d == pytest.approx(0.80)
    assert hs_d > mid_d


def test_align_leg_bars_inner_join():
    s = pd.DataFrame(
        {
            "date": [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1.0, 0.9, 0.5],
            "volume": [1, 1, 1],
        }
    )
    l = pd.DataFrame(
        {
            "date": [date(2025, 1, 3), date(2025, 1, 6)],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [0.3, 0.2],
            "volume": [1, 1],
        }
    )
    m = align_leg_bars(s, l)
    assert len(m) == 2
    assert list(m["short_close"]) == [0.9, 0.5]


def test_walk_hits_tp_on_crush():
    rows = []
    d0 = date(2025, 1, 2)
    for i, (sc, lc) in enumerate([(1.5, 0.4), (0.6, 0.25), (0.35, 0.15)]):
        rows.append(
            {
                "date": d0 + timedelta(days=i),
                "short_close": sc,
                "long_close": lc,
            }
        )
    aligned = pd.DataFrame(rows)
    res = walk_vertical_tp_sl(
        aligned,
        entry_date=d0,
        width=5.0,
        expiry=d0 + timedelta(days=40),
        spread_floor=0.05,
        spread_frac=0.08,
    )
    assert res.status == "ok"
    assert res.exit_reason == "tp50"
    assert res.ror_hs is not None and res.ror_hs > 0


def test_resolve_vertical_tickers():
    st, lt, exp, sk, lk = resolve_vertical_tickers(
        "AAPL", date(2025, 1, 15), 220.0, 215.0, 37
    )
    assert st.startswith("O:AAPL")
    assert lt.startswith("O:AAPL")
    assert "P" in st and "P" in lt
    assert sk > lk
    assert exp


def test_select_recent_pcs_sample():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "NVDA"],
            "entry_date": ["2024-11-01", "2025-01-01", "2025-02-01", "2023-01-01"],
            "short_k": [200, 210, 400, 100],
            "long_k": [195, 205, 395, 95],
            "width": [5, 5, 5, 5],
            "dte": [37, 37, 37, 37],
            "credit": [1, 1, 1, 1],
            "ror_pct": [1, 1, 1, 1],
            "hold_days": [5, 5, 5, 5],
            "exit_reason": ["tp50"] * 4,
            "exit_date": ["2024-11-10", "2025-01-10", "2025-02-10", "2023-01-10"],
        }
    )
    s = select_recent_pcs_sample(df, n=10, min_entry="2024-10-01")
    assert len(s) == 3
    assert set(s["symbol"]) <= {"AAPL", "MSFT"}


def test_book_stats_from_live():
    rows = [
        {"ror_hs": 9.0, "hold_days": 8, "exit_reason": "tp50"},
        {"ror_hs": -19.0, "hold_days": 5, "exit_reason": "sl_2x"},
    ]
    st = book_stats_from_live(rows)
    assert st["n"] == 2
    assert st["win_rate_pct"] == pytest.approx(50.0)
