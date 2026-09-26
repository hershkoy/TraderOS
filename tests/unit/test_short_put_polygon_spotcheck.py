"""Unit tests for Polygon short-put spot-check helpers."""
from __future__ import annotations

from datetime import date

import pandas as pd

from utils.research.short_put_polygon_spotcheck import (
    bars_from_aggs_response,
    pick_entry_exit_bars,
    polygon_option_ticker,
    select_hard_stop_sample,
)


def test_polygon_option_ticker():
    assert polygon_option_ticker("AVGO", "2025-04-11", 210.0, "P") == "O:AVGO250411P00210000"
    assert polygon_option_ticker("aapl", "2024-12-20", 150.5, "p") == "O:AAPL241220P00150500"


def test_bars_and_pick():
    # ms timestamps for 2025-02-26 and 2025-02-27 UTC
    resp = {
        "results": [
            {"t": 1740528000000, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 3},
            {"t": 1740614400000, "o": 12, "h": 13, "l": 11, "c": 12.5, "v": 5},
        ]
    }
    bars = bars_from_aggs_response(resp)
    assert len(bars) == 2
    entry, exit_bar, status = pick_entry_exit_bars(
        bars, date(2025, 2, 26), date(2025, 2, 27)
    )
    assert status == "ok"
    assert entry["close"] == 10.5
    assert exit_bar["close"] == 12.5


def test_same_day_thin_tape_uses_open_as_entry():
    resp = {
        "results": [
            {"t": 1740614400000, "o": 18.0, "h": 20.0, "l": 17.0, "c": 19.26, "v": 4},
        ]
    }
    bars = bars_from_aggs_response(resp)
    entry, exit_bar, status = pick_entry_exit_bars(
        bars, date(2025, 2, 26), date(2025, 2, 27)
    )
    assert status == "ok"
    assert entry["close"] == 18.0
    assert exit_bar["close"] == 19.26


def test_closest_listed_expiry():
    from utils.research.short_put_polygon_spotcheck import closest_listed_expiry

    got = closest_listed_expiry(
        ["2025-01-17", "2025-01-24", "2025-02-21"], date(2025, 1, 20)
    )
    assert got == "2025-01-17"


def test_select_hard_stop_sample():
    df = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": "2025-01-10",
                "put_exit_date": "2025-01-12",
                "put_exit_reason": "hard_stop",
                "put_strike": 50,
                "put_credit": 2.0,
                "put_exit_premium": 3.0,
                "adv_20": 5_000_000,
            },
            {
                "stock": "BBB",
                "buy_date": "2025-02-10",
                "put_exit_date": "2025-02-12",
                "put_exit_reason": "hard_stop",
                "put_strike": 40,
                "put_credit": 1.5,
                "put_exit_premium": 2.5,
                "adv_20": 2_000_000,
            },
            {
                "stock": "CCC",
                "buy_date": "2023-01-10",
                "put_exit_date": "2023-01-12",
                "put_exit_reason": "hard_stop",
                "put_strike": 10,
                "put_credit": 1.0,
                "put_exit_premium": 2.0,
                "adv_20": 9_000_000,
            },
            {
                "stock": "DDD",
                "buy_date": "2025-03-10",
                "put_exit_date": "2025-03-12",
                "put_exit_reason": "tp50",
                "put_strike": 20,
                "put_credit": 1.0,
                "put_exit_premium": 0.5,
                "adv_20": 9_000_000,
            },
        ]
    )
    got = select_hard_stop_sample(df, n=5, min_buy="2024-10-01", min_adv=1_000_000)
    assert len(got) == 2
    assert set(got["stock"]) == {"AAA", "BBB"}
