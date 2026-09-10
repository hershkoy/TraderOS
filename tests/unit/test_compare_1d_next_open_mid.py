"""Unit tests for next-session 09:30 15m-mid compare helper."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.research.compare_1d_next_open_mid import compare_before_rows
from utils.research.realistic_purchaser import bar_mid

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_compare_before_rows_uses_next_session_open_mid():
    open_bar = {
        "ts": _et(2025, 6, 16, 9, 30),
        "open": 27.20,
        "high": 27.50,
        "low": 27.00,
        "close": 27.36,
    }
    indexed = {
        "RDWR": {
            date(2025, 6, 13): [
                {
                    "ts": _et(2025, 6, 13, 15, 45),
                    "open": 26.4,
                    "high": 26.7,
                    "low": 26.2,
                    "close": 26.58,
                }
            ],
            date(2025, 6, 16): [open_bar],
        }
    }
    before = pd.DataFrame(
        [
            {
                "stock": "RDWR",
                "buy_date": "2025-06-13",
                "buy_datetime_before": "2025-06-13 10:30",
                "buy_price_before": 24.64,
                "sell_datetime": "2025-07-11",
                "sell_price": 28.251,
                "gain_before": 14.655,
            }
        ]
    )
    out = compare_before_rows(before, indexed)
    assert len(out) == 1
    assert out.iloc[0]["row_kind"] == "matched"
    assert float(out.iloc[0]["buy_price_after"]) == bar_mid(27.50, 27.00)
    assert str(out.iloc[0]["session_date"]) == "2025-06-16"
    assert str(out.iloc[0]["buy_datetime_after"]).endswith("09:30")
