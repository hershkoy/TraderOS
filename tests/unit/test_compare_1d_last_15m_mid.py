"""Unit tests for last-RTH 15m open-above-rail mid compare."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts.research.compare_1d_last_15m_mid import compare_before_rows
from utils.research.realistic_purchaser import bar_mid

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_compare_last_15m_mid_requires_open_above_rail():
    last = {
        "ts": _et(2019, 6, 7, 15, 45),
        "open": 17.80,
        "high": 17.90,
        "low": 17.70,
        "close": 17.85,
    }
    indexed = {
        "TFSL": {
            date(2019, 6, 7): [
                {
                    "ts": _et(2019, 6, 7, 9, 30),
                    "open": 17.50,
                    "high": 17.60,
                    "low": 17.40,
                    "close": 17.55,
                },
                last,
            ]
        }
    }
    before = pd.DataFrame(
        [
            {
                "stock": "TFSL",
                "buy_date": "2019-06-07",
                "buy_price": 17.69,
                "sell_date": "2020-03-09",
                "sell_price": 20.223,
                "gain_pct": 14.32,
                "touch_price": 17.68,
                "atr_pct": 1.384,
            }
        ]
    )
    out = compare_before_rows(before, indexed)
    assert out.iloc[0]["row_kind"] == "matched"
    assert float(out.iloc[0]["buy_price_after"]) == pytest.approx(bar_mid(17.90, 17.70), rel=1e-6)
    assert str(out.iloc[0]["buy_datetime_after"]).endswith("15:45")
    assert float(out.iloc[0]["last_15m_open"]) == 17.80
    assert float(out.iloc[0]["open_above_rail_pct"]) > 0

    below = before.copy()
    below.loc[0, "touch_price"] = 18.00
    skip = compare_before_rows(below, indexed)
    assert skip.iloc[0]["row_kind"] == "before_skipped"
    assert skip.iloc[0]["skip_reason"] == "no_open_cross"
    assert float(skip.iloc[0]["last_15m_open"]) == 17.80
