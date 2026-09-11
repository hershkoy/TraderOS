"""Unit tests for last-15m realistic sell overlay."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts.research.compare_1d_last_15m_realistic_sells import apply_realistic_sells
from utils.research.realistic_purchaser import bar_mid

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c}


def test_apply_realistic_sells_wve_like_both_modes():
    entry = _bar(_et(2023, 12, 6, 15, 45), 6.80, 7.00, 6.70, 6.90)
    d7_0930 = _bar(_et(2023, 12, 7, 9, 30), 5.50, 5.80, 5.20, 5.40)
    d7_0945 = _bar(_et(2023, 12, 7, 9, 45), 5.42, 5.60, 5.30, 5.50)
    d7_1545 = _bar(_et(2023, 12, 7, 15, 45), 5.45, 5.50, 5.30, 5.35)
    d8_0930 = _bar(_et(2023, 12, 8, 9, 30), 5.10, 5.30, 4.90, 5.00)
    indexed = {
        "WVE": {
            entry["ts"].date(): [entry],
            d7_0930["ts"].date(): [d7_0930, d7_0945, d7_1545],
            d8_0930["ts"].date(): [d8_0930],
        }
    }
    trades = pd.DataFrame(
        [
            {
                "stock": "WVE",
                "buy_date": "2023-12-06",
                "buy_time": "2023-12-06 20:45",
                "buy_price": 6.85,
                "sell_date": "2023-12-07",
                "sell_price": 6.0254,
                "gain_pct": -12.038,
                "exit_reason": "hard_stop",
                "atr_pct": 4.694,
                "buy_price_before": 6.41,
            }
        ]
    )
    intra = apply_realistic_sells(trades, indexed, mode="15m-next-mid")
    assert intra.iloc[0]["status"] == "ok"
    assert float(intra.iloc[0]["sell_price_after"]) == pytest.approx(bar_mid(5.60, 5.30))
    assert "09:45" in str(intra.iloc[0]["sell_time_after"])
    assert float(intra.iloc[0]["gain_after"]) < float(intra.iloc[0]["gain_before"])

    eod = apply_realistic_sells(trades, indexed, mode="daily-close-next-open-mid")
    assert eod.iloc[0]["status"] == "ok"
    assert float(eod.iloc[0]["sell_price_after"]) == pytest.approx(bar_mid(5.30, 4.90))
    assert str(eod.iloc[0]["sell_datetime_after"]) == "2023-12-08"
    assert float(eod.iloc[0]["sell_price_after"]) != pytest.approx(6.0254)
