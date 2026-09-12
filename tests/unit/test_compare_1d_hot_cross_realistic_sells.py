"""Hot-cross realistic sells: same-session stop after a mid-RTH lerp85 fill."""
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


def test_hot_cross_15m_next_mid_can_stop_same_session():
    """Buy 10:30 ET. 10:45 tags the 6% stop. Fill 11:00 mid — not next-day daily clip."""
    entry = _bar(_et(2019, 5, 2, 10, 30), 33.40, 33.50, 33.30, 33.45)
    tag = _bar(_et(2019, 5, 2, 10, 45), 33.20, 33.30, 31.00, 31.20)
    nxt = _bar(_et(2019, 5, 2, 11, 0), 31.10, 31.40, 30.90, 31.20)
    d3 = _bar(_et(2019, 5, 3, 9, 30), 31.40, 31.50, 31.20, 31.30)
    d4 = _bar(_et(2019, 5, 4, 9, 30), 31.10, 31.20, 30.80, 31.00)
    indexed = {
        "HCC": {
            entry["ts"].date(): [entry, tag, nxt],
            d3["ts"].date(): [d3],
            d4["ts"].date(): [d4],
        }
    }
    trades = pd.DataFrame(
        [
            {
                "stock": "HCC",
                "buy_date": "2019-05-02",
                "buy_time": "2019-05-02 14:30",
                "buy_price": 33.4326,
                "sell_date": "2019-05-03",
                "sell_price": 31.4267,
                "gain_pct": -6.00,
                "exit_reason": "hard_stop",
                "atr_pct": 3.461,
            }
        ]
    )
    intra = apply_realistic_sells(trades, indexed, mode="15m-next-mid")
    assert intra.iloc[0]["status"] == "ok"
    assert float(intra.iloc[0]["sell_price_after"]) == pytest.approx(bar_mid(31.40, 30.90))
    assert "11:00" in str(intra.iloc[0]["sell_time_after"])
    assert str(intra.iloc[0]["sell_datetime_after"]) == "2019-05-02"
    assert float(intra.iloc[0]["gain_after"]) < float(intra.iloc[0]["gain_before"])

    eod = apply_realistic_sells(trades, indexed, mode="daily-close-next-open-mid")
    assert eod.iloc[0]["status"] == "ok"
    assert str(eod.iloc[0]["sell_datetime_after"]) == "2019-05-04"
    assert float(eod.iloc[0]["sell_price_after"]) == pytest.approx(bar_mid(31.20, 30.80))
    assert float(eod.iloc[0]["sell_price_after"]) != pytest.approx(31.4267)
