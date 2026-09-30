"""Unit tests for PCS minute open/first-hour helpers."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from utils.research.post_earnings_pcs_minute import (
    first_hour_vwap,
    first_rth_mark,
    mark_at_or_after,
    minute_bars_from_aggs,
    summarize_minute_pilot,
    vertical_credit_from_marks,
)

ET = ZoneInfo("America/New_York")


def _ms(y, m, d, hh, mm):
    return int(datetime(y, m, d, hh, mm, tzinfo=ET).timestamp() * 1000)


def test_minute_bars_from_aggs():
    resp = {
        "results": [
            {"t": _ms(2025, 1, 2, 9, 30), "o": 2.0, "h": 2.1, "l": 1.9, "c": 2.05, "v": 10},
            {"t": _ms(2025, 1, 2, 9, 31), "o": 2.1, "h": 2.2, "l": 2.0, "c": 2.1, "v": 5},
        ]
    }
    df = minute_bars_from_aggs(resp)
    assert len(df) == 2
    assert df.iloc[0]["close"] == pytest.approx(2.05)


def test_first_rth_and_10():
    resp = {
        "results": [
            {"t": _ms(2025, 1, 2, 9, 15), "o": 9.0, "h": 9, "l": 9, "c": 9, "v": 1},
            {"t": _ms(2025, 1, 2, 9, 30), "o": 3.0, "h": 3, "l": 3, "c": 3, "v": 2},
            {"t": _ms(2025, 1, 2, 10, 0), "o": 1.5, "h": 1.5, "l": 1.5, "c": 1.4, "v": 4},
        ]
    }
    df = minute_bars_from_aggs(resp)
    day = date(2025, 1, 2)
    rth = first_rth_mark(df, day)
    assert rth is not None and rth.open == pytest.approx(3.0)
    t10 = mark_at_or_after(df, day, datetime(2025, 1, 2, 10, 0).time())
    assert t10 is not None and t10.close == pytest.approx(1.4)


def test_first_hour_vwap():
    resp = {
        "results": [
            {"t": _ms(2025, 1, 2, 9, 30), "o": 2, "h": 2, "l": 2, "c": 2.0, "v": 10},
            {"t": _ms(2025, 1, 2, 10, 0), "o": 1, "h": 1, "l": 1, "c": 1.0, "v": 10},
            {"t": _ms(2025, 1, 2, 11, 0), "o": 0.5, "h": 0.5, "l": 0.5, "c": 0.5, "v": 100},
        ]
    }
    df = minute_bars_from_aggs(resp)
    vwap = first_hour_vwap(df, date(2025, 1, 2))
    assert vwap == pytest.approx(1.5)


def test_vertical_credit():
    assert vertical_credit_from_marks(2.0, 0.5) == pytest.approx(1.5)
    assert vertical_credit_from_marks(None, 0.5) is None


def test_summarize_empty():
    s = summarize_minute_pilot([])
    assert s["attempted"] == 0
    assert s["ok_n"] == 0
