"""Unit tests for weekly PCE detector / books."""
from __future__ import annotations

import pandas as pd

from utils.research.pce_weekly import align_weekly_pair, backtest_pce_weekly, detect_pce_events


def _daily(rows):
    return pd.DataFrame(rows)


def test_align_detects_green_red_pce():
    # Two Fri weeks: week1 both green (no PCE), week2 QQQ green / SPY red (PCE).
    qqq = _daily(
        [
            {"ts": "2020-01-06T21:00:00Z", "open": 100, "high": 102, "low": 99, "close": 101},
            {"ts": "2020-01-07T21:00:00Z", "open": 101, "high": 103, "low": 100, "close": 102},
            {"ts": "2020-01-08T21:00:00Z", "open": 102, "high": 104, "low": 101, "close": 103},
            {"ts": "2020-01-09T21:00:00Z", "open": 103, "high": 105, "low": 102, "close": 104},
            {"ts": "2020-01-10T21:00:00Z", "open": 104, "high": 106, "low": 103, "close": 105},
            # week 2 — green
            {"ts": "2020-01-13T21:00:00Z", "open": 105, "high": 110, "low": 104, "close": 106},
            {"ts": "2020-01-14T21:00:00Z", "open": 106, "high": 111, "low": 105, "close": 107},
            {"ts": "2020-01-15T21:00:00Z", "open": 107, "high": 112, "low": 106, "close": 108},
            {"ts": "2020-01-16T21:00:00Z", "open": 108, "high": 113, "low": 107, "close": 109},
            {"ts": "2020-01-17T21:00:00Z", "open": 105, "high": 114, "low": 104, "close": 110},
            # week 3 exit
            {"ts": "2020-01-20T21:00:00Z", "open": 110, "high": 115, "low": 109, "close": 111},
            {"ts": "2020-01-21T21:00:00Z", "open": 111, "high": 116, "low": 110, "close": 112},
            {"ts": "2020-01-22T21:00:00Z", "open": 112, "high": 117, "low": 111, "close": 113},
            {"ts": "2020-01-23T21:00:00Z", "open": 113, "high": 118, "low": 112, "close": 114},
            {"ts": "2020-01-24T21:00:00Z", "open": 114, "high": 119, "low": 113, "close": 115},
        ]
    )
    spy = _daily(
        [
            {"ts": "2020-01-06T21:00:00Z", "open": 200, "high": 202, "low": 199, "close": 201},
            {"ts": "2020-01-07T21:00:00Z", "open": 201, "high": 203, "low": 200, "close": 202},
            {"ts": "2020-01-08T21:00:00Z", "open": 202, "high": 204, "low": 201, "close": 203},
            {"ts": "2020-01-09T21:00:00Z", "open": 203, "high": 205, "low": 202, "close": 204},
            {"ts": "2020-01-10T21:00:00Z", "open": 204, "high": 206, "low": 203, "close": 205},
            # week 2 — red
            {"ts": "2020-01-13T21:00:00Z", "open": 205, "high": 206, "low": 200, "close": 204},
            {"ts": "2020-01-14T21:00:00Z", "open": 204, "high": 205, "low": 199, "close": 203},
            {"ts": "2020-01-15T21:00:00Z", "open": 203, "high": 204, "low": 198, "close": 202},
            {"ts": "2020-01-16T21:00:00Z", "open": 202, "high": 203, "low": 197, "close": 201},
            {"ts": "2020-01-17T21:00:00Z", "open": 205, "high": 206, "low": 196, "close": 200},
            # week 3
            {"ts": "2020-01-20T21:00:00Z", "open": 200, "high": 205, "low": 199, "close": 201},
            {"ts": "2020-01-21T21:00:00Z", "open": 201, "high": 206, "low": 200, "close": 202},
            {"ts": "2020-01-22T21:00:00Z", "open": 202, "high": 207, "low": 201, "close": 203},
            {"ts": "2020-01-23T21:00:00Z", "open": 203, "high": 208, "low": 202, "close": 204},
            {"ts": "2020-01-24T21:00:00Z", "open": 204, "high": 209, "low": 203, "close": 205},
        ]
    )
    weekly = align_weekly_pair(qqq, spy)
    assert weekly["pce"].sum() >= 1
    events = detect_pce_events(weekly)
    assert len(events) >= 1
    assert str(events.iloc[0]["green_leg"]) == "QQQ"
    assert str(events.iloc[0]["red_leg"]) == "SPY"

    res = backtest_pce_weekly(qqq, spy, book="long_green_next_open")
    assert res["n"] >= 1
    assert res["trades"][0]["symbol"] == "QQQ"
