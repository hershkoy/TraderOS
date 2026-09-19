"""Unit tests for QS five-swing fill realism helpers."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from utils.research.qs_five_swing_realism import (
    backtest_next_open,
    backtest_oracle,
    backtest_proximity_then_1m,
    developing_asof,
    rth_intraday,
)


def test_rth_filter_and_developing():
    rows = []
    day = pd.Timestamp("2020-01-06 14:30", tz="UTC")
    px = 100.0
    for i in range(78):
        ts = day + timedelta(minutes=5 * i)
        rows.append(
            {
                "ts": ts,
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px + 0.2,
                "volume": 1,
            }
        )
        px += 0.1
    rows.insert(
        0,
        {
            "ts": pd.Timestamp("2020-01-06 13:00", tz="UTC"),
            "open": 99,
            "high": 99,
            "low": 99,
            "close": 99,
            "volume": 1,
        },
    )
    rth = rth_intraday(pd.DataFrame(rows))
    assert (rth["ts"] >= pd.Timestamp("2020-01-06 14:30", tz="UTC")).all()
    sess = pd.Timestamp("2020-01-06", tz="UTC")
    dev = developing_asof(rth, sess, 15, 45)
    assert dev is not None
    dev1555 = developing_asof(rth, sess, 15, 55)
    assert dev1555 is not None
    assert float(dev1555["close"]) == float(rth.iloc[-1]["close"])


def test_1m_developing_1557_vs_1558():
    base = pd.Timestamp("2020-01-06 09:30", tz="America/New_York")
    rows = []
    px = 100.0
    for i in range(390):
        ts = (base + timedelta(minutes=i)).tz_convert("UTC")
        rows.append(
            {
                "ts": ts,
                "open": px,
                "high": px + 0.5,
                "low": px - 0.5,
                "close": px + (0.05 if i < 387 else -0.2),
                "volume": 1,
            }
        )
        px = float(rows[-1]["close"])
    rth = rth_intraday(pd.DataFrame(rows))
    sess = pd.Timestamp("2020-01-06", tz="UTC")
    d57 = developing_asof(rth, sess, 15, 57)
    d58 = developing_asof(rth, sess, 15, 58)
    assert d57 is not None and d58 is not None
    assert float(d58["close"]) != float(d57["close"])


def test_proximity_then_1m_book_label():
    day = pd.Timestamp("2020-01-06", tz="UTC")
    daily_rows = []
    px = 100.0
    for _ in range(40):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        daily_rows.append(
            {
                "ts": day,
                "open": px,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1,
            }
        )
        day += timedelta(days=1)
    daily = pd.DataFrame(daily_rows)
    sess_date = pd.Timestamp(daily.iloc[-1]["ts"]).date()
    base = pd.Timestamp(str(sess_date) + " 09:30", tz="America/New_York")
    rows15 = []
    for mi in range(0, 391, 15):
        ts = (base + timedelta(minutes=mi)).tz_convert("UTC")
        rows15.append(
            {"ts": ts, "open": 100, "high": 102, "low": 98, "close": 100, "volume": 1}
        )
    rows1 = []
    for mi in range(390):
        ts = (base + timedelta(minutes=mi)).tz_convert("UTC")
        rows1.append(
            {"ts": ts, "open": 100, "high": 102, "low": 98, "close": 100, "volume": 1}
        )
    out = backtest_proximity_then_1m(
        daily,
        rth_intraday(pd.DataFrame(rows15)),
        rth_intraday(pd.DataFrame(rows1)),
        decision_hour=15,
        decision_minute=57,
    )
    assert out["book"] == "prox15m_1m_1557"
    assert "agreement" in out


def test_oracle_and_next_open_smoke():
    day = pd.Timestamp("2019-11-04", tz="UTC")
    rows = []
    px = 100.0
    for _ in range(50):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        rows.append(
            {
                "ts": day,
                "open": px,
                "high": px + 2,
                "low": px - 3,
                "close": px - 0.5,
                "volume": 1,
            }
        )
        px = px - 0.2
        day += timedelta(days=1)
    df = pd.DataFrame(rows)
    o = backtest_oracle(df, friction_pct=0.0)
    n = backtest_next_open(df, friction_pct=0.0)
    assert o["n"] >= 0
    assert n["n"] >= 0
    for t in o.get("trades") or []:
        assert int(t["hold_days"]) >= 1
