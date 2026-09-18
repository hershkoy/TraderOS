"""Unit tests for QS five-swing fill realism helpers."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from utils.research.qs_five_swing_realism import (
    backtest_next_open,
    backtest_oracle,
    developing_asof,
    rth_intraday,
)


def test_rth_filter_and_developing():
    # 2020-01-06 Monday RTH 5m stubs
    rows = []
    day = pd.Timestamp("2020-01-06 14:30", tz="UTC")  # 09:30 ET
    px = 100.0
    for i in range(78):  # through 15:55
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
    # add premarket to ensure filter
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
    assert float(dev["close"]) != float(dev1555["close"]) or len(rth) == 1


def test_oracle_and_next_open_smoke():
    day = pd.Timestamp("2020-01-06", tz="UTC")
    rows = []
    px = 100.0
    for i in range(40):
        # skip weekends
        ts = day + timedelta(days=i)
        while ts.weekday() >= 5:
            ts += timedelta(days=1)
            day = ts - timedelta(days=i)
        rows.append(
            {
                "ts": ts,
                "open": px,
                "high": px + 2,
                "low": px - 2,
                "close": px - 1 if i % 7 == 0 else px + 0.5,
                "volume": 1,
            }
        )
        px = rows[-1]["close"]
        day = ts
    # rebuild clean weekday series
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
