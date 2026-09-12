"""Last-15m RTH open-to-open exhaustion helpers."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.research.last_15m_exhaustion import (
    attach_exhaustion,
    keep_unless_run_atr,
    session_rth_extension,
)

ET = ZoneInfo("America/New_York")


def _bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c}


def test_session_run_is_1545_open_minus_0930_open():
    d = datetime(2022, 6, 15, tzinfo=ET)
    bars = [
        _bar(d.replace(hour=9, minute=30), 10.0, 10.2, 9.9, 10.1),
        _bar(d.replace(hour=12, minute=0), 10.4, 10.5, 10.3, 10.4),
        _bar(d.replace(hour=15, minute=30), 11.0, 11.1, 10.9, 11.0),
        _bar(d.replace(hour=15, minute=45), 12.0, 12.2, 11.8, 12.1),
    ]
    rec = session_rth_extension(bars, d.date())
    assert rec["rth_open"] == 10.0
    assert rec["last15_open"] == 12.0
    assert rec["run"] == 2.0
    assert rec["atr_15m"] is None


def test_15m_atr_excludes_fill_bar():
    d = datetime(2022, 6, 15, tzinfo=ET)
    bars = []
    px = 10.0
    for hh, mm in [(9, 30), (9, 45), (10, 0), (10, 15), (10, 30), (10, 45), (11, 0), (15, 45)]:
        bars.append(_bar(d.replace(hour=hh, minute=mm), px, px + 0.1, px - 0.1, px + 0.05))
        px += 0.2
    rec = session_rth_extension(bars, d.date())
    huge = dict(bars[-1])
    huge["high"] = 50.0
    huge["low"] = 1.0
    rec2 = session_rth_extension(list(bars[:-1]) + [huge], d.date())
    assert rec["atr_15m"] is not None and rec2["atr_15m"] is not None
    assert abs(rec["atr_15m"] - rec2["atr_15m"]) < 1e-12


def test_keep_unless_run_atr_skips_extended():
    trades = pd.DataFrame({"run_atr_15m": [0.5, 2.5, np.nan]})
    keep = keep_unless_run_atr(trades, col="run_atr_15m", max_run_atr=2.0)
    assert list(keep.astype(int)) == [1, 0, 1]


def test_attach_exhaustion_fill_day_only():
    fill = datetime(2022, 6, 15, 15, 45, tzinfo=ET)
    d = fill.date()
    bars = [
        _bar(fill.replace(hour=9, minute=30), 20.0, 20.2, 19.9, 20.1),
        _bar(fill.replace(hour=15, minute=30), 20.4, 20.5, 20.3, 20.4),
        _bar(fill, 21.0, 21.1, 20.9, 21.0),
    ]
    trades = pd.DataFrame(
        [{"stock": "AAA", "buy_date": "2022-06-15", "buy_time": fill, "gain_pct": 1.0}]
    )
    out = attach_exhaustion(trades, {"AAA": {d: bars}})
    assert abs(float(out.loc[0, "run"]) - 1.0) < 1e-12
