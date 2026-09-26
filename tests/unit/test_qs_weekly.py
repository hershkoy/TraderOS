"""Unit tests for utils.research.qs_weekly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.research.qs_weekly import (
    backtest_rotation69,
    backtest_spy_down_week,
    to_weekly_friday,
)


def _synth_daily(n: int = 160, seed: int = 1, start: str = "2018-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = []
    px = 100.0
    for i in range(n):
        if (i // 5) % 2 == 0:
            px *= 0.992
        else:
            px *= 1.008
        px *= float(1.0 + 0.001 * rng.normal())
        closes.append(px)
    opens = [closes[0]] + closes[:-1]
    ts = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": [c * 1.002 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        }
    )


def test_to_weekly_friday_ends_friday():
    df = _synth_daily()
    w = to_weekly_friday(df)
    assert len(w) >= 10
    # Pandas W-FRI labels are Fridays.
    assert all(pd.Timestamp(t).dayofweek == 4 for t in w["ts"])


def test_down_week_claim_produces_trades():
    df = _synth_daily()
    r = backtest_spy_down_week(df, fill_mode="claim_close")
    assert r["n"] >= 1
    t0 = r["trades"][0]
    assert t0["entry_px"] > 0
    assert t0["exit_px"] > 0
    assert "gate" in r


def test_down_week_honest_uses_next_open():
    df = _synth_daily()
    r = backtest_spy_down_week(df, fill_mode="next_open")
    assert r["n"] >= 1
    t0 = r["trades"][0]
    # Entry date should be after signal Friday (next business day).
    assert t0["entry_date"] >= t0["signal_date"]


def test_down_week_friction_reduces_avg():
    df = _synth_daily()
    a = backtest_spy_down_week(df, fill_mode="claim_close", friction_pct=0.0)
    b = backtest_spy_down_week(df, fill_mode="claim_close", friction_pct=0.25)
    assert a["n"] == b["n"]
    assert a["n"] >= 1
    assert b["avg_gain_pct"] < a["avg_gain_pct"]


def test_rotation69_lowest_and_highest():
    frames = {
        "SPY": _synth_daily(seed=1),
        "EFA": _synth_daily(seed=2),
        "GLD": _synth_daily(seed=3),
        "TLT": _synth_daily(seed=4),
    }
    lo = backtest_rotation69(frames, lookback=2, pick="lowest", cadence="weekly")
    hi = backtest_rotation69(frames, lookback=2, pick="highest", cadence="weekly")
    assert lo["n"] >= 5
    assert hi["n"] >= 5
    assert lo["pick"] == "lowest"
    assert hi["pick"] == "highest"
    assert "gate" in lo
