"""Unit tests for utils.research.roc_rebound."""
from __future__ import annotations

import pandas as pd
import pytest

from utils.research.roc_rebound import (
    backtest_roc_rebound,
    roc_series,
    sweep_exit_thresholds,
    sweep_roc_lengths,
)


def _synth_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    # Deterministic path: drop hard then rebound so ROC crosses thresholds.
    closes = []
    px = 100.0
    for i in range(n):
        if 10 <= i <= 16:
            px *= 0.985  # sharp decline -> ROC < -3
        elif 17 <= i <= 28:
            px *= 1.012  # rebound -> ROC > +3
        else:
            px *= 1.001
        closes.append(px)
    opens = [closes[0]] + closes[:-1]
    ts = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
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


def test_roc_series_matches_pct_change():
    s = pd.Series([100.0, 102.0, 101.0, 105.0, 98.0])
    got = roc_series(s, 2)
    expect = (s / s.shift(2) - 1.0) * 100.0
    pd.testing.assert_series_equal(got, expect)


def test_backtest_produces_trade_on_synth():
    df = _synth_df()
    r = backtest_roc_rebound(df, symbol="TEST", roc_length=7, entry_thresh_pct=3.0, exit_thresh_pct=3.0)
    assert r["n"] >= 1
    t0 = r["trades"][0]
    assert t0["entry_px"] > 0
    assert t0["exit_px"] > 0
    assert t0["hold_days"] >= 1
    # Fill is next open after signal close, so entry date != first bar.
    assert "entry_date" in t0 and "exit_date" in t0


def test_entry_uses_next_open_not_signal_close():
    df = _synth_df()
    r = backtest_roc_rebound(df, symbol="TEST")
    assert r["n"] >= 1
    t0 = r["trades"][0]
    # Find signal bar: day before entry_date
    entry_ts = pd.Timestamp(t0["entry_date"], tz="UTC")
    row = df[df["ts"].dt.normalize() == entry_ts].iloc[0]
    assert abs(float(t0["entry_px"]) - float(row["open"])) < 1e-3


def test_friction_reduces_return():
    df = _synth_df()
    a = backtest_roc_rebound(df, friction_pct=0.0)
    b = backtest_roc_rebound(df, friction_pct=0.25)
    assert a["n"] == b["n"]
    assert a["n"] >= 1
    assert b["trades"][0]["ret_pct"] == pytest.approx(a["trades"][0]["ret_pct"] - 0.25)


def test_sweeps_return_rows():
    df = _synth_df(n=60)
    lengths = sweep_roc_lengths(df, lengths=[5, 7])
    assert len(lengths) == 2
    exits = sweep_exit_thresholds(df, exits=[2.0, 3.0])
    assert len(exits) == 2


def test_empty_frame():
    r = backtest_roc_rebound(pd.DataFrame())
    assert r["n"] == 0
    assert r.get("reason") == "empty"


def test_sma200_can_block_entries():
    # Short synth never reaches SMA200 warmup -> 0 trades with filter.
    df = _synth_df(n=40)
    r = backtest_roc_rebound(df, require_sma200=True)
    assert r["n"] == 0


def test_time_stop_caps_hold():
    df = _synth_df(n=80)
    # Force a long hold: exit threshold unreachable.
    r = backtest_roc_rebound(df, exit_thresh_pct=50.0, time_stop_bars=5)
    assert r["n"] >= 1
    assert all(t["hold_days"] <= 5 for t in r["trades"])
    assert any(t.get("exit_reason") == "time_stop" for t in r["trades"])


def test_vol_scale_runs():
    df = _synth_df(n=80)
    r = backtest_roc_rebound(df, vol_scale=True, vol_k=1.0, vol_lookback=10)
    assert "params" in r
    assert r["params"]["vol_scale"] is True


def test_occupancy_fcfs_single_slot():
    from utils.research.roc_rebound import occupancy_fcfs

    trades = [
        {
            "symbol": "AAA",
            "entry_date": "2020-01-02",
            "exit_date": "2020-01-10",
            "entry_px": 10,
            "exit_px": 11,
            "ret_pct": 10,
            "entry_roc": -5.0,
        },
        {
            "symbol": "BBB",
            "entry_date": "2020-01-03",
            "exit_date": "2020-01-08",
            "entry_px": 10,
            "exit_px": 12,
            "ret_pct": 20,
            "entry_roc": -8.0,
        },
        {
            "symbol": "CCC",
            "entry_date": "2020-01-11",
            "exit_date": "2020-01-15",
            "entry_px": 10,
            "exit_px": 10.5,
            "ret_pct": 5,
            "entry_roc": -4.0,
        },
    ]
    kept, meta = occupancy_fcfs(trades, max_open=1)
    assert meta["n_kept"] == 2
    assert meta["n_skipped"] == 1
    assert [t["symbol"] for t in kept] == ["AAA", "CCC"]


def test_occupancy_ranked_prefers_deeper_same_day():
    from utils.research.roc_rebound_capital import occupancy_ranked

    trades = [
        {
            "symbol": "WEAK",
            "entry_date": "2020-01-02",
            "exit_date": "2020-01-10",
            "entry_px": 10,
            "exit_px": 11,
            "ret_pct": 10,
            "entry_roc": -3.1,
        },
        {
            "symbol": "DEEP",
            "entry_date": "2020-01-02",
            "exit_date": "2020-01-10",
            "entry_px": 10,
            "exit_px": 11,
            "ret_pct": 10,
            "entry_roc": -8.0,
        },
    ]
    kept, meta = occupancy_ranked(trades, max_open=1, rank_by="deepest_roc")
    assert meta["n_kept"] == 1
    assert kept[0]["symbol"] == "DEEP"
