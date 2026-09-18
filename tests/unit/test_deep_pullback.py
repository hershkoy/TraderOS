"""Unit tests for Deep Pullback research helpers."""
from __future__ import annotations

import pandas as pd

from utils.research.deep_pullback import (
    backtest_deep_pullback,
    book_from_trades,
    geom_mean_year_pf,
    year_bucket_stats,
)


def _synth_uptrend(n: int = 260, start: float = 100.0) -> pd.DataFrame:
    """Mostly rising closes with one engineered deep pullback entry window."""
    rows = []
    px = start
    for i in range(n):
        # Mild drift up
        px = px * 1.001
        high = px * 1.005
        low = px * 0.995
        close = px
        rows.append(
            {
                "ts": pd.Timestamp("2019-01-02", tz="UTC") + pd.Timedelta(days=i),
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000,
            }
        )
    df = pd.DataFrame(rows)
    # Carve a sharp drop that is both 15d low close and worst 10d ret, still > SMA200
    # Use last segment so SMA200 exists.
    i = n - 5
    base = float(df.loc[i - 1, "close"])
    # Drop ~3% in one day (likely worst ret) and make it the lowest close of last 15
    drop = base * 0.97
    df.loc[i, "close"] = drop
    df.loc[i, "open"] = base
    df.loc[i, "high"] = base
    df.loc[i, "low"] = drop * 0.99
    # Next day rebound above prior high -> QS exit
    rebound = float(df.loc[i, "high"]) * 1.01
    df.loc[i + 1, "open"] = drop
    df.loc[i + 1, "low"] = drop
    df.loc[i + 1, "high"] = rebound
    df.loc[i + 1, "close"] = rebound
    return df


def test_backtest_finds_engineered_pullback():
    df = _synth_uptrend()
    res = backtest_deep_pullback(df, symbol="TEST", friction_pct=0.0)
    assert res["n"] >= 1
    assert res["trades"][0]["symbol"] == "TEST"
    assert res["trades"][0]["ret_pct"] > 0


def test_book_and_year_buckets_empty():
    assert book_from_trades([])["n"] == 0
    assert year_bucket_stats([]) == []
    assert geom_mean_year_pf([]) is None


def test_short_history_returns_reason():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=50, tz="UTC"),
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
        }
    )
    res = backtest_deep_pullback(df, symbol="X")
    assert res["n"] == 0
    assert res.get("reason") == "short_history"
