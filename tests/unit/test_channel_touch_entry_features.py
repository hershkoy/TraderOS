"""Unit tests for channel-touch entry features and max_beyond_width."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import filter_trades
from analyze_channel_touch_entry_features import analyze_entry_features
from utils.research.channel_touch_entry_features import (
    completed_asof,
    enrich_spy_entry_features,
    max_beyond_width,
    snapshot_stock_features,
    stock_entry_feature_series,
)


def test_max_beyond_width_never_above():
    high = np.array([10.0, 10.5, 11.0])
    # support 9, slope 0, width 2 -> resist 11; highs never exceed
    v = max_beyond_width(high, 9.0, 0, 0.0, 2.0, 0, 2)
    assert v == 0.0


def test_max_beyond_width_pierce():
    high = np.array([10.0, 13.0, 11.0])
    # resist 11, width 2, high 13 -> (13-11)/2 = 1.0
    v = max_beyond_width(high, 9.0, 0, 0.0, 2.0, 0, 2)
    assert abs(v - 1.0) < 1e-9


def test_max_beyond_width_invalid_width():
    high = np.array([10.0, 11.0])
    assert np.isnan(max_beyond_width(high, 9.0, 0, 0.0, 0.0, 0, 1))


def test_filter_max_beyond_width():
    df = pd.DataFrame(
        [
            {"stock": "AAA", "max_beyond_width": 0.1, "channel_pos": 0.5},
            {"stock": "BBB", "max_beyond_width": 1.5, "channel_pos": 0.5},
            {"stock": "CCC", "max_beyond_width": 0.25, "channel_pos": 0.5},
        ]
    )
    out = filter_trades(df, max_beyond_width=0.25)
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_filter_max_formation_beyond_width():
    df = pd.DataFrame(
        [
            {"stock": "ADM", "formation_beyond_width": 1.88, "channel_pos": 0.9},
            {"stock": "OK", "formation_beyond_width": 0.1, "channel_pos": 0.5},
            {"stock": "EDGE", "formation_beyond_width": 0.25, "channel_pos": 0.5},
        ]
    )
    out = filter_trades(df, max_formation_beyond_width=0.25)
    assert set(out["stock"]) == {"OK", "EDGE"}


def test_stock_features_sma_distance():
    n = 60
    close = np.arange(1.0, n + 1.0)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )
    series = stock_entry_feature_series(df)
    snap = snapshot_stock_features(series, n - 1)
    assert snap["rsi_14"] is not None
    assert snap["dist_sma50_pct"] is not None
    assert snap["dist_sma50_pct"] > 0
    assert snap["range_pct"] is not None
    assert snap["close_loc"] is not None
    sma50 = float(np.mean(close[-50:]))
    expected = (close[-1] / sma50 - 1.0) * 100.0
    assert abs(snap["dist_sma50_pct"] - expected) < 0.05


def test_enrich_spy_and_analyze_buckets():
    idx = pd.date_range("2024-01-02", periods=80, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, 80), index=idx)
    spy = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        }
    )
    trades = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": idx[-1].strftime("%Y-%m-%d"),
                "gain_pct": 5.0,
                "rsi_14": 70.0,
                "max_beyond_width": 0.1,
            },
            {
                "stock": "BBB",
                "buy_date": idx[-1].strftime("%Y-%m-%d"),
                "gain_pct": -2.0,
                "rsi_14": 30.0,
                "max_beyond_width": 1.2,
            },
            {
                "stock": "CCC",
                "buy_date": idx[-1].strftime("%Y-%m-%d"),
                "gain_pct": 1.0,
                "rsi_14": 50.0,
                "max_beyond_width": 0.4,
            },
        ]
    )
    out = enrich_spy_entry_features(trades, spy)
    assert out["spy_ret_20d"].notna().all()
    assert out["spy_above_sma50"].notna().all()
    buckets, spearman, _clones = analyze_entry_features(out, friction_pct=0.0)
    assert not buckets.empty
    assert "rsi_14" in set(spearman["feature"])


def test_completed_asof_daily_excludes_fill_session():
    row = {
        "buy_date": "2024-06-03",
        "buy_time": "2024-06-03 10:15",
        "feature_asof": "2024-06-03 10:00",
    }
    asof = completed_asof(row, series_is_daily=True)
    assert asof == pd.Timestamp("2024-06-02")
    intra = completed_asof(row, series_is_daily=False)
    assert intra == pd.Timestamp("2024-06-03 10:00")


def test_enrich_spy_does_not_use_fill_day_close():
    idx = pd.date_range("2024-01-02", periods=80, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, 80), index=idx)
    close.iloc[-1] = 999.0
    spy = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        }
    )
    trades = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": idx[-1].strftime("%Y-%m-%d"),
                "buy_time": idx[-1].strftime("%Y-%m-%d") + " 10:15",
                "feature_asof": idx[-1].strftime("%Y-%m-%d") + " 10:00",
                "gain_pct": 1.0,
            }
        ]
    )
    out = enrich_spy_entry_features(trades, spy)
    # last completed daily close is prior session, not 999
    assert float(out.loc[0, "spy_ret_20d"]) < 50.0


def test_enrich_spy_sma200_uses_prior_session():
    idx = pd.date_range("2023-01-02", periods=220, freq="B")
    close = pd.Series(np.linspace(100.0, 140.0, 220), index=idx)
    close.iloc[-1] = 10.0
    spy = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        }
    )
    trades = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": idx[-1].strftime("%Y-%m-%d"),
                "buy_time": idx[-1].strftime("%Y-%m-%d") + " 15:45",
                "gain_pct": 1.0,
            }
        ]
    )
    out = enrich_spy_entry_features(trades, spy)
    assert int(out.loc[0, "spy_above_sma200"]) == 1
