"""Unit tests for channel-touch quality post-filters."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import enrich_rs, filter_trades, select_same_day_rs


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock": "AAA",
                "channel_start": "2025-01-01",
                "channel_end": "2025-06-01",
                "buy_date": "2025-06-15",
                "channel_pos": 0.3,
                "channel_width_pct": 10.0,
                "slope_pct_per_bar": 0.05,
                "adv_20": 1e7,
                "atr_pct": 2.0,
                "max_beyond_width": 0.1,
            },
            {
                "stock": "BBB",
                "channel_start": "2023-01-01",
                "channel_end": "2025-06-01",
                "buy_date": "2025-06-15",
                "channel_pos": 1.5,
                "channel_width_pct": 10.0,
                "slope_pct_per_bar": 0.05,
                "adv_20": 1e7,
                "atr_pct": 2.0,
                "max_beyond_width": 1.8,
            },
            {
                "stock": "CCC",
                "channel_start": "2024-01-01",
                "channel_end": "2024-06-01",
                "buy_date": "2025-06-15",
                "channel_pos": 0.8,
                "channel_width_pct": 10.0,
                "slope_pct_per_bar": 0.05,
                "adv_20": 1e7,
                "atr_pct": 2.0,
                "max_beyond_width": 0.4,
            },
        ]
    )


def test_require_in_channel_drops_above_resist():
    out = filter_trades(_sample(), require_in_channel=True)
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_max_channel_span_days():
    out = filter_trades(_sample(), max_channel_span_days=400)
    # AAA ~151d, BBB ~882d, CCC ~152d
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_max_channel_age_days():
    out = filter_trades(_sample(), max_channel_age_days=400)
    # AAA age ~165, BBB ~896, CCC ~531
    assert set(out["stock"]) == {"AAA"}


def test_combined_in_channel_and_span():
    out = filter_trades(_sample(), require_in_channel=True, max_channel_span_days=400)
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_max_beyond_width_filter_on_sample():
    out = filter_trades(_sample(), max_beyond_width=0.5)
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_select_same_day_rs_groups_calendar_date_not_time():
    df = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2024-06-03", "buy_time": "2024-06-03 10:00", "rs_spy_126d": 1.0},
            {"stock": "BBB", "buy_date": "2024-06-03", "buy_time": "2024-06-03 14:00", "rs_spy_126d": 5.0},
            {"stock": "CCC", "buy_date": "2024-06-04", "buy_time": "2024-06-04 10:00", "rs_spy_126d": 2.0},
        ]
    )
    kept = select_same_day_rs(df, max_per_day=1)
    assert set(kept["stock"]) == {"BBB", "CCC"}


def test_enrich_rs_session_bars():
    idx = pd.date_range("2024-01-02 09:30", periods=10, freq="15min")
    close = pd.Series([100.0, 101, 102, 103, 104, 110, 111, 112, 113, 120], index=idx)
    panel = pd.DataFrame({"close": close})
    trades = pd.DataFrame(
        [{"stock": "AAA", "buy_date": "2024-01-02", "buy_time": "2024-01-02 11:45"}]
    )
    # 11:45 is the 10th bar (index 9). lookback 2 sessions * 2 bars = 4 bars back.
    spy = panel.copy()
    out = enrich_rs(trades, {"AAA": panel}, spy, lookbacks=(2,), bars_per_session=2)
    assert "rs_spy_2d" in out.columns
    assert pd.notna(out.loc[0, "rs_spy_2d"])
    # identical stock and spy returns -> RS 0
    assert abs(float(out.loc[0, "rs_spy_2d"])) < 1e-9

