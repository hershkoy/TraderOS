"""Unit tests for channel-touch quality post-filters."""
from __future__ import annotations

from unittest import mock
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (
    enrich_rs,
    filter_trades,
    keep_one_per_symbol_day,
    select_same_day_rs,
    trades_for_symbol,
)


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
                "rsi_14": 45.0,
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
                "rsi_14": 80.0,
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
                "rsi_14": 40.0,
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


def test_max_rsi_filter_on_sample():
    out = filter_trades(_sample(), max_rsi=50)
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_min_close_loc_filter():
    df = _sample().copy()
    df["close_loc"] = [0.8, 0.2, 0.55]
    out = filter_trades(df, min_close_loc=0.5)
    assert set(out["stock"]) == {"AAA", "CCC"}
    out = filter_trades(_sample(), max_rsi=50)
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


def test_keep_one_per_symbol_day_allows_many_names_drops_same_symbol():
    df = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2024-06-03", "buy_time": "2024-06-03 10:00", "gain_pct": 1.0},
            {"stock": "AAA", "buy_date": "2024-06-03", "buy_time": "2024-06-03 14:00", "gain_pct": 9.0},
            {"stock": "BBB", "buy_date": "2024-06-03", "buy_time": "2024-06-03 11:00", "gain_pct": 2.0},
            {"stock": "CCC", "buy_date": "2024-06-04", "buy_time": "2024-06-04 10:00", "gain_pct": 3.0},
        ]
    )
    kept = keep_one_per_symbol_day(df)
    assert len(kept) == 3
    assert set(kept["stock"]) == {"AAA", "BBB", "CCC"}
    aaa = kept.loc[kept["stock"] == "AAA"].iloc[0]
    assert str(aaa["buy_time"]).startswith("2024-06-03 10:00")


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


def _occ_ohlcv(n: int = 80) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _occ_setup(df: pd.DataFrame, *, h2: int, width: float) -> dict:
    return {
        "support_x0": 10,
        "support_y0": 100.0,
        "support_slope": 0.05,
        "channel_width": float(width),
        "h2_idx": int(h2),
        "l1_idx": 10,
        "l2_idx": 20,
        "h2_date": df.index[h2].strftime("%Y-%m-%d"),
        "start_date": df.index[10].strftime("%Y-%m-%d"),
        "end_date": df.index[h2].strftime("%Y-%m-%d"),
        "touch_indices": [10, 20],
        "slope_pct_per_bar": 0.05,
        "channel_width_pct": 6.0,
        "bars_span": int(h2 - 10),
        "pivot_len": 5,
    }


def test_h2_resist_break_only_skips_l3_occupancy():
    df = _occ_ohlcv(80)
    ch_l3 = _occ_setup(df, h2=20, width=8.0)
    ch_brk = _occ_setup(df, h2=21, width=9.0)

    def fake_fills(high, low, close, **kwargs):
        if int(kwargs["h2"]) == 20:
            return [(30, 110.0, 3, False, False)]
        return [(50, 115.0, 3, False, True)]

    def fake_sim(high, low, close, dates, entry_i, **kwargs):
        exit_i = min(int(entry_i) + 25, len(close) - 1)
        px = float(kwargs.get("entry_px") or close[entry_i])
        return {
            "buy_date": dates[entry_i].strftime("%Y-%m-%d"),
            "buy_price": px,
            "sell_date": dates[exit_i].strftime("%Y-%m-%d"),
            "sell_price": float(close[exit_i]),
            "gain_pct": 1.0,
            "hold_days": 10,
            "exit_reason": "trail_stop",
            "entry_i": int(entry_i),
            "exit_i": int(exit_i),
        }

    scan = dict(
        entry_mode="l3_touch",
        h2_resist_break=True,
        entry_features=False,
        squeeze_adaptive=False,
        window_bars=None,
        pivot_len=5,
        min_l3_wait_bars=1,
        max_l3_wait_bars=252,
    )
    with mock.patch(
        "backtest_channel_touch_trades.find_h2_l3_setups", return_value=[ch_l3, ch_brk]
    ), mock.patch(
        "backtest_channel_touch_trades._h2_rail_tag_fills", side_effect=fake_fills
    ), mock.patch(
        "backtest_channel_touch_trades._simulate_trade", side_effect=fake_sim
    ):
        mixed = trades_for_symbol("AAA", df, h2_resist_break_only=False, **scan)
        only = trades_for_symbol("AAA", df, h2_resist_break_only=True, **scan)

    assert [r["entry_i"] for r in mixed] == [30]
    assert [r["entry_i"] for r in only] == [50]
    assert only[0]["resist_break"] is True

