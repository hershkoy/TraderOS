"""Unit tests for channel-touch quality post-filters."""
from __future__ import annotations

from unittest import mock
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

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


def test_require_in_channel_drops_below_support():
    below = _sample().copy()
    below.loc[below["stock"] == "AAA", "channel_pos"] = -0.4
    out = filter_trades(below, require_in_channel=True)
    assert set(out["stock"]) == {"CCC"}


def test_require_in_channel_keeps_resist_break_above_rail():
    df = _sample().copy()
    df["resist_break"] = False
    df.loc[df["stock"] == "BBB", "resist_break"] = True
    out = filter_trades(df, require_in_channel=True)
    assert set(out["stock"]) == {"AAA", "BBB", "CCC"}


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
    assert mixed[0]["l1_price"] == 100.0
    assert mixed[0]["l2_price"] == round(float(df["low"].iloc[20]), 6)
    assert mixed[0]["channel_width"] == 8.0
    assert mixed[0]["l1_ms"] > 0
    assert mixed[0]["h2_ms"] > 0


def test_realistic_fill_15m_shifts_entry_and_price():
    df = _occ_ohlcv(80)
    ch_brk = _occ_setup(df, h2=21, width=9.0)

    def fake_fills(high, low, close, **kwargs):
        return [(50, 115.0, 3, False, True)]

    def fake_sim(high, low, close, dates, entry_i, **kwargs):
        exit_i = min(int(entry_i) + 10, len(close) - 1)
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
        h2_resist_break_only=True,
        entry_features=False,
        squeeze_adaptive=False,
        window_bars=None,
        pivot_len=5,
        min_l3_wait_bars=1,
        max_l3_wait_bars=252,
        include_time=True,
        realistic_fill=True,
        realistic_fill_mode="next-mid",
    )
    with mock.patch(
        "backtest_channel_touch_trades.find_h2_l3_setups", return_value=[ch_brk]
    ), mock.patch(
        "backtest_channel_touch_trades._h2_rail_tag_fills", side_effect=fake_fills
    ), mock.patch(
        "backtest_channel_touch_trades.exec_fill_15m_after_signal",
        return_value=(51, 111.11),
    ), mock.patch(
        "backtest_channel_touch_trades._simulate_trade", side_effect=fake_sim
    ):
        rows = trades_for_symbol("AAA", df, **scan)
    assert len(rows) == 1
    assert rows[0]["entry_i"] == 51
    assert rows[0]["buy_price"] == 111.11
    assert rows[0]["wait_bars"] == 50 - 21


def test_realistic_fill_15m_signal_close_stays_on_tag_bar():
    df = _occ_ohlcv(80)
    ch_brk = _occ_setup(df, h2=21, width=9.0)

    def fake_fills(high, low, close, **kwargs):
        return [(50, 115.0, 3, False, True)]

    def fake_sim(high, low, close, dates, entry_i, **kwargs):
        exit_i = min(int(entry_i) + 10, len(close) - 1)
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
        h2_resist_break_only=True,
        entry_features=False,
        squeeze_adaptive=False,
        window_bars=None,
        pivot_len=5,
        min_l3_wait_bars=1,
        max_l3_wait_bars=252,
        include_time=True,
        realistic_fill=True,
        realistic_fill_mode="signal-close",
    )
    with mock.patch(
        "backtest_channel_touch_trades.find_h2_l3_setups", return_value=[ch_brk]
    ), mock.patch(
        "backtest_channel_touch_trades._h2_rail_tag_fills", side_effect=fake_fills
    ), mock.patch(
        "backtest_channel_touch_trades.exec_fill_15m_after_signal",
        return_value=(50, 114.25),
    ), mock.patch(
        "backtest_channel_touch_trades._simulate_trade", side_effect=fake_sim
    ):
        rows = trades_for_symbol("AAA", df, **scan)
    assert len(rows) == 1
    assert rows[0]["entry_i"] == 50
    assert rows[0]["buy_price"] == 114.25
    assert rows[0]["wait_bars"] == 50 - 21


def test_realistic_fill_daily_drops_without_15m_print():
    df = _occ_ohlcv(80)
    ch_brk = _occ_setup(df, h2=21, width=9.0)

    def fake_fills(high, low, close, **kwargs):
        return [(50, 115.0, 3, False, True)]

    scan = dict(
        entry_mode="l3_touch",
        h2_resist_break=True,
        h2_resist_break_only=True,
        entry_features=False,
        squeeze_adaptive=False,
        window_bars=None,
        pivot_len=5,
        min_l3_wait_bars=1,
        max_l3_wait_bars=252,
        include_time=False,
        realistic_fill=True,
        df_15m=pd.DataFrame(),
    )
    with mock.patch(
        "backtest_channel_touch_trades.find_h2_l3_setups", return_value=[ch_brk]
    ), mock.patch(
        "backtest_channel_touch_trades._h2_rail_tag_fills", side_effect=fake_fills
    ):
        rows = trades_for_symbol("AAA", df, **scan)
    assert rows == []


def test_realistic_fill_daily_open_cross_uses_15m_close():
    from utils.research.realistic_purchaser import PurchaseResult, REASON_FILLED

    df = _occ_ohlcv(80)
    ch_brk = _occ_setup(df, h2=21, width=9.0)
    ts = pd.Timestamp("2025-06-13 13:45:00", tz="UTC")

    def fake_fills(high, low, close, **kwargs):
        return [(50, 115.0, 3, False, True)]

    def fake_sim(high, low, close, dates, entry_i, **kwargs):
        exit_i = min(int(entry_i) + 10, len(close) - 1)
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
        h2_resist_break_only=True,
        entry_features=False,
        squeeze_adaptive=False,
        window_bars=None,
        pivot_len=5,
        min_l3_wait_bars=1,
        max_l3_wait_bars=252,
        include_time=False,
        realistic_fill=True,
        realistic_fill_mode="open-cross",
        df_15m=pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}),
    )
    got = PurchaseResult(
        filled=True,
        reason=REASON_FILLED,
        fill_px=115.5,
        exec_bar_ts=ts,
    )
    with mock.patch(
        "backtest_channel_touch_trades.find_h2_l3_setups", return_value=[ch_brk]
    ), mock.patch(
        "backtest_channel_touch_trades._h2_rail_tag_fills", side_effect=fake_fills
    ), mock.patch(
        "backtest_channel_touch_trades.purchase_open_cross_15m",
        return_value=got,
    ), mock.patch(
        "backtest_channel_touch_trades._simulate_trade", side_effect=fake_sim
    ):
        rows = trades_for_symbol("AAA", df, **scan)
    assert len(rows) == 1
    assert rows[0]["buy_price"] == 115.5
    assert rows[0]["buy_time"] == "2025-06-13 13:45"


def test_realistic_fill_daily_next_open_uses_next_bar_open():
    df = _occ_ohlcv(80)
    df.iloc[51, df.columns.get_loc("open")] = 118.25
    ch_brk = _occ_setup(df, h2=21, width=9.0)

    def fake_fills(high, low, close, **kwargs):
        return [(50, 115.0, 3, False, True)]

    captured = {}

    def fake_sim(high, low, close, dates, entry_i, **kwargs):
        captured["entry_i"] = int(entry_i)
        captured["entry_px"] = kwargs.get("entry_px")
        captured["skip"] = kwargs.get("skip_entry_bar_stop")
        exit_i = min(int(entry_i) + 10, len(close) - 1)
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
        h2_resist_break_only=True,
        entry_features=False,
        squeeze_adaptive=False,
        window_bars=None,
        pivot_len=5,
        min_l3_wait_bars=1,
        max_l3_wait_bars=252,
        include_time=False,
        realistic_fill=True,
        realistic_fill_mode="next-open",
    )
    with mock.patch(
        "backtest_channel_touch_trades.find_h2_l3_setups", return_value=[ch_brk]
    ), mock.patch(
        "backtest_channel_touch_trades._h2_rail_tag_fills", side_effect=fake_fills
    ), mock.patch(
        "backtest_channel_touch_trades._simulate_trade", side_effect=fake_sim
    ):
        rows = trades_for_symbol("AAA", df, **scan)
    assert len(rows) == 1
    assert captured["entry_i"] == 51
    assert captured["entry_px"] == pytest.approx(118.25)
    assert captured["skip"] is False
    assert rows[0]["buy_price"] == pytest.approx(118.25)
    assert rows[0]["buy_date"] == df.index[51].strftime("%Y-%m-%d")


def test_channel_rail_fields_iso_utc():
    from backtest_channel_touch_trades import channel_rail_fields, iso_utc_ms

    idx = pd.bdate_range("2024-06-03", periods=30)
    low = np.linspace(10.0, 12.0, 30)
    ch = {
        "support_x0": 2,
        "support_y0": 10.5,
        "support_slope": 0.02,
        "channel_width": 1.25,
        "l1_idx": 2,
        "l2_idx": 8,
        "h2_idx": 12,
    }
    rails = channel_rail_fields(ch, idx, low)
    iso, ms = iso_utc_ms(idx[2])
    assert rails["l1_time"] == iso
    assert rails["l1_ms"] == ms
    assert rails["l1_price"] == 10.5
    assert rails["l2_price"] == round(float(low[8]), 6)
    assert rails["channel_width"] == 1.25
    assert rails["h2_time"].endswith("Z")


def test_resist_arm_trail_holds_until_upper_rail():
    """Peak trail from entry: ratchet stop with highs; exit when low hits stop."""
    from backtest_channel_touch_trades import _simulate_trade

    # Entry 105, stop 3% -> 101.85. Peak 114 -> trail 110.58; bar5 low 110 exits.
    dates = pd.bdate_range("2025-01-02", periods=8)
    high = np.array([106, 107, 108, 112, 114, 113, 110, 109], dtype=float)
    low = np.array([104, 105, 106, 109, 111, 110, 109.5, 108], dtype=float)
    close = np.array([105, 106, 107, 111, 113, 112, 110, 109], dtype=float)
    out = _simulate_trade(
        high,
        low,
        close,
        dates,
        entry_i=0,
        stop_pct=0.03,
        trail_pct=0.03,
        resist_arm_trail=True,
        entry_px=105.0,
        skip_entry_bar_stop=True,
    )
    assert out is not None
    assert out["exit_reason"] == "peak_trail"
    assert out["exit_i"] == 5
    assert abs(out["sell_price"] - 114.0 * 0.97) < 1e-6
    assert abs(out["peak_price"] - 114.0) < 1e-6


def test_resist_arm_trail_hard_stop_before_arm():
    """If price never lifts the trail above entry-3%, that floor still binds."""
    from backtest_channel_touch_trades import _simulate_trade

    dates = pd.bdate_range("2025-01-02", periods=5)
    # Peak never above entry 105 -> trail stays at/below hard stop 101.85
    high = np.array([105.0, 104.8, 104.5, 104.0, 103.0])
    low = np.array([104.5, 101.0, 100.0, 99.0, 98.0])  # bar1 through 101.85
    close = np.array([104.9, 102.0, 101.0, 100.0, 99.0])
    out = _simulate_trade(
        high,
        low,
        close,
        dates,
        entry_i=0,
        stop_pct=0.03,
        trail_pct=0.03,
        resist_arm_trail=True,
        entry_px=105.0,
        skip_entry_bar_stop=True,
    )
    assert out is not None
    assert out["exit_reason"] == "hard_stop"
    assert out["exit_i"] == 1
    assert abs(out["sell_price"] - 105.0 * 0.97) < 1e-6


def test_peak_trail_exits_after_31_high_like_mtsi():
    """After high 31.03, stop=30.09; next bar low through stop exits."""
    from backtest_channel_touch_trades import _simulate_trade

    dates = pd.bdate_range("2020-01-02", periods=6)
    high = np.array([16.0, 20.0, 25.0, 31.03, 30.50, 28.0])
    low = np.array([15.2, 19.5, 24.5, 30.50, 29.80, 27.0])  # bar4 low 29.80 < 30.09
    close = np.array([15.5, 19.8, 24.8, 30.80, 30.00, 27.5])
    out = _simulate_trade(
        high,
        low,
        close,
        dates,
        entry_i=0,
        stop_pct=0.03,
        trail_pct=0.03,
        resist_arm_trail=True,
        entry_px=15.33,
        skip_entry_bar_stop=True,
    )
    assert out is not None
    assert out["exit_reason"] == "peak_trail"
    assert out["exit_i"] == 4
    assert abs(out["sell_price"] - 31.03 * 0.97) < 1e-6
    assert abs(out["peak_price"] - 31.03) < 1e-6


def test_peak_trail_width_time_decay():
    from backtest_channel_touch_trades import _peak_trail_width

    w0 = _peak_trail_width(
        mode="time_decay",
        trail_pct=0.04,
        bars_held=0,
        peak=100.0,
        entry_px=100.0,
        trail_floor=0.01,
        trail_decay_per_bar=0.0002,
    )
    assert abs(w0 - 0.04) < 1e-12
    w50 = _peak_trail_width(
        mode="time_decay",
        trail_pct=0.04,
        bars_held=50,
        peak=100.0,
        entry_px=100.0,
        trail_floor=0.01,
        trail_decay_per_bar=0.0002,
    )
    assert abs(w50 - (0.04 - 50 * 0.0002)) < 1e-12
    w_floor = _peak_trail_width(
        mode="time_decay",
        trail_pct=0.04,
        bars_held=1000,
        peak=100.0,
        entry_px=100.0,
        trail_floor=0.01,
        trail_decay_per_bar=0.0002,
    )
    assert abs(w_floor - 0.01) < 1e-12


def test_peak_trail_width_gain_tighten():
    from backtest_channel_touch_trades import _peak_trail_width

    # +3% peak from entry -> floor(3)=3 steps * 0.33pp
    w = _peak_trail_width(
        mode="gain_tighten",
        trail_pct=0.04,
        bars_held=10,
        peak=103.0,
        entry_px=100.0,
        trail_floor=0.01,
        trail_tighten_per_pct=0.0033,
    )
    assert abs(w - (0.04 - 3 * 0.0033)) < 1e-12


def test_peak_trail_time_decay_sim_tightens():
    """After many bars, 4% trail decays enough that a shallow dip exits."""
    from backtest_channel_touch_trades import _simulate_trade

    n = 160
    dates = pd.bdate_range("2024-01-02", periods=n)
    # Flat peak at 110 after bar 1; after 150 bars width=4%-3%=1%; stop=108.9
    high = np.full(n, 110.0)
    low = np.full(n, 109.5)
    close = np.full(n, 109.8)
    high[0], low[0], close[0] = 105.0, 104.0, 105.0
    high[1], low[1], close[1] = 110.0, 108.0, 109.0
    low[155] = 108.5  # below 110*0.99=108.9 after decay
    out = _simulate_trade(
        high,
        low,
        close,
        dates,
        entry_i=0,
        stop_pct=0.04,
        trail_pct=0.04,
        peak_trail_mode="time_decay",
        trail_floor=0.01,
        trail_decay_per_bar=0.0002,
        entry_px=105.0,
        skip_entry_bar_stop=True,
    )
    assert out is not None
    assert out["exit_reason"] == "peak_trail"
    assert out["exit_i"] == 155

