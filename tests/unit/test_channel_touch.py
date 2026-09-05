"""Unit tests for live channel-touch trigger helpers."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd

from utils.scanning.channel_touch import (
    format_triggers_message,
    live_entries_for_symbol,
    scan_live_triggers,
)


def _ohlcv(n: int = 80, start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 120.0, n)
    high = close + 1.0
    low = close - 1.0
    open_ = close.copy()
    vol = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_live_entries_emits_when_entry_is_last_bar():
    df = _ohlcv()
    n = len(df)
    fake_channels = [
        {
            "touch_indices": [10, 30, n - 1 - 15],
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 5.0,
            "start_date": "2024-01-15",
            "end_date": df.index[-1].strftime("%Y-%m-%d"),
            "slope_pct_per_bar": 0.05,
            "channel_width_pct": 4.0,
            "pivot_len": 15,
        }
    ]
    with mock.patch("utils.scanning.channel_touch.find_channels", return_value=fake_channels):
        rows = live_entries_for_symbol(
            "AAA", df, atr_stop_mult=2.0, entry_mode="pivot", window_bars=0
        )
    assert len(rows) == 1
    assert rows[0]["stock"] == "AAA"
    assert rows[0]["buy_date"] == df.index[-1].strftime("%Y-%m-%d")
    assert rows[0]["hard_stop"] < rows[0]["buy_price"]
    assert rows[0]["entry_mode"] == "pivot"


def test_live_l3_touch_emits_fill_on_asof_bar():
    df = _ohlcv(n=100)
    n = len(df)
    h2 = n - 10
    fake_setups = [
        {
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 8.0,
            "h2_idx": h2,
            "h2_date": df.index[h2].strftime("%Y-%m-%d"),
            "start_date": df.index[10].strftime("%Y-%m-%d"),
            "end_date": df.index[h2].strftime("%Y-%m-%d"),
            "slope_pct_per_bar": 0.05,
            "channel_width_pct": 6.0,
            "pivot_len": 15,
        }
    ]
    fill_px = 110.5
    with mock.patch(
        "utils.scanning.channel_touch.find_h2_l3_setups", return_value=fake_setups
    ), mock.patch(
        "utils.scanning.channel_touch._h2_rail_tag_fills",
        return_value=[(n - 1, fill_px, 3)],
    ):
        rows = live_entries_for_symbol(
            "BBB",
            df,
            entry_mode="l3_touch",
            window_bars=0,
            min_l3_wait_bars=6,
            h2_resist_break=False,
            h2_resist_break_only=False,
        )
    assert len(rows) == 1
    assert rows[0]["stock"] == "BBB"
    assert rows[0]["buy_price"] == round(fill_px, 4)
    assert rows[0]["entry_mode"] == "l3_touch"
    assert rows[0]["wait_bars"] == (n - 1) - h2
    assert rows[0]["touch_num"] == 3


def test_live_l3_touch_skips_fill_on_other_bar():
    df = _ohlcv(n=100)
    n = len(df)
    fake_setups = [
        {
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 8.0,
            "h2_idx": n - 20,
            "h2_date": df.index[n - 20].strftime("%Y-%m-%d"),
            "start_date": df.index[10].strftime("%Y-%m-%d"),
            "end_date": df.index[n - 20].strftime("%Y-%m-%d"),
            "pivot_len": 15,
        }
    ]
    with mock.patch(
        "utils.scanning.channel_touch.find_h2_l3_setups", return_value=fake_setups
    ), mock.patch(
        "utils.scanning.channel_touch._h2_rail_tag_fills",
        return_value=[(n - 5, 110.0, 3)],
    ):
        rows = live_entries_for_symbol(
            "CCC",
            df,
            entry_mode="l3_touch",
            window_bars=0,
            h2_resist_break=False,
            h2_resist_break_only=False,
        )
    assert rows == []


def test_live_h2_resist_break_emits_fill_on_asof_bar():
    df = _ohlcv(n=100)
    n = len(df)
    h2 = n - 10
    fake_setups = [
        {
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 8.0,
            "h2_idx": h2,
            "h2_date": df.index[h2].strftime("%Y-%m-%d"),
            "start_date": df.index[10].strftime("%Y-%m-%d"),
            "end_date": df.index[h2].strftime("%Y-%m-%d"),
            "slope_pct_per_bar": 0.05,
            "channel_width_pct": 6.0,
            "pivot_len": 15,
        }
    ]
    fill_px = 118.25
    with mock.patch(
        "utils.scanning.channel_touch.find_h2_l3_setups", return_value=fake_setups
    ), mock.patch(
        "utils.scanning.channel_touch._h2_rail_tag_fills",
        return_value=[(n - 1, fill_px, 3, False, True)],
    ):
        rows = live_entries_for_symbol(
            "DDD",
            df,
            entry_mode="l3_touch",
            window_bars=0,
            min_l3_wait_bars=6,
        )
    assert len(rows) == 1
    assert rows[0]["stock"] == "DDD"
    assert rows[0]["buy_price"] == round(fill_px, 4)
    assert rows[0]["resist_break"] is True


def test_live_h2_resist_break_only_skips_l3_support_tag():
    df = _ohlcv(n=100)
    n = len(df)
    fake_setups = [
        {
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 8.0,
            "h2_idx": n - 10,
            "h2_date": df.index[n - 10].strftime("%Y-%m-%d"),
            "start_date": df.index[10].strftime("%Y-%m-%d"),
            "end_date": df.index[n - 10].strftime("%Y-%m-%d"),
            "pivot_len": 15,
        }
    ]
    with mock.patch(
        "utils.scanning.channel_touch.find_h2_l3_setups", return_value=fake_setups
    ), mock.patch(
        "utils.scanning.channel_touch._h2_rail_tag_fills",
        return_value=[(n - 1, 110.0, 3, False, False)],
    ):
        rows = live_entries_for_symbol("EEE", df, entry_mode="l3_touch", window_bars=0)
    assert rows == []


def _trigger_row(stock: str, *, rsi: float, idx) -> dict:
    return {
        "stock": stock,
        "buy_date": idx[-1].strftime("%Y-%m-%d"),
        "buy_price": 100.0,
        "hard_stop": 94.0,
        "stop_pct_used": 6.0,
        "channel_start": "2024-01-15",
        "channel_end": "2024-06-01",
        "touch_num": 3,
        "channel_pos": 0.05,
        "max_beyond_width": 0.10,
        "channel_span_days": 120,
        "rsi_14": rsi,
        "entry_mode": "l3_touch",
        "wait_bars": 12,
    }


def test_scan_quality_drops_high_rsi_before_rs():
    n = 140
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.linspace(100.0, 110.0, n)
    spy = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    aaa = spy.copy()
    bbb = spy.copy()
    bbb["close"] = close * 1.2
    panels = {"AAA": aaa, "BBB": bbb, "SPY": spy}

    def fake_live(symbol, df, **kwargs):
        if symbol == "AAA":
            return [_trigger_row("AAA", rsi=80.0, idx=idx)]
        if symbol == "BBB":
            return [_trigger_row("BBB", rsi=40.0, idx=idx)]
        return []

    stats = {}
    with mock.patch("utils.scanning.channel_touch.live_entries_for_symbol", side_effect=fake_live):
        out = scan_live_triggers(
            panels,
            symbols=["AAA", "BBB", "SPY"],
            spy_df=spy,
            workers=1,
            max_entries_per_day=1,
            max_rsi=50.0,
            stats=stats,
        )
    assert stats["n_raw"] == 2
    assert stats["n_quality"] == 1
    assert len(out) == 1
    assert out.iloc[0]["stock"] == "BBB"


def test_scan_live_keeps_all_symbols_without_rs_cap():
    n = 140
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.linspace(100.0, 110.0, n)
    spy = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    aaa = spy.copy()
    bbb = spy.copy()
    bbb["close"] = close * 1.2
    panels = {"AAA": aaa, "BBB": bbb, "SPY": spy}

    def fake_live(symbol, df, **kwargs):
        if symbol == "AAA":
            return [_trigger_row("AAA", rsi=40.0, idx=idx)]
        if symbol == "BBB":
            return [_trigger_row("BBB", rsi=41.0, idx=idx)]
        return []

    with mock.patch("utils.scanning.channel_touch.live_entries_for_symbol", side_effect=fake_live):
        out = scan_live_triggers(
            panels,
            symbols=["AAA", "BBB", "SPY"],
            spy_df=spy,
            workers=1,
            max_entries_per_day=0,
        )
    assert set(out["stock"]) == {"AAA", "BBB"}


def test_format_triggers_message_no_signal():
    msg = format_triggers_message(pd.DataFrame(), as_of="2026-08-25", n_candidates=0)
    assert "No new triggers" in msg
    assert "as_of=2026-08-25" in msg
    assert "mode=h2_resist_break" in msg
    assert "min_wait=6" in msg
    assert "max_rsi=off" in msg
    assert "shakeout_breakout=on" in msg


def test_live_shakeout_breakout_emits_extra_when_parent_closed():
    df = _ohlcv(n=100)
    n = len(df)
    h2 = n - 40
    parent_i = n - 20
    extra_i = n - 1
    fake_setups = [
        {
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 8.0,
            "h2_idx": h2,
            "h2_date": df.index[h2].strftime("%Y-%m-%d"),
            "start_date": df.index[10].strftime("%Y-%m-%d"),
            "end_date": df.index[h2].strftime("%Y-%m-%d"),
            "slope_pct_per_bar": 0.05,
            "channel_width_pct": 6.0,
            "pivot_len": 15,
        }
    ]
    tags = [
        (parent_i, 110.0, 3, False, True),
        (extra_i, 112.5, 3, False, True, True),
    ]
    with mock.patch(
        "utils.scanning.channel_touch.find_h2_l3_setups", return_value=fake_setups
    ), mock.patch(
        "utils.scanning.channel_touch._h2_rail_tag_fills",
        return_value=tags,
    ), mock.patch(
        "utils.scanning.channel_touch.first_trade_still_open",
        return_value=False,
    ):
        rows = live_entries_for_symbol(
            "SXI",
            df,
            entry_mode="l3_touch",
            window_bars=0,
            min_l3_wait_bars=6,
        )
    assert len(rows) == 1
    assert rows[0]["stock"] == "SXI"
    assert rows[0]["shakeout_breakout"] is True
    assert rows[0]["resist_break"] is True
    assert rows[0]["buy_price"] == 112.5


def test_live_shakeout_breakout_skips_extra_while_parent_open():
    df = _ohlcv(n=100)
    n = len(df)
    h2 = n - 40
    parent_i = n - 20
    extra_i = n - 1
    fake_setups = [
        {
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 8.0,
            "h2_idx": h2,
            "h2_date": df.index[h2].strftime("%Y-%m-%d"),
            "start_date": df.index[10].strftime("%Y-%m-%d"),
            "end_date": df.index[h2].strftime("%Y-%m-%d"),
            "pivot_len": 15,
        }
    ]
    tags = [
        (parent_i, 110.0, 3, False, True),
        (extra_i, 112.5, 3, False, True, True),
    ]
    with mock.patch(
        "utils.scanning.channel_touch.find_h2_l3_setups", return_value=fake_setups
    ), mock.patch(
        "utils.scanning.channel_touch._h2_rail_tag_fills",
        return_value=tags,
    ), mock.patch(
        "utils.scanning.channel_touch.first_trade_still_open",
        return_value=True,
    ):
        rows = live_entries_for_symbol(
            "SXI",
            df,
            entry_mode="l3_touch",
            window_bars=0,
            min_l3_wait_bars=6,
        )
    assert rows == []


def test_live_entries_passes_shakeout_breakout_to_fills():
    df = _ohlcv(n=80)
    n = len(df)
    fake_setups = [
        {
            "support_x0": 10,
            "support_y0": 100.0,
            "support_slope": 0.05,
            "channel_width": 8.0,
            "h2_idx": n - 20,
            "h2_date": df.index[n - 20].strftime("%Y-%m-%d"),
            "start_date": df.index[10].strftime("%Y-%m-%d"),
            "end_date": df.index[n - 20].strftime("%Y-%m-%d"),
            "pivot_len": 15,
        }
    ]
    with mock.patch(
        "utils.scanning.channel_touch.find_h2_l3_setups", return_value=fake_setups
    ), mock.patch(
        "utils.scanning.channel_touch._h2_rail_tag_fills",
        return_value=[],
    ) as fills:
        live_entries_for_symbol("AAA", df, entry_mode="l3_touch", window_bars=0)
    assert fills.call_args.kwargs["shakeout_breakout"] is True
    assert fills.call_args.kwargs["shakeout_breakout_min_inside"] == 1