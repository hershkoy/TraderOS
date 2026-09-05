"""Unit tests for H2-then-L3 rail-touch entry."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    _ensure_fill_not_below_support,
    _h2_rail_tag_fills,
    _h2_rail_tag_fills_on_15m,
    _l3_rail_touch,
    _limit_fill_at_support,
    _map_15m_to_daily_i,
    _reentry_or_breakout_fill,
    _shakeout_breakout_fill,
    _shakeout_rebuy_fill,
    _support_tagged,
    trades_for_symbol,
)


def test_support_tag_and_limit_fill():
    assert _support_tagged(10.0, 12.0, 10.5, 1.2)
    assert not _support_tagged(11.0, 12.0, 10.0, 1.2)
    fill = _limit_fill_at_support(10.5, 10.0, 12.0, 0.001)
    assert fill is not None
    assert abs(fill - 10.5 * 1.001) < 1e-9
    assert _limit_fill_at_support(10.5, 9.0, 10.2, 0.001) is None


def test_gap_through_is_not_l3_rail_touch():
    """WTFC 2019-07-16: entire bar below support is not a from-above tag."""
    support = 67.14
    assert not _l3_rail_touch(66.49, 63.77, 65.07, support, 1.2)
    assert _l3_rail_touch(11.45, 10.505, 11.32, 10.44, 1.2)


def test_touch_error_pct_zero_requires_real_intersection():
    """0.24% near-miss is a tag; 0% requires low <= support <= high."""
    support = 100.0
    # Low 0.20% above support: tags at 0.24%, not at 0%.
    assert _support_tagged(100.20, 101.0, support, 0.24)
    assert not _support_tagged(100.20, 101.0, support, 0.0)
    assert not _l3_rail_touch(101.0, 100.20, 100.80, support, 0.0)
    # Real wick through the rail.
    assert _support_tagged(99.90, 101.0, support, 0.0)
    assert _l3_rail_touch(101.0, 99.90, 100.50, support, 0.0)


def test_l3_touch_fills_at_support_not_mid_channel():
    n = 160
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    support = 10.0 + 0.03 * np.arange(n)
    close = support + 1.0
    high = close + 0.4
    low = close - 0.4
    l1, h1, l2, h2, l3 = 20, 32, 44, 58, 72
    low[l1] = support[l1]
    high[l1] = support[l1] + 0.3
    close[l1] = support[l1] + 0.15
    high[h1] = support[h1] + 2.0
    low[h1] = support[h1] + 1.4
    close[h1] = support[h1] + 1.8
    low[l2] = support[l2]
    high[l2] = support[l2] + 0.3
    close[l2] = support[l2] + 0.15
    high[h2] = support[h2] + 2.0
    low[h2] = support[h2] + 1.4
    close[h2] = support[h2] + 1.8
    low[l3] = support[l3] - 0.05
    high[l3] = support[l3] + 0.8
    close[l3] = support[l3] + 0.6
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1e5},
        index=idx,
    )
    rows = trades_for_symbol(
        "TEST",
        df,
        entry_mode="l3_touch",
        pivot_len=3,
        entry_features=False,
        squeeze_adaptive=False,
        stop_pct=0.03,
        trail_pct=0.10,
        min_bars_apart=8,
        max_low_pivots=16,
        error_pct=2.0,
        min_rally_pct=2.0,
        min_pullback_pct=2.0,
        min_total_rise_pct=2.0,
        entry_slip_pct=0.001,
        max_l3_wait_bars=40,
    )
    assert rows, "expected an L3 rail-touch trade"
    trade = rows[0]
    buy = pd.Timestamp(trade["buy_date"])
    assert buy <= idx[l3 + 2]
    pos = float(trade["channel_pos"])
    assert pos <= 0.35, pos


def test_l3_touch_two_highs_after_l2_no_rail_h1():
    """EYE-style: Nov peak inside the channel, both resistance touches after L2."""
    n = 160
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    support = 10.0 + 0.03 * np.arange(n)
    close = support + 1.0
    high = close + 0.4
    low = close - 0.4
    l1, h_inside, l2, h2a, h2b, l3 = 20, 32, 44, 58, 72, 90
    low[l1] = support[l1]
    high[l1] = support[l1] + 0.3
    close[l1] = support[l1] + 0.15
    high[h_inside] = support[h_inside] + 1.65
    low[h_inside] = support[h_inside] + 0.9
    close[h_inside] = support[h_inside] + 1.3
    low[l2] = support[l2]
    high[l2] = support[l2] + 0.3
    close[l2] = support[l2] + 0.15
    for h in (h2a, h2b):
        high[h] = support[h] + 2.0
        low[h] = support[h] + 1.4
        close[h] = support[h] + 1.8
    low[l3] = support[l3] - 0.05
    high[l3] = support[l3] + 0.8
    close[l3] = support[l3] + 0.6
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1e5},
        index=idx,
    )
    rows = trades_for_symbol(
        "TEST",
        df,
        entry_mode="l3_touch",
        pivot_len=3,
        entry_features=False,
        squeeze_adaptive=False,
        stop_pct=0.03,
        trail_pct=0.10,
        min_bars_apart=8,
        max_low_pivots=16,
        error_pct=2.0,
        min_rally_pct=2.0,
        min_pullback_pct=2.0,
        min_total_rise_pct=2.0,
        entry_slip_pct=0.001,
        max_l3_wait_bars=40,
    )
    assert rows, "expected L3 fill when both resistance touches are after L2"
    trade = rows[0]
    assert pd.Timestamp(trade["buy_date"]) >= idx[l3] - pd.Timedelta(days=10)
    assert float(trade["channel_pos"]) <= 0.35


def test_min_l3_wait_skips_immediate_tag():
    n = 160
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    support = 10.0 + 0.03 * np.arange(n)
    close = support + 1.0
    high = close + 0.4
    low = close - 0.4
    l1, h1, l2, h2, l3 = 20, 32, 44, 58, 72
    low[l1] = support[l1]
    high[l1] = support[l1] + 0.3
    close[l1] = support[l1] + 0.15
    high[h1] = support[h1] + 2.0
    low[h1] = support[h1] + 1.4
    close[h1] = support[h1] + 1.8
    low[l2] = support[l2]
    high[l2] = support[l2] + 0.3
    close[l2] = support[l2] + 0.15
    high[h2] = support[h2] + 2.0
    low[h2] = support[h2] + 1.4
    close[h2] = support[h2] + 1.8
    low[l3] = support[l3] - 0.05
    high[l3] = support[l3] + 0.8
    close[l3] = support[l3] + 0.6
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1e5},
        index=idx,
    )
    kw = dict(
        entry_mode="l3_touch",
        pivot_len=3,
        entry_features=False,
        squeeze_adaptive=False,
        stop_pct=0.03,
        trail_pct=0.10,
        min_bars_apart=8,
        max_low_pivots=16,
        error_pct=2.0,
        min_rally_pct=2.0,
        min_pullback_pct=2.0,
        min_total_rise_pct=2.0,
        entry_slip_pct=0.001,
        max_l3_wait_bars=40,
    )
    assert trades_for_symbol("TEST", df, min_l3_wait_bars=1, **kw)
    late = trades_for_symbol("TEST", df, min_l3_wait_bars=80, **kw)
    assert late == []


def _rail_series(n=90, h2=12):
    y0, slope, width = 10.0, 0.02, 2.0
    high = np.full(n, np.nan)
    low = np.full(n, np.nan)
    close = np.full(n, np.nan)
    for i in range(n):
        sup = y0 + slope * i
        close[i] = sup + 1.0
        high[i] = close[i] + 0.25
        low[i] = close[i] - 0.25
    high[h2] = y0 + slope * h2 + width
    return y0, slope, width, high, low, close


def _tag_bar(high, low, close, i, y0, slope):
    sup = y0 + slope * i
    low[i] = sup - 0.04
    high[i] = sup + 0.45
    close[i] = sup + 0.25


def test_h2_rail_tag_fills_l4_skips_l3():
    y0, slope, width, high, low, close = _rail_series()
    h2 = 12
    _tag_bar(high, low, close, 22, y0, slope)
    for i in range(23, 36):
        sup = y0 + slope * i
        close[i] = sup + 1.1
        high[i] = close[i] + 0.2
        low[i] = close[i] - 0.2
    high[30] = y0 + slope * 30 + 2.4
    _tag_bar(high, low, close, 42, y0, slope)
    kw = dict(
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=h2,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_wait=6,
    )
    l3 = _h2_rail_tag_fills(high, low, close, entry_touch=3, **kw)
    l4 = _h2_rail_tag_fills(high, low, close, entry_touch=4, **kw)
    assert l3 and l3[0][0] == 22 and l3[0][2] == 3
    assert l4 and l4[0][0] == 42 and l4[0][2] == 4


def test_h2_rail_tag_fills_early_l3_aborts_l4():
    y0, slope, width, high, low, close = _rail_series()
    h2 = 12
    _tag_bar(high, low, close, 14, y0, slope)
    _tag_bar(high, low, close, 40, y0, slope)
    kw = dict(
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=h2,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_wait=6,
        entry_touch=4,
    )
    assert _h2_rail_tag_fills(high, low, close, **kw) == []


def test_l4_touch_trade_fills_second_tag():
    n = 180
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    support = 10.0 + 0.03 * np.arange(n)
    close = support + 1.0
    high = close + 0.4
    low = close - 0.4
    l1, h1, l2, h2, l3, l4 = 20, 32, 44, 58, 72, 96
    low[l1] = support[l1]
    high[l1] = support[l1] + 0.3
    close[l1] = support[l1] + 0.15
    high[h1] = support[h1] + 2.0
    low[h1] = support[h1] + 1.4
    close[h1] = support[h1] + 1.8
    low[l2] = support[l2]
    high[l2] = support[l2] + 0.3
    close[l2] = support[l2] + 0.15
    high[h2] = support[h2] + 2.0
    low[h2] = support[h2] + 1.4
    close[h2] = support[h2] + 1.8
    low[l3] = support[l3] - 0.05
    high[l3] = support[l3] + 0.8
    close[l3] = support[l3] + 0.6
    high[84] = support[84] + 2.2
    low[84] = support[84] + 1.3
    close[84] = support[84] + 1.7
    low[l4] = support[l4] - 0.05
    high[l4] = support[l4] + 0.8
    close[l4] = support[l4] + 0.6
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1e5},
        index=idx,
    )
    kw = dict(
        entry_mode="l3_touch",
        pivot_len=3,
        entry_features=False,
        squeeze_adaptive=False,
        stop_pct=0.03,
        trail_pct=0.10,
        min_bars_apart=8,
        max_low_pivots=16,
        error_pct=2.0,
        min_rally_pct=2.0,
        min_pullback_pct=2.0,
        min_total_rise_pct=2.0,
        entry_slip_pct=0.001,
        max_l3_wait_bars=80,
        min_l3_wait_bars=6,
    )
    t3 = trades_for_symbol("TEST", df, entry_touch=3, **kw)
    t4 = trades_for_symbol("TEST", df, entry_touch=4, **kw)
    assert t3, "expected L3 fill"
    assert pd.Timestamp(t3[0]["buy_date"]) <= idx[l3 + 2]
    assert int(t3[0]["touch_num"]) == 3
    assert t4, "expected L4 fill after leave-rail"
    assert int(t4[0]["touch_num"]) == 4
    assert pd.Timestamp(t4[0]["buy_date"]) >= idx[l4] - pd.Timedelta(days=5)
    assert pd.Timestamp(t4[0]["buy_date"]) > pd.Timestamp(t3[0]["buy_date"])


L3_TRADE_KW = dict(
    entry_mode="l3_touch",
    pivot_len=3,
    squeeze_adaptive=False,
    stop_pct=0.03,
    trail_pct=0.10,
    min_bars_apart=8,
    max_low_pivots=16,
    error_pct=2.0,
    min_rally_pct=2.0,
    min_pullback_pct=2.0,
    min_total_rise_pct=2.0,
    entry_slip_pct=0.001,
    max_l3_wait_bars=80,
    min_l3_wait_bars=6,
)


def _l3_daily_channel():
    n = 160
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    support = 10.0 + 0.03 * np.arange(n)
    close = support + 1.0
    high = close + 0.4
    low = close - 0.4
    l1, h1, l2, h2, l3 = 20, 32, 44, 58, 72
    low[l1] = support[l1]
    high[l1] = support[l1] + 0.3
    close[l1] = support[l1] + 0.15
    high[h1] = support[h1] + 2.0
    low[h1] = support[h1] + 1.4
    close[h1] = support[h1] + 1.8
    low[l2] = support[l2]
    high[l2] = support[l2] + 0.3
    close[l2] = support[l2] + 0.15
    high[h2] = support[h2] + 2.0
    low[h2] = support[h2] + 1.4
    close[h2] = support[h2] + 1.8
    low[l3] = support[l3] - 0.05
    high[l3] = support[l3] + 0.8
    close[l3] = support[l3] + 0.6
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1e5},
        index=idx,
    )
    return df, idx, support, h2, l3


def test_l3_touch_prior_bar_snapshot_skips_fill_close():
    df, _idx, _support, _h2, _l3 = _l3_daily_channel()
    leaky = trades_for_symbol("TEST", df, entry_features=True, feature_asof_prior_bar=False, **L3_TRADE_KW)
    lagged = trades_for_symbol("TEST", df, entry_features=True, feature_asof_prior_bar=True, **L3_TRADE_KW)
    assert leaky and lagged
    fill_i = int(leaky[0]["entry_i"])
    from utils.research.channel_touch_entry_features import (  # noqa: WPS433
        snapshot_stock_features,
        stock_entry_feature_series,
    )

    series = stock_entry_feature_series(df)
    prior = snapshot_stock_features(series, fill_i - 1)
    fill = snapshot_stock_features(series, fill_i)
    assert lagged[0]["close_loc"] == prior["close_loc"]
    assert leaky[0]["close_loc"] == fill["close_loc"]
    assert lagged[0]["close_loc"] != leaky[0]["close_loc"]
    assert lagged[0]["rsi_14"] == prior["rsi_14"]
    assert int(lagged[0]["entry_i"]) == fill_i


def test_hybrid_skips_when_15m_panel_missing():
    df, _idx, _support, _h2, _l3 = _l3_daily_channel()
    daily = trades_for_symbol("TEST", df, entry_features=False, **L3_TRADE_KW)
    assert daily, "daily wick path should still fill"
    skipped = trades_for_symbol(
        "TEST",
        df,
        entry_features=False,
        intraday_fill="15m",
        df_15m=None,
        **L3_TRADE_KW,
    )
    assert skipped == []
    empty = trades_for_symbol(
        "TEST",
        df,
        entry_features=False,
        intraday_fill="15m",
        df_15m=pd.DataFrame(),
        **L3_TRADE_KW,
    )
    assert empty == []


def test_h2_rail_tag_on_15m_fourth_bar():
    daily_idx = pd.date_range("2024-03-01", periods=12, freq="B")
    h2 = 3
    fill_di = 9
    y0, slope, width = 10.0, 0.03, 2.0
    times = pd.date_range(daily_idx[fill_di] + pd.Timedelta(hours=9, minutes=30), periods=6, freq="15min")
    n15 = len(times)
    sup = y0 + slope * (fill_di - 0)
    close = np.full(n15, sup + 1.0)
    high = close + 0.25
    low = close - 0.25
    i_tag = 3
    low[i_tag] = sup - 0.04
    high[i_tag] = sup + 0.45
    close[i_tag] = sup + 0.25
    daily_i = _map_15m_to_daily_i(pd.DatetimeIndex(times), pd.DatetimeIndex(daily_idx))
    assert int(daily_i[i_tag]) == fill_di
    tags = _h2_rail_tag_fills_on_15m(
        high,
        low,
        close,
        daily_i,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=h2,
        error_pct=1.2,
        slip=0.001,
        wait_daily=20,
        min_wait_daily=1,
        entry_touch=3,
        h2_high=sup + 3.0,
    )
    assert tags, "expected 15m tag"
    assert tags[0][0] == i_tag


def test_hybrid_tag_bar4_features_from_bar3():
    df, idx, support, h2, l3 = _l3_daily_channel()
    fill_di = l3
    fill_day = idx[fill_di]
    times = pd.date_range(fill_day + pd.Timedelta(hours=9, minutes=30), periods=8, freq="15min")
    sup = float(support[fill_di])
    n15 = len(times)
    close15 = np.full(n15, sup + 1.0)
    high15 = close15 + 0.3
    low15 = close15 - 0.3
    for i in range(3):
        low15[i] = sup + 0.70
        high15[i] = sup + 1.10
        close15[i] = low15[i] + 0.02
    i_tag = 3
    low15[i_tag] = sup - 0.04
    high15[i_tag] = sup + 0.80
    close15[i_tag] = high15[i_tag] - 0.02
    for i in range(4, n15):
        close15[i] = sup + 1.3
        high15[i] = close15[i] + 0.2
        low15[i] = close15[i] - 0.2
    df15 = pd.DataFrame(
        {"open": close15, "high": high15, "low": low15, "close": close15, "volume": 1e4},
        index=times,
    )
    from utils.research.channel_touch_entry_features import (  # noqa: WPS433
        snapshot_stock_features,
        stock_entry_feature_series,
    )

    series15 = stock_entry_feature_series(df15)
    loc3 = snapshot_stock_features(series15, 2)["close_loc"]
    loc4 = snapshot_stock_features(series15, 3)["close_loc"]
    assert loc3 is not None and loc4 is not None
    assert abs(float(loc3) - float(loc4)) > 0.4

    rows = trades_for_symbol(
        "TEST",
        df,
        entry_features=True,
        intraday_fill="15m",
        df_15m=df15,
        **L3_TRADE_KW,
    )
    assert rows, "expected hybrid 15m fill"
    trade = rows[0]
    assert trade["buy_time"] == times[i_tag].strftime("%Y-%m-%d %H:%M")
    assert trade["close_loc"] == loc3
    assert trade["close_loc"] != loc4
    assert trade["feature_asof"] == times[2].strftime("%Y-%m-%d %H:%M")
    assert int(trade["entry_i"]) == i_tag


def _below_bar(high, low, close, i, y0, slope):
    sup = y0 + slope * i
    close[i] = sup * 0.97
    high[i] = sup * 0.99
    low[i] = sup * 0.95


def test_shakeout_rebuy_within_10_bars():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    h2 = 12
    _tag_bar(high, low, close, 22, y0, slope)
    for i in range(23, 27):
        _below_bar(high, low, close, i, y0, slope)
    _tag_bar(high, low, close, 31, y0, slope)
    kw = dict(
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=h2,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_wait=6,
        entry_touch=3,
    )
    off = _h2_rail_tag_fills(high, low, close, shakeout_rebuy_bars=0, **kw)
    assert len(off) == 1 and off[0][0] == 22 and off[0][3] is False
    tight = _h2_rail_tag_fills(high, low, close, shakeout_rebuy_bars=5, **kw)
    assert len(tight) == 1 and tight[0][0] == 22
    on = _h2_rail_tag_fills(high, low, close, shakeout_rebuy_bars=10, **kw)
    assert len(on) == 2
    assert on[0][0] == 22 and on[0][3] is False
    assert on[1][0] == 31 and on[1][3] is True
    extra = _shakeout_rebuy_fill(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        l3_i=22,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        shakeout_bars=10,
    )
    assert extra is not None and extra[0] == 31


def test_shakeout_rebuy_skips_without_close_below():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    _tag_bar(high, low, close, 22, y0, slope)
    _tag_bar(high, low, close, 31, y0, slope)
    kw = dict(
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_wait=6,
        entry_touch=3,
        shakeout_rebuy_bars=10,
    )
    tags = _h2_rail_tag_fills(high, low, close, **kw)
    assert len(tags) == 1 and tags[0][0] == 22


def test_shakeout_rebuy_cancels_if_close_above_resist():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    _tag_bar(high, low, close, 22, y0, slope)
    _below_bar(high, low, close, 23, y0, slope)
    sup24 = y0 + slope * 24
    close[24] = sup24 + width + 0.5
    high[24] = close[24] + 0.2
    low[24] = close[24] - 0.2
    _tag_bar(high, low, close, 31, y0, slope)
    extra = _shakeout_rebuy_fill(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        l3_i=22,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        shakeout_bars=10,
    )
    assert extra is None


def test_h2_resist_break_fills_close_above_rail():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    i = 22
    resist = y0 + slope * i + width
    close[i] = resist * 1.025
    high[i] = close[i] + 0.08
    low[i] = resist - 0.25
    kw = dict(
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_wait=6,
        entry_touch=3,
    )
    off = _h2_rail_tag_fills(high, low, close, h2_resist_break=False, **kw)
    assert off == []
    on = _h2_rail_tag_fills(high, low, close, h2_resist_break=True, **kw)
    assert len(on) == 1 and on[0][0] == 22 and on[0][4] is True


def test_h2_resist_break_error_pct_zero_fires_tick_above_rail():
    """RDWR Jun 9: close +0.69% over the rail is not a 1.2% break."""
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    i = 22
    resist = y0 + slope * i + width
    close[i] = resist * 1.0069
    high[i] = close[i] + 0.05
    low[i] = resist - 0.20
    kw = dict(
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        slip=0.001,
        wait=80,
        min_wait=6,
        entry_touch=3,
        h2_resist_break=True,
    )
    buffered = _h2_rail_tag_fills(high, low, close, error_pct=1.2, **kw)
    assert buffered == []
    tick = _h2_rail_tag_fills(high, low, close, error_pct=0.0, **kw)
    assert len(tick) == 1 and tick[0][0] == 22 and tick[0][4] is True


def test_h2_resist_break_skips_if_support_already_broken():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    _below_bar(high, low, close, 20, y0, slope)
    i = 22
    resist = y0 + slope * i + width
    close[i] = resist * 1.025
    high[i] = close[i] + 0.08
    low[i] = resist - 0.25
    kw = dict(
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_wait=6,
        entry_touch=3,
        h2_resist_break=True,
    )
    assert _h2_rail_tag_fills(high, low, close, **kw) == []


def test_reentry_or_breakout_fill_reclaims_support():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    for i in range(20, 24):
        _below_bar(high, low, close, i, y0, slope)
    i = 24
    sup = y0 + slope * i
    low[i] = sup - 0.08
    high[i] = sup + 0.55
    close[i] = sup + 0.20
    got = _reentry_or_breakout_fill(
        high,
        low,
        close,
        start_i=20,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
    )
    assert got is not None
    assert got[0] == 24
    assert got[2] is False
    assert got[1] + 1e-12 >= sup


def test_reentry_or_breakout_fill_resist_break():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    for i in range(20, 23):
        _below_bar(high, low, close, i, y0, slope)
    i = 23
    resist = y0 + slope * i + width
    close[i] = resist * 1.02
    high[i] = close[i] + 0.1
    low[i] = resist - 0.2
    got = _reentry_or_breakout_fill(
        high,
        low,
        close,
        start_i=20,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
    )
    assert got is not None
    assert got[0] == 23 and got[2] is True


def test_ensure_fill_not_below_support_keeps_in_channel():
    y0, slope, width, high, low, close = _rail_series(n=40, h2=12)
    i = 20
    sup = y0 + slope * i
    got = _ensure_fill_not_below_support(
        high,
        low,
        close,
        fill_i=i,
        fill_px=sup + 0.4,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
    )
    assert got == (20, sup + 0.4, False)


def test_ensure_fill_below_support_defers_to_reentry():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    for i in range(20, 24):
        _below_bar(high, low, close, i, y0, slope)
    i = 24
    sup = y0 + slope * i
    low[i] = sup - 0.08
    high[i] = sup + 0.55
    close[i] = sup + 0.20
    crash_px = (y0 + slope * 20) * 0.97
    got = _ensure_fill_not_below_support(
        high,
        low,
        close,
        fill_i=20,
        fill_px=crash_px,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
    )
    assert got is not None
    assert got[0] == 24
    assert got[1] + 1e-12 >= sup
    assert got[2] is False


def _resist_break_bar(high, low, close, i, y0, slope, width):
    resist = y0 + slope * i + width
    close[i] = resist * 1.025
    high[i] = close[i] + 0.08
    low[i] = resist - 0.25


def _inside_near_resist(high, low, close, i, y0, slope, width):
    resist = y0 + slope * i + width
    close[i] = resist * 0.995
    high[i] = close[i] + 0.05
    low[i] = close[i] - 0.05


def _h2_break_kw(n, h2=12):
    return dict(
        support_x0=0,
        support_y0=10.0,
        support_slope=0.02,
        width=2.0,
        h2=h2,
        n=n,
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_wait=6,
        entry_touch=3,
        h2_resist_break=True,
    )


def test_shakeout_breakout_fills_after_inside_then_rebreak():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    _resist_break_bar(high, low, close, 22, y0, slope, width)
    _inside_near_resist(high, low, close, 30, y0, slope, width)
    _resist_break_bar(high, low, close, 40, y0, slope, width)
    kw = _h2_break_kw(len(high))
    off = _h2_rail_tag_fills(high, low, close, shakeout_breakout=False, **kw)
    assert len(off) == 1 and off[0][0] == 22 and off[0][4] is True
    on = _h2_rail_tag_fills(high, low, close, shakeout_breakout=True, **kw)
    assert len(on) == 2
    assert on[0][0] == 22 and on[0][4] is True
    assert len(on[1]) > 5 and on[1][5] is True
    assert on[1][0] == 40 and on[1][4] is True
    extra = _shakeout_breakout_fill(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        first_i=22,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_inside_bars=1,
    )
    assert extra is not None and extra[0] == 40 and extra[2] >= 1


def test_shakeout_breakout_skips_if_never_inside():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    _resist_break_bar(high, low, close, 22, y0, slope, width)
    for i in range(23, 50):
        _resist_break_bar(high, low, close, i, y0, slope, width)
    extra = _shakeout_breakout_fill(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        first_i=22,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_inside_bars=1,
    )
    assert extra is None
    tags = _h2_rail_tag_fills(high, low, close, shakeout_breakout=True, **_h2_break_kw(len(high)))
    assert len(tags) == 1 and tags[0][0] == 22


def test_shakeout_breakout_cancels_if_support_broken():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    _resist_break_bar(high, low, close, 22, y0, slope, width)
    _below_bar(high, low, close, 28, y0, slope)
    _inside_near_resist(high, low, close, 30, y0, slope, width)
    _resist_break_bar(high, low, close, 40, y0, slope, width)
    extra = _shakeout_breakout_fill(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        first_i=22,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_inside_bars=1,
    )
    assert extra is None


def test_shakeout_breakout_min_inside_5_waits():
    y0, slope, width, high, low, close = _rail_series(n=80, h2=12)
    _resist_break_bar(high, low, close, 22, y0, slope, width)
    for i in range(23, 27):
        _inside_near_resist(high, low, close, i, y0, slope, width)
    _resist_break_bar(high, low, close, 27, y0, slope, width)
    early = _shakeout_breakout_fill(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        first_i=22,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_inside_bars=5,
    )
    assert early is None
    _inside_near_resist(high, low, close, 28, y0, slope, width)
    _resist_break_bar(high, low, close, 35, y0, slope, width)
    late = _shakeout_breakout_fill(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        first_i=22,
        h2=12,
        n=len(high),
        error_pct=1.2,
        slip=0.001,
        wait=80,
        min_inside_bars=5,
    )
    assert late is not None and late[0] == 35 and late[2] >= 5


def _h2_sbo_frame(after_first):
    """Daily channel with first resist-break at h2+8, then caller mutates after_first(df, h2, support)."""
    df, idx, support, h2, _l3 = _l3_daily_channel()
    width = 2.0
    i0 = int(h2) + 8
    resist = float(support[i0]) + width
    df.loc[idx[i0], "close"] = resist * 1.025
    df.loc[idx[i0], "high"] = resist * 1.025 + 0.08
    df.loc[idx[i0], "low"] = resist - 0.25
    after_first(df, idx, support, i0, width)
    return df, idx, i0


H2_SBO_KW = dict(
    **L3_TRADE_KW,
    h2_resist_break=True,
    h2_resist_break_only=True,
    shakeout_breakout=True,
    entry_features=False,
    atr_stop_mult=2.0,
    stop_pct_floor=0.015,
    stop_pct_ceil=0.06,
)


def test_shakeout_breakout_skips_while_first_trade_open():
    def _tiny_rebreak(df, idx, support, i0, width):
        i1 = i0 + 1
        i2 = i0 + 2
        resist1 = float(support[i1]) + width
        df.loc[idx[i1], "close"] = resist1 * 0.995
        df.loc[idx[i1], "high"] = resist1 * 0.995 + 0.04
        df.loc[idx[i1], "low"] = resist1 * 0.995 - 0.04
        resist2 = float(support[i2]) + width
        df.loc[idx[i2], "close"] = resist2 * 1.025
        df.loc[idx[i2], "high"] = resist2 * 1.025 + 0.08
        df.loc[idx[i2], "low"] = resist2 - 0.2

    df, _idx, i0 = _h2_sbo_frame(_tiny_rebreak)
    rows = trades_for_symbol("TEST", df, **H2_SBO_KW)
    brk = [r for r in rows if r.get("resist_break")]
    assert brk, "expected first H2 resist-break"
    assert not any(r.get("shakeout_breakout") for r in brk)
    assert int(brk[0]["entry_i"]) == i0


def test_shakeout_breakout_hard_stop_filter():
    def _stop_then_rebreak(df, idx, support, i0, width):
        entry = float(df.loc[idx[i0], "close"])
        crash = i0 + 1
        df.loc[idx[crash], "open"] = entry
        df.loc[idx[crash], "high"] = entry
        df.loc[idx[crash], "close"] = entry * 0.92
        df.loc[idx[crash], "low"] = entry * 0.91
        for j in range(crash + 1, crash + 6):
            resist = float(support[j]) + width
            mid = resist * 0.995
            df.loc[idx[j], "close"] = mid
            df.loc[idx[j], "high"] = mid + 0.05
            df.loc[idx[j], "low"] = mid - 0.05
        i2 = crash + 6
        resist = float(support[i2]) + width
        df.loc[idx[i2], "close"] = resist * 1.025
        df.loc[idx[i2], "high"] = resist * 1.025 + 0.08
        df.loc[idx[i2], "low"] = resist - 0.2

    df, _idx, _i0 = _h2_sbo_frame(_stop_then_rebreak)
    any_closed = trades_for_symbol("TEST", df, shakeout_breakout_hard_stop=False, **H2_SBO_KW)
    extras = [r for r in any_closed if r.get("shakeout_breakout")]
    assert extras, "expected second breakout after hard stop"
    assert extras[0].get("parent_exit_reason") == "hard_stop"
    hard_only = trades_for_symbol("TEST", df, shakeout_breakout_hard_stop=True, **H2_SBO_KW)
    assert [r for r in hard_only if r.get("shakeout_breakout")]

    def _trail_then_rebreak(df, idx, support, i0, width):
        entry = float(df.loc[idx[i0], "close"])
        peak_i = i0 + 3
        for j in range(i0 + 1, peak_i + 1):
            px = entry * (1.0 + 0.05 * (j - i0))
            df.loc[idx[j], "close"] = px
            df.loc[idx[j], "high"] = px + 0.1
            df.loc[idx[j], "low"] = px - 0.1
        peak = float(df.loc[idx[peak_i], "high"])
        trail_i = peak_i + 1
        stop_px = peak * 0.90
        df.loc[idx[trail_i], "high"] = peak
        df.loc[idx[trail_i], "open"] = peak
        df.loc[idx[trail_i], "close"] = stop_px * 0.99
        df.loc[idx[trail_i], "low"] = stop_px * 0.98
        for j in range(trail_i + 1, trail_i + 4):
            resist = float(support[j]) + width
            mid = resist * 0.995
            df.loc[idx[j], "close"] = mid
            df.loc[idx[j], "high"] = mid + 0.05
            df.loc[idx[j], "low"] = mid - 0.05
        i2 = trail_i + 4
        resist = float(support[i2]) + width
        df.loc[idx[i2], "close"] = resist * 1.025
        df.loc[idx[i2], "high"] = resist * 1.025 + 0.08
        df.loc[idx[i2], "low"] = resist - 0.2

    trail_df, _idx2, _ = _h2_sbo_frame(_trail_then_rebreak)
    trail_any = trades_for_symbol("TEST", trail_df, shakeout_breakout_hard_stop=False, **H2_SBO_KW)
    trail_extras = [r for r in trail_any if r.get("shakeout_breakout")]
    if trail_extras:
        assert trail_extras[0].get("parent_exit_reason") != "hard_stop"
        trail_hard = trades_for_symbol(
            "TEST", trail_df, shakeout_breakout_hard_stop=True, **H2_SBO_KW
        )
        assert not [r for r in trail_hard if r.get("shakeout_breakout")]

