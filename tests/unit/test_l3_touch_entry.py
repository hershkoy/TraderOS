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
    _h2_rail_tag_fills,
    _h2_rail_tag_fills_on_15m,
    _l3_rail_touch,
    _limit_fill_at_support,
    _map_15m_to_daily_i,
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


def test_gap_through_is_not_l3_rail_touch():
    """WTFC 2019-07-16: entire bar below support is not a from-above tag."""
    support = 67.14
    assert not _l3_rail_touch(66.49, 63.77, 65.07, support, 1.2)
    assert _l3_rail_touch(11.45, 10.505, 11.32, 10.44, 1.2)


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

