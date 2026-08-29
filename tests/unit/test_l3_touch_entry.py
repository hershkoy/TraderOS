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
    _l3_rail_touch,
    _limit_fill_at_support,
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
