"""Unit tests for L1-to-buy channel outside-area ratio."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from utils.research.channel_outside_area import (
    SKIP_NO_L1,
    areas_for_trade,
    channel_outside_areas,
    stamp_date,
    support_slope_from_row,
)


def test_stamp_date_keeps_utc_midnight_calendar():
    assert stamp_date("2018-12-24T00:00:00Z") == date(2018, 12, 24)
    assert stamp_date("2019-04-29") == date(2019, 4, 29)
    assert stamp_date(datetime(2019, 4, 29, 19, 45)) == date(2019, 4, 29)


def test_flat_channel_undershoot_ratio():
    # support=10, width=2, 5 bars. Lows dip 1 then 2 below.
    low = np.array([10.0, 10.0, 9.0, 8.0, 10.0])
    high = low + 1.0
    close = low + 0.5
    got = channel_outside_areas(
        low,
        high,
        close,
        support_y0=10.0,
        support_x0=0,
        support_slope=0.0,
        width=2.0,
        start_i=0,
        end_i=4,
    )
    assert got["n_valid"] == 5.0
    assert got["channel_area"] == 10.0
    assert abs(got["outside_below_low_area"] - 3.0) < 1e-12
    assert abs(got["outside_below_low_ratio"] - 0.3) < 1e-12
    assert abs(got["max_undershoot_width"] - 1.0) < 1e-12
    assert got["n_bars_below_low"] == 2.0


def test_sloped_support_last_bar_only():
    # support 10, 10.1, 10.2; low tags until last bar 9.2
    low = np.array([10.0, 10.1, 9.2])
    high = np.array([11.0, 11.1, 10.5])
    close = np.array([10.5, 10.6, 9.8])
    got = channel_outside_areas(
        low,
        high,
        close,
        support_y0=10.0,
        support_x0=0,
        support_slope=0.1,
        width=2.0,
        start_i=0,
        end_i=2,
    )
    assert abs(got["outside_below_low_area"] - 1.0) < 1e-12
    assert abs(got["outside_below_low_ratio"] - (1.0 / 6.0)) < 1e-12
    assert abs(got["outside_below_close_area"] - 0.4) < 1e-12


def test_no_undershoot_is_zero():
    low = np.array([10.0, 10.2, 10.4])
    high = low + 2.0
    close = low + 1.0
    got = channel_outside_areas(
        low,
        high,
        close,
        support_y0=10.0,
        support_x0=0,
        support_slope=0.2,
        width=3.0,
        start_i=0,
        end_i=2,
    )
    assert got["outside_below_low_ratio"] == 0.0
    assert got["n_bars_below_low"] == 0.0


def test_invalid_width_is_nan():
    low = np.array([10.0, 11.0])
    got = channel_outside_areas(
        low,
        low + 1,
        low,
        support_y0=10.0,
        support_x0=0,
        support_slope=0.0,
        width=0.0,
        start_i=0,
        end_i=1,
    )
    assert np.isnan(got["outside_below_low_ratio"])


def test_slope_prefers_l1_l2_prices():
    row = {
        "l1_price": 24.27,
        "l2_price": 25.85,
        "slope_pct_per_bar": 0.148,
    }
    slope = support_slope_from_row(row, l1_i=0, l2_i=44)
    assert abs(slope - (25.85 - 24.27) / 44.0) < 1e-12


def test_slope_falls_back_to_pct():
    row = {"l1_price": 24.27, "slope_pct_per_bar": 0.148}
    slope = support_slope_from_row(row, l1_i=0, l2_i=None)
    assert abs(slope - 0.00148 * 24.27) < 1e-12


def test_areas_for_trade_l1_through_buy():
    idx = pd.date_range("2018-12-24", periods=6, freq="B")
    # L1 low on rail, then a 1.0 dip, buy on last bar.
    low = np.array([10.0, 10.2, 9.0, 10.6, 10.8, 11.0])
    df = pd.DataFrame(
        {
            "open": low + 0.2,
            "high": low + 1.0,
            "low": low,
            "close": low + 0.4,
            "volume": np.full(6, 1000.0),
        },
        index=idx,
    )
    row = {
        "l1_time": "2018-12-24T00:00:00Z",
        "l2_time": "2018-12-25T00:00:00Z",
        "buy_date": "2018-12-31",
        "l1_price": 10.0,
        "l2_price": 10.2,
        "channel_width": 2.0,
        "slope_pct_per_bar": 2.0,
    }
    stats, skip = areas_for_trade(row, df)
    assert skip is None
    assert stats is not None
    assert stats["n_bars"] == 6.0
    # one bar low=9 vs support ~10.4 -> about 1.4; exact from L1/L2 slope 0.2
    assert stats["outside_below_low_area"] > 1.0
    assert stats["outside_below_low_ratio"] > 0.0


def test_areas_for_trade_missing_l1():
    idx = pd.date_range("2019-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.5, 10.6, 10.7],
            "low": [9.8, 9.9, 10.0],
            "close": [10.1, 10.2, 10.3],
            "volume": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    row = {
        "l1_time": "2018-12-24T00:00:00Z",
        "buy_date": "2019-01-04",
        "l1_price": 10.0,
        "channel_width": 2.0,
        "slope_pct_per_bar": 0.1,
    }
    stats, skip = areas_for_trade(row, df)
    assert stats is None
    assert skip == SKIP_NO_L1
