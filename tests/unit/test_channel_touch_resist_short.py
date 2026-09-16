"""Unit tests for channel-touch resistance short (H3+ fade)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.research.channel_touch_resist_short import (
    find_h3_plus_resist_tags,
    geo_year_pf,
    limit_fill_at_resist,
    resist_rail_touch,
    simulate_short_trade,
)


def test_resist_rail_touch_from_below():
    assert resist_rail_touch(10.0, 9.0, 9.5, 10.0, 1.2) is True
    # Close broken out above rail
    assert resist_rail_touch(10.5, 9.5, 10.4, 10.0, 1.2) is False
    # Gap entirely above rail
    assert resist_rail_touch(11.0, 10.5, 10.8, 10.0, 1.2) is False


def test_limit_fill_at_resist_never_above_rail():
    fill = limit_fill_at_resist(10.0, 9.0, 10.2, 0.01)
    assert fill is not None
    assert fill <= 10.0 + 1e-12
    assert abs(fill - 9.9) < 1e-9
    # Never traded at/near rail
    assert limit_fill_at_resist(10.0, 8.0, 8.5, 0.0) is None


def test_simulate_short_support_cover_and_hard_stop():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    high = np.array([10.0, 10.1, 9.8, 9.5, 9.2, 9.0], dtype=float)
    low = np.array([9.5, 9.6, 9.4, 9.0, 8.5, 8.4], dtype=float)
    close = np.array([9.8, 9.7, 9.5, 9.1, 8.8, 8.7], dtype=float)
    sim = simulate_short_trade(
        high,
        low,
        close,
        dates,
        0,
        stop_pct=0.10,
        trail_pct=0.50,
        atr_stop_mult=None,
        support_x0=0,
        support_y0=8.5,
        support_slope=0.0,
        channel_width=1.5,
        support_exit=True,
        entry_px=10.0,
    )
    assert sim is not None
    assert sim["side"] == "short"
    assert sim["exit_reason"] == "support_exit"
    assert sim["gain_pct"] > 0

    high2 = np.array([10.0, 11.5, 11.6], dtype=float)
    low2 = np.array([9.5, 10.2, 10.3], dtype=float)
    close2 = np.array([9.8, 11.0, 11.1], dtype=float)
    dates2 = pd.date_range("2020-01-01", periods=3, freq="D")
    stop = simulate_short_trade(
        high2,
        low2,
        close2,
        dates2,
        0,
        stop_pct=0.05,
        trail_pct=0.50,
        atr_stop_mult=None,
        support_exit=False,
        entry_px=10.0,
    )
    assert stop is not None
    assert stop["exit_reason"] == "hard_stop"
    assert stop["gain_pct"] < 0


def test_find_h3_requires_leave_then_tag():
    # H2 at i=2 (high=12). Leave into channel, then tag resist ~12 again at i=6.
    n = 10
    high = np.full(n, 11.0)
    low = np.full(n, 10.0)
    close = np.full(n, 10.5)
    high[2] = 12.0
    close[2] = 11.8
    # leave
    for i in range(3, 6):
        high[i] = 11.2
        low[i] = 10.2
        close[i] = 10.8
    # H3 tag
    high[6] = 12.05
    low[6] = 11.0
    close[6] = 11.5
    tags = find_h3_plus_resist_tags(
        high,
        low,
        close,
        support_x0=0,
        support_y0=10.0,
        support_slope=0.0,
        width=2.0,
        h2=2,
        n=n,
        error_pct=1.2,
        slip=0.0,
        wait=20,
        min_wait=2,
        entry_touch=3,
        leave_width_frac=0.20,
    )
    assert len(tags) >= 1
    assert tags[0][2] == 3
    assert tags[0][0] == 6


def test_geo_year_pf_skips_thin_buckets():
    ydf = pd.DataFrame(
        [
            {"bucket": "2020-21", "n_trades": 40, "profit_factor": 1.5},
            {"bucket": "2022-23", "n_trades": 10, "profit_factor": 3.0},
            {"bucket": "2024-26", "n_trades": 50, "profit_factor": 1.0},
            {"bucket": "ALL", "n_trades": 100, "profit_factor": 1.2},
        ]
    )
    geo = geo_year_pf(ydf, min_n=30)
    assert geo is not None
    assert abs(geo - float(np.exp(np.mean(np.log([1.5, 1.0]))))) < 1e-9
