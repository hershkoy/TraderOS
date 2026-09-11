"""Bar-count windows for the Charts page."""
from __future__ import annotations

import pandas as pd

from utils.charting.ohlcv_window import (
    _scale_pads,
    _source_timeframe,
    center_index,
    clamp_pad,
    parse_around_ts,
    slice_bar_window,
)


def test_slice_bar_window_clamps():
    assert slice_bar_window(100, 0, 50, 50) == (0, 51)
    assert slice_bar_window(100, 99, 50, 50) == (49, 100)
    assert slice_bar_window(10, 5, 50, 50) == (0, 10)


def test_slice_bar_window_center():
    assert slice_bar_window(101, 50, 50, 50) == (0, 101)
    assert slice_bar_window(200, 100, 50, 50) == (50, 151)


def test_center_index_latest_when_missing():
    idx = pd.date_range("2023-06-01", periods=10, freq="h", tz="UTC")
    assert center_index(idx, None) == 9


def test_center_index_at_or_before():
    idx = pd.date_range("2023-06-30 13:30", periods=8, freq="h", tz="UTC")
    pos = center_index(idx, pd.Timestamp("2023-06-30 16:10", tz="UTC"))
    assert idx[pos] == pd.Timestamp("2023-06-30 15:30", tz="UTC")


def test_parse_around_date_only_end_of_utc_day():
    ts = parse_around_ts("2023-06-30")
    assert ts is not None
    assert ts.hour == 23
    assert ts.minute == 59


def test_clamp_pad_caps():
    assert clamp_pad("50", 50) == 50
    assert clamp_pad("1000", 50) == 1000
    assert clamp_pad("9999", 50) == 1000
    assert clamp_pad("nope", 50) == 50
    assert clamp_pad(-3, 50) == 0


def test_source_timeframe_picks_smaller_native():
    assert _source_timeframe("1h", ["15m", "1d"]) == "15m"
    assert _source_timeframe("1h", ["1h", "15m"]) == "1h"


def test_candidate_source_timeframes_exact_then_smaller():
    from utils.charting.ohlcv_window import candidate_source_timeframes

    assert candidate_source_timeframes("1h")[0] == "1h"
    assert candidate_source_timeframes("1h")[1:4] == ["30m", "15m", "5m"]
    assert "1m" in candidate_source_timeframes("1h")
    assert candidate_source_timeframes("15m") == ["15m", "5m", "1m"]


def test_scale_pads_15m_to_1h():
    assert _scale_pads("15m", "1h", 50, 50) == (200, 200)

