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


def test_unique_ohlcv_sql_dedupes_by_ts():
    from utils.charting.ohlcv_window import UNIQUE_OHLCV_SELECT

    assert "DISTINCT ON (ts)" in UNIQUE_OHLCV_SELECT


def test_has_more_latest_window_needs_full_page():
    from utils.charting.ohlcv_window import _has_more

    short = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.bdate_range("2026-08-31", periods=14, tz="UTC"),
    )
    assert _has_more(short, None, 50, 50) == (False, False)
    full = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.bdate_range("2026-01-01", periods=100, tz="UTC"),
    )
    assert _has_more(full, None, 50, 50) == (True, False)


def test_load_ohlcv_window_unique_full_page_can_zoom_out(monkeypatch):
    from utils.charting.ohlcv_window import load_ohlcv_window

    idx = pd.bdate_range("2026-01-01", periods=100, tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=idx,
    )
    seen = {}

    def fake_unique(symbol, timeframe, **kwargs):
        seen["symbol"] = symbol
        seen["timeframe"] = timeframe
        seen["before"] = kwargs.get("before")
        seen["after"] = kwargs.get("after")
        return df

    class FakeClient:
        def ensure_connection(self):
            return True

    monkeypatch.setattr("utils.charting.ohlcv_window.fetch_unique_ohlcv_window", fake_unique)
    monkeypatch.setattr(
        "utils.db.timescaledb_client.get_timescaledb_client", lambda: FakeClient()
    )
    win = load_ohlcv_window("SPY", "1d", before=50, after=50)
    assert seen["symbol"] == "SPY"
    assert seen["timeframe"] == "1d"
    assert len(win["df"]) == 100
    assert win["has_more_before"] is True
    assert win["has_more_after"] is False


def test_market_data_window_sql_limits_unique_ts():
    import inspect

    from utils.db.timescaledb_client import TimescaleDBClient

    src = inspect.getsource(TimescaleDBClient.get_market_data_window)
    assert "DISTINCT ON (ts)" in src
    assert "created_at DESC NULLS LAST" in src

