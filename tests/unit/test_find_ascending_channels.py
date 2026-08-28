"""Unit tests for classical channel finder wrappers."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from find_ascending_channels import find_channels, find_channels_windowed  # noqa: E402


def test_min_total_rise_pct_default_is_daily():
    params = inspect.signature(find_channels).parameters
    assert params["min_total_rise_pct"].default == 3.0


def test_find_channels_skips_zero_low_pivot():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    df = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.5,
            "low": 10.0,
            "close": 10.2,
            "volume": 1000.0,
        },
        index=idx,
    )
    df.loc[df.index[40], "low"] = 0.0
    # Should not raise ZeroDivisionError
    find_channels(df, pivot_len=5)


def test_windowed_remaps_slice_indices():
    idx = pd.date_range("2024-01-02 09:30", periods=80, freq="15min")
    df = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "volume": 1_000.0,
        },
        index=idx,
    )

    def fake_find(frame, **kwargs):
        return [
            {
                "touch_indices": [1, 4, 8],
                "support_x0": 1,
                "support_y0": 9.9,
                "support_slope": 0.01,
                "channel_width": 0.5,
                "start_date": "2024-01-02",
                "end_date": "2024-01-02",
                "bars_span": 7,
            }
        ]

    with mock.patch("find_ascending_channels.find_channels", side_effect=fake_find):
        out = find_channels_windowed(df, window_bars=20, step_bars=20, pivot_len=5)

    assert out
    assert out[0]["support_x0"] == 1
    assert out[0]["touch_indices"] == [1, 4, 8]
    offsets = {c["support_x0"] for c in out}
    assert 21 in offsets
    assert all(c["touch_indices"][0] == c["support_x0"] for c in out)
