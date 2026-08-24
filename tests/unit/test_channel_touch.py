"""Unit tests for live channel-touch trigger helpers."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd

from utils.scanning.channel_touch import format_triggers_message, live_entries_for_symbol


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
        rows = live_entries_for_symbol("AAA", df, atr_stop_mult=2.0)
    assert len(rows) == 1
    assert rows[0]["stock"] == "AAA"
    assert rows[0]["buy_date"] == df.index[-1].strftime("%Y-%m-%d")
    assert rows[0]["hard_stop"] < rows[0]["buy_price"]


def test_format_triggers_message_no_signal():
    msg = format_triggers_message(pd.DataFrame(), as_of="2026-08-25", n_candidates=0)
    assert "No new triggers" in msg
    assert "as_of=2026-08-25" in msg
