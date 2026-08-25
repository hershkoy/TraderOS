"""Unit tests for channel-touch quality post-filters."""
from __future__ import annotations

import pandas as pd

from backtest_channel_touch_trades import filter_trades


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock": "AAA",
                "channel_start": "2025-01-01",
                "channel_end": "2025-06-01",
                "buy_date": "2025-06-15",
                "channel_pos": 0.3,
                "channel_width_pct": 10.0,
                "slope_pct_per_bar": 0.05,
                "adv_20": 1e7,
                "atr_pct": 2.0,
            },
            {
                "stock": "BBB",
                "channel_start": "2023-01-01",
                "channel_end": "2025-06-01",
                "buy_date": "2025-06-15",
                "channel_pos": 1.5,
                "channel_width_pct": 10.0,
                "slope_pct_per_bar": 0.05,
                "adv_20": 1e7,
                "atr_pct": 2.0,
            },
            {
                "stock": "CCC",
                "channel_start": "2024-01-01",
                "channel_end": "2024-06-01",
                "buy_date": "2025-06-15",
                "channel_pos": 0.8,
                "channel_width_pct": 10.0,
                "slope_pct_per_bar": 0.05,
                "adv_20": 1e7,
                "atr_pct": 2.0,
            },
        ]
    )


def test_require_in_channel_drops_above_resist():
    out = filter_trades(_sample(), require_in_channel=True)
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_max_channel_span_days():
    out = filter_trades(_sample(), max_channel_span_days=400)
    # AAA ~151d, BBB ~882d, CCC ~152d
    assert set(out["stock"]) == {"AAA", "CCC"}


def test_max_channel_age_days():
    out = filter_trades(_sample(), max_channel_age_days=400)
    # AAA age ~165, BBB ~896, CCC ~531
    assert set(out["stock"]) == {"AAA"}


def test_combined_in_channel_and_span():
    out = filter_trades(_sample(), require_in_channel=True, max_channel_span_days=400)
    assert set(out["stock"]) == {"AAA", "CCC"}
