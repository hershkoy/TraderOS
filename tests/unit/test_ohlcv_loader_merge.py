"""Unit tests for Alpaca+IB prefix merge in ohlcv_loader."""
from __future__ import annotations

import pandas as pd

from utils.data.ohlcv_loader import _first_late, merge_provider_prefix


def _df(dates, closes):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


def test_merge_prefix_prepends_ib_before_alpaca():
    alp = _df(["2022-01-03", "2022-01-04"], [100.0, 101.0])
    ib = _df(["2021-12-01", "2021-12-02", "2022-01-03"], [50.0, 51.0, 99.0])
    merged, meta = merge_provider_prefix(alp, ib, max_jump_pct=15.0)
    assert meta["fallback_prefix_bars"] == 2
    assert merged.index.min() == pd.Timestamp("2021-12-01")
    # Alpaca owns overlapping 2022-01-03
    assert float(merged.loc[pd.Timestamp("2022-01-03"), "close"]) == 100.0


def test_merge_scales_large_jump():
    alp = _df(["2022-01-03"], [200.0])
    ib = _df(["2021-12-01"], [50.0])  # 300% jump -> scale
    merged, meta = merge_provider_prefix(alp, ib, max_jump_pct=15.0)
    assert meta["scaled"] is True
    assert abs(float(merged.iloc[0]["close"]) - 200.0) < 1e-6


def test_first_late():
    df = _df(["2022-06-01"], [10.0])
    assert _first_late(df, pd.Timestamp("2018-11-01").to_pydatetime()) is True
    assert _first_late(df, pd.Timestamp("2022-05-30").to_pydatetime()) is False
    assert _first_late(None, pd.Timestamp("2018-11-01").to_pydatetime()) is True
