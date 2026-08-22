"""Unit tests for utils.data.ohlcv_loader cache helpers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.data.ohlcv_loader import _cache_path, _read_cache, _write_cache


def test_cache_path_layout(tmp_path: Path) -> None:
    p = _cache_path(
        tmp_path,
        "ALPACA",
        "1d",
        "msft",
        datetime(2018, 1, 1),
        datetime(2025, 11, 26),
    )
    assert p == tmp_path / "ALPACA" / "1d" / "MSFT_20180101_20251126.parquet"


def test_write_read_cache_roundtrip(tmp_path: Path) -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100.0, 200.0],
        },
        index=idx,
    )
    path = tmp_path / "MSFT.parquet"
    _write_cache(path, df)
    got = _read_cache(path)
    assert got is not None
    assert list(got.columns) == ["open", "high", "low", "close", "volume"]
    assert len(got) == 2
    assert float(got["close"].iloc[-1]) == 2.2


def test_read_missing_cache_returns_none(tmp_path: Path) -> None:
    assert _read_cache(tmp_path / "missing.parquet") is None
