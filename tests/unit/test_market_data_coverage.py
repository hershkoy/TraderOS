"""Unit tests for index-friendly IB coverage (no full-table GROUP BY)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.db.market_data_coverage import (
    cache_path_for,
    coverage_frame,
    distinct_recent_symbols_sql,
    edge_ts_sql,
    fetch_edge_ts,
    iter_ib_symbols,
    load_first_ts_cache,
    load_ib_coverage,
    load_ib_coverage_range,
    max_ts_sql,
    range_edge_sql,
    save_coverage_cache,
)


def test_sql_is_limit_one_not_group_by():
    assert "MAX(ts)" in max_ts_sql()
    assert "GROUP BY" not in max_ts_sql()
    distinct = distinct_recent_symbols_sql()
    assert "DISTINCT symbol" in distinct
    assert "ts >= %s" in distinct
    newest = edge_ts_sql(True)
    oldest = edge_ts_sql(False)
    assert "ORDER BY ts DESC LIMIT 1" in newest
    assert "ORDER BY ts ASC LIMIT 1" in oldest
    assert "COUNT(" not in newest
    ranged = range_edge_sql()
    assert "COUNT(" not in ranged
    assert "symbol = %s" in ranged


def test_iter_ib_symbols_uses_recent_distinct():
    def execute(sql, params):
        if "MAX(ts)" in sql:
            return [(datetime(2026, 9, 4, 19, 55, tzinfo=timezone.utc),)]
        if "DISTINCT symbol" in sql:
            assert params[2] < datetime(2026, 9, 4, 19, 55, tzinfo=timezone.utc)
            return [("AAPL",), ("MSFT",)]
        raise AssertionError(sql)

    assert iter_ib_symbols(execute, "15m") == ["AAPL", "MSFT"]


def test_fetch_edge_ts_localizes_naive():
    def execute(_sql, params):
        assert params[0] == "AAPL"
        return [(datetime(2026, 9, 3, 19, 55),)]

    ts = fetch_edge_ts(execute, "aapl", "15m", newest=True)
    assert ts.tzinfo is not None
    assert str(ts.tz).lower() == "utc"
    assert ts.year == 2026


def test_first_ts_cache_roundtrip(tmp_path: Path):
    df = coverage_frame(
        [
            {
                "symbol": "aapl",
                "first_ts": pd.Timestamp("2018-01-02T14:30:00Z"),
                "last_ts": pd.Timestamp("2026-09-03T19:55:00Z"),
                "n_bars": None,
            }
        ]
    )
    path = cache_path_for("15m", tmp_path)
    save_coverage_cache(path, "15m", df)
    cached = load_first_ts_cache(path)
    assert "AAPL" in cached
    assert "2018-01-02" in cached["AAPL"]


def test_load_ib_coverage_uses_cached_first_ts(tmp_path: Path):
    cached_first = "2018-01-02 14:30:00+00:00"
    path = cache_path_for("15m", tmp_path)
    path.write_text(
        '{"rows": {"AAPL": {"first_ts": "%s", "last_ts": "old"}}}' % cached_first,
        encoding="utf-8",
    )
    seen = []

    def execute(sql, params):
        seen.append(sql)
        if "MAX(ts)" in sql:
            return [(datetime(2026, 9, 4, 19, 55, tzinfo=timezone.utc),)]
        if "DISTINCT symbol" in sql:
            return [("AAPL",)]
        if "ORDER BY ts DESC LIMIT 1" in sql:
            return [(datetime(2026, 9, 4, 19, 55, tzinfo=timezone.utc),)]
        if "ORDER BY ts ASC LIMIT 1" in sql:
            raise AssertionError("must not refetch cached first_ts")
        raise AssertionError(sql)

    df = load_ib_coverage("15m", cache_dir=tmp_path, execute=execute)
    assert list(df["symbol"]) == ["AAPL"]
    assert not any("GROUP BY" in s for s in seen)
    last = pd.Timestamp(df.iloc[0]["last_ts"])
    assert last.day == 4


def test_load_ib_coverage_range_skips_empty_symbols():
    def execute(sql, params):
        if "MIN(ts)" in sql:
            if params[0] == "AAA":
                return [(None, None)]
            return [
                (
                    datetime(2025, 1, 2, tzinfo=timezone.utc),
                    datetime(2025, 6, 1, tzinfo=timezone.utc),
                )
            ]
        raise AssertionError(sql)

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df = load_ib_coverage_range(
        "5m", start, end, symbols=["AAA", "BBB"], execute=execute
    )
    assert list(df["symbol"]) == ["BBB"]
    assert "COUNT(" not in range_edge_sql()
