"""Unit tests for live IB 15m gap window helpers (no Gateway)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from utils.scanning.channel_touch_15m_refresh import (
    LIVE_IB_CLIENT_ID,
    _fetch_window_start,
    gateway_tcp_open,
    last_ts_map_from_rows,
    rewind_start,
)


def test_live_client_id_is_not_backfill_ids():
    assert LIVE_IB_CLIENT_ID == 8823
    assert LIVE_IB_CLIENT_ID not in (8821, 8822)


def test_rewind_start_overlaps_bars():
    last = pd.Timestamp("2026-09-01T19:45:00Z")
    start = rewind_start(last, 2)
    assert start == datetime(2026, 9, 1, 19, 15, tzinfo=timezone.utc)


def test_fetch_window_caps_old_last_ts():
    now = datetime(2026, 9, 2, 18, 30, tzinfo=timezone.utc)
    old = pd.Timestamp("2025-12-02T20:45:00Z")
    start = _fetch_window_start(old, now=now, overlap_bars=2, max_days=5)
    assert start >= datetime(2026, 8, 28, 18, 30, tzinfo=timezone.utc)


def test_fetch_window_from_yesterday_is_not_capped():
    now = datetime(2026, 9, 2, 18, 30, tzinfo=timezone.utc)
    last = pd.Timestamp("2026-09-01T19:45:00Z")
    start = _fetch_window_start(last, now=now, overlap_bars=2, max_days=5)
    assert start == datetime(2026, 9, 1, 19, 15, tzinfo=timezone.utc)


def test_last_ts_map_from_rows_uses_15m_as_of():
    rows = [
        {"stock": "aaa", "timeframe": "15m", "as_of": "2026-09-01 19:45:00"},
        {"stock": "BBB", "timeframe": "1d", "as_of": "2026-08-31"},
        {"stock": "CCC", "timeframe": "15m", "as_of": None},
    ]
    got = last_ts_map_from_rows(rows, ["AAA", "BBB", "CCC"])
    assert list(got) == ["AAA"]
    assert str(got["AAA"]) == "2026-09-01 19:45:00+00:00"


def test_gateway_tcp_open_returns_bool():
    assert gateway_tcp_open(host="127.0.0.1", port=1, timeout=0.05) is False
