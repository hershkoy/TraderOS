"""Prefix-window planner for the 20-year IB daily backfill."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from utils.research.ib_daily_prefix import (
    duration_str,
    in_rth_yield_window,
    iter_prefix_chunks,
    next_rth_yield_dt,
)

NY = ZoneInfo("America/New_York")


def _utc(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


def test_twenty_year_gap_walks_backward_from_the_stored_edge():
    start = _utc(2006, 1, 1)
    end = _utc(2026, 1, 1)
    chunks = list(iter_prefix_chunks(start, end, chunk_years=10))
    assert len(chunks) == 3 or len(chunks) == 2
    assert chunks[0][1] == end
    assert chunks[-1][0] == start
    for i in range(len(chunks) - 1):
        assert chunks[i + 1][1] == chunks[i][0]
        assert chunks[i][0] < chunks[i][1]


def test_short_gap_is_one_chunk():
    start = _utc(2024, 1, 1)
    end = _utc(2025, 6, 1)
    assert list(iter_prefix_chunks(start, end, chunk_years=10)) == [(start, end)]


def test_empty_when_history_already_reaches_start():
    start = _utc(2006, 1, 1)
    end = _utc(2006, 1, 1)
    assert list(iter_prefix_chunks(start, end)) == []
    assert list(iter_prefix_chunks(start, _utc(2005, 1, 1))) == []


def test_duration_str_uses_years_past_one_year():
    assert duration_str(_utc(2006, 1, 1), _utc(2016, 1, 1)).endswith(" Y")
    assert duration_str(_utc(2026, 1, 1), _utc(2026, 4, 1)).endswith(" D")


def test_rth_yield_is_weekday_morning_through_cash_close():
    assert in_rth_yield_window(datetime(2026, 10, 26, 14, 0, tzinfo=timezone.utc))  # 10:00 ET Monday
    assert in_rth_yield_window(datetime(2026, 10, 26, 20, 0, tzinfo=timezone.utc))  # 16:00 ET still inside
    assert in_rth_yield_window(datetime(2026, 10, 26, 20, 30, tzinfo=timezone.utc)) is False  # 16:30 ET
    assert in_rth_yield_window(datetime(2026, 10, 24, 15, 0, tzinfo=timezone.utc)) is False  # Saturday


def test_next_stop_is_monday_after_friday_evening():
    friday_evening = datetime(2026, 10, 23, 21, 0, tzinfo=NY)  # Friday 21:00 ET
    nxt = next_rth_yield_dt(friday_evening)
    assert nxt.astimezone(NY).strftime("%Y-%m-%d %H:%M") == "2026-10-26 09:15"
