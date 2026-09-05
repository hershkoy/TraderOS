"""Unit tests for IB 5m universe backfill helpers (no Gateway)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from backfill_ib_5m_universe import (  # noqa: E402
    DEFAULT_CLIENT_ID,
    DEFAULT_YEAR_FROM,
    DEFAULT_YEAR_TO,
    PidLock,
    in_rth_yield_window,
    is_transient_ib_error,
    make_should_stop,
    needs_backfill,
    next_rth_yield_dt,
    parse_args,
    parse_year_list,
    rewind_start,
    symbol_year_job,
    year_fetch_start,
    year_slice_bounds,
    year_window_complete,
)

NY = ZoneInfo("America/New_York")


def test_default_client_id_is_distinct():
    args = parse_args([])
    assert args.ib_client_id == DEFAULT_CLIENT_ID == 8823
    assert args.sleep == 1.0
    assert args.batch_days == 7
    assert args.allow_rth is False
    assert args.ib_port == 4001
    assert args.year_from == DEFAULT_YEAR_FROM == 2025
    assert args.year_to == DEFAULT_YEAR_TO == 2020
    assert args.no_year_slice is False
    assert args.no_through_now is False


def test_inventory_and_skip_if_job():
    args = parse_args(["--inventory", "--skip-if-job-running", "after_rth_ib_backfill"])
    assert args.inventory is True
    assert args.skip_if_job_running == "after_rth_ib_backfill"
    reset = parse_args(["--reset-failed", "--skip-if-job-running", "after_rth_ib_backfill"])
    assert reset.reset_failed is True


def test_transient_ib_error_not_qualify():
    assert is_transient_ib_error(OSError("API connection failed: ConnectionRefusedError"))
    assert is_transient_ib_error(
        ConnectionRefusedError(22, "The remote computer refused the network connection")
    )
    assert is_transient_ib_error(RuntimeError("Not connected"))
    assert is_transient_ib_error(TimeoutError("timed out"))
    assert not is_transient_ib_error(ValueError("qualify failed for AAPL"))
    assert not is_transient_ib_error(RuntimeError("TimescaleDB insert failed"))


def test_pid_lock_steals_recycled_pid(tmp_path: Path):
    import json
    import os

    lock_path = tmp_path / "ib_5m_universe.lock"
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started": "stale"}), encoding="utf-8")
    lock = PidLock(lock_path)
    assert lock.acquire() is True
    lock.release()
    assert not lock_path.exists()


def test_needs_backfill_missing_behind_and_fresh():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    last_15m = pd.Timestamp("2026-09-03T20:00:00Z")
    assert needs_backfill(None, last_15m, now=now) is True
    behind = pd.Timestamp("2019-06-01T14:30:00Z")
    assert needs_backfill(behind, last_15m, now=now, fresh_hours=36.0) is True
    caught = pd.Timestamp("2026-09-03T20:00:00Z")
    assert needs_backfill(caught, last_15m, now=now, fresh_hours=36.0) is False
    stale = pd.Timestamp("2026-09-01T20:00:00Z")
    assert needs_backfill(stale, stale, now=now, fresh_hours=36.0) is True


def test_rewind_start_overlaps_5m_bars():
    last = pd.Timestamp("2026-09-03T20:00:00Z")
    start = rewind_start(last, 2)
    assert start == datetime(2026, 9, 3, 19, 50, tzinfo=timezone.utc)


def test_rth_yield_window_weekdays_only():
    monday_open = datetime(2026, 9, 7, 9, 30, tzinfo=NY)
    monday_pre = datetime(2026, 9, 7, 9, 14, tzinfo=NY)
    monday_yield = datetime(2026, 9, 7, 9, 15, tzinfo=NY)
    monday_after = datetime(2026, 9, 7, 16, 30, tzinfo=NY)
    saturday = datetime(2026, 9, 5, 12, 0, tzinfo=NY)
    assert in_rth_yield_window(monday_open) is True
    assert in_rth_yield_window(monday_yield) is True
    assert in_rth_yield_window(monday_pre) is False
    assert in_rth_yield_window(monday_after) is False
    assert in_rth_yield_window(saturday) is False


def test_next_rth_yield_friday_evening_is_monday():
    friday_after = datetime(2026, 9, 4, 16, 45, tzinfo=NY)
    nxt = next_rth_yield_dt(friday_after)
    assert nxt.tzinfo is not None
    ny = nxt.astimezone(NY)
    assert ny.weekday() == 0
    assert ny.hour == 9 and ny.minute == 15
    monday_pre = datetime(2026, 9, 7, 9, 0, tzinfo=NY)
    same_morning = next_rth_yield_dt(monday_pre).astimezone(NY)
    assert same_morning.date() == monday_pre.date()
    monday_during = datetime(2026, 9, 7, 10, 0, tzinfo=NY)
    tuesday = next_rth_yield_dt(monday_during).astimezone(NY)
    assert tuesday.weekday() == 1


def test_stop_file_triggers_should_stop(tmp_path: Path):
    stop = tmp_path / "ib_5m.stop"
    flag = {"stop": False}
    until = datetime(2099, 1, 1, tzinfo=timezone.utc)
    check = make_should_stop(stop_file=stop, until=until, allow_rth=True, flag=flag)
    assert check() is False
    stop.write_text("stop\n", encoding="utf-8")
    assert check() is True
    flag["stop"] = True
    stop.unlink()
    check2 = make_should_stop(stop_file=stop, until=until, allow_rth=True, flag=flag)
    assert check2() is True


def test_parse_year_list_newest_first():
    assert parse_year_list("", 2025, 2020) == [2025, 2024, 2023, 2022, 2021, 2020]
    assert parse_year_list("2025,2024,2023", 2025, 2020) == [2025, 2024, 2023]
    try:
        parse_year_list("", 2020, 2025)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_year_slice_bounds_2025_includes_now():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    start, end = year_slice_bounds(2025, newest_year=2025, now=now, through_now=True)
    assert start == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert end == now
    start24, end24 = year_slice_bounds(2024, newest_year=2025, now=now, through_now=True)
    assert start24 == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert end24 == datetime(2025, 1, 1, tzinfo=timezone.utc)
    start25_cal, end25_cal = year_slice_bounds(
        2025, newest_year=2025, now=now, through_now=False
    )
    assert end25_cal == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_year_window_complete_and_fetch_start():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    ystart = datetime(2025, 1, 1, tzinfo=timezone.utc)
    yend = now
    first_15m = pd.Timestamp("2018-01-02T14:30:00Z")
    assert (
        year_window_complete(
            None,
            None,
            window_start=ystart,
            window_end=yend,
            first_15m=first_15m,
            last_15m=pd.Timestamp("2026-09-03T20:00:00Z"),
            now=now,
            through_now=True,
        )
        is False
    )
    caught_first = pd.Timestamp("2025-01-02T14:30:00Z")
    caught_last = pd.Timestamp("2026-09-03T20:00:00Z")
    assert (
        year_window_complete(
            caught_first,
            caught_last,
            window_start=ystart,
            window_end=yend,
            first_15m=first_15m,
            last_15m=caught_last,
            now=now,
            through_now=True,
        )
        is True
    )
    mid = pd.Timestamp("2025-06-15T14:30:00Z")
    start = year_fetch_start(
        mid,
        caught_first,
        window_start=ystart,
        first_15m=first_15m,
        overlap_bars=2,
    )
    assert start == rewind_start(mid, 2)
    gap_start = year_fetch_start(
        caught_last,
        pd.Timestamp("2025-06-15T14:30:00Z"),
        window_start=ystart,
        first_15m=first_15m,
        overlap_bars=2,
    )
    assert gap_start == ystart


def test_symbol_year_jobs_finish_2025_before_2024():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    first_15m = pd.Timestamp("2018-01-02T14:30:00Z")
    last_15m = pd.Timestamp("2026-09-03T20:00:00Z")
    symbols = ["AAA", "BBB", "CCC"]
    jobs = []
    for year in (2025, 2024):
        ystart, yend = year_slice_bounds(
            year, newest_year=2025, now=now, through_now=True
        )
        for sym in symbols:
            job = symbol_year_job(
                sym,
                first_15m,
                last_15m,
                None,
                None,
                year=year,
                window_start=ystart,
                window_end=yend,
                through_now=(year == 2025),
                now=now,
                overlap_bars=2,
                fresh_hours=36.0,
            )
            assert job is not None
            jobs.append(job)
    years_order = [j[1] for j in jobs]
    assert years_order == [2025, 2025, 2025, 2024, 2024, 2024]
    assert [j[0] for j in jobs[:3]] == symbols
    assert jobs[0][2] == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert jobs[0][3] == now
    assert jobs[3][2] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert jobs[3][3] == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_symbol_year_job_skips_complete_and_pre_listing():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    ystart, yend = year_slice_bounds(2024, newest_year=2025, now=now, through_now=True)
    ipo = pd.Timestamp("2025-03-01T14:30:00Z")
    assert (
        symbol_year_job(
            "NEW",
            ipo,
            pd.Timestamp("2026-09-03T20:00:00Z"),
            None,
            None,
            year=2024,
            window_start=ystart,
            window_end=yend,
            through_now=False,
            now=now,
            overlap_bars=2,
            fresh_hours=36.0,
        )
        is None
    )
    complete = symbol_year_job(
        "OLD",
        pd.Timestamp("2018-01-02T14:30:00Z"),
        pd.Timestamp("2026-09-03T20:00:00Z"),
        pd.Timestamp("2024-01-02T14:30:00Z"),
        pd.Timestamp("2024-12-31T21:00:00Z"),
        year=2024,
        window_start=ystart,
        window_end=yend,
        through_now=False,
        now=now,
        overlap_bars=2,
        fresh_hours=36.0,
    )
    assert complete is None
