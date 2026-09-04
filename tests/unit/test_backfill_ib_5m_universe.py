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
    in_rth_yield_window,
    make_should_stop,
    needs_backfill,
    next_rth_yield_dt,
    parse_args,
    rewind_start,
)

NY = ZoneInfo("America/New_York")


def test_default_client_id_is_distinct():
    args = parse_args([])
    assert args.ib_client_id == DEFAULT_CLIENT_ID == 8823
    assert args.sleep == 1.0
    assert args.batch_days == 7
    assert args.allow_rth is False
    assert args.ib_port == 4001


def test_inventory_and_skip_if_job():
    args = parse_args(["--inventory", "--skip-if-job-running", "after_rth_ib_backfill"])
    assert args.inventory is True
    assert args.skip_if_job_running == "after_rth_ib_backfill"


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
