"""Unit tests for CronManager crontab IO and due-job selection."""
from datetime import datetime
from pathlib import Path

import os
import yaml

from utils.cron.manager import CronJob, CronManager


def _mgr(tmp_path: Path) -> CronManager:
    root = tmp_path
    (root / "crons").mkdir(exist_ok=True)
    return CronManager(root=root, crontab_path=root / "crons" / "crontab.yaml")


def test_add_list_enable_remove(tmp_path: Path):
    mgr = _mgr(tmp_path)
    job = CronJob(
        name="nightly",
        schedule=["0 23 * * 1-5"],
        command=r"crons\foo.bat",
        enabled=False,
        description="demo",
    )
    mgr.add_job(job)
    names = [j.name for j in mgr.jobs()]
    assert names == ["nightly"]
    loaded = mgr.job("nightly")
    assert loaded.command == r"crons\foo.bat"
    assert loaded.enabled is False

    mgr.set_enabled("nightly", True)
    assert mgr.job("nightly").enabled is True

    mgr.remove_job("nightly")
    assert mgr.jobs() == []


def test_due_jobs_respects_enabled_and_schedule(tmp_path: Path):
    mgr = _mgr(tmp_path)
    mgr.add_job(CronJob(name="hit", schedule=["0 23 * * 1-5"], command="echo hit", enabled=True))
    mgr.add_job(CronJob(name="off", schedule=["0 23 * * 1-5"], command="echo off", enabled=False))
    mgr.add_job(CronJob(name="other", schedule=["0 9 * * *"], command="echo other", enabled=True))

    monday_2300 = datetime(2026, 8, 31, 23, 0, 0)
    due = [j.name for j in mgr.due_jobs(now=monday_2300)]
    assert due == ["hit"]

    monday_0900 = datetime(2026, 8, 31, 9, 0, 0)
    due = [j.name for j in mgr.due_jobs(now=monday_0900)]
    assert due == ["other"]


def test_tick_skips_same_minute_and_lock(tmp_path: Path):
    mgr = _mgr(tmp_path)
    mgr.add_job(CronJob(name="once", schedule=["* * * * *"], command="echo once", enabled=True))
    when = datetime(2026, 8, 30, 15, 1, 0)

    started = mgr.tick(now=when, dry_run=True)
    assert started == ["once"]

    mgr._mark_fired("once", "2026-08-30T15:01")
    started = mgr.tick(now=when, dry_run=True)
    assert started == []

    mgr2 = _mgr(tmp_path)
    mgr2.add_job(
        CronJob(name="locked", schedule=["* * * * *"], command="echo locked", enabled=True),
        overwrite=True,
    )
    mgr2.state_dir.mkdir(parents=True, exist_ok=True)
    mgr2.lock_dir.mkdir(parents=True, exist_ok=True)
    mgr2._write_lock("locked", os.getpid())
    started = mgr2.tick(now=when, dry_run=True)
    assert "locked" not in started


def test_tick_skip_if_ran_today(tmp_path: Path):
    mgr = _mgr(tmp_path)
    mgr.add_job(
        CronJob(
            name="nightly",
            schedule=["0 23 * * 1-5"],
            command="echo nightly",
            enabled=True,
            skip_if_ran_today=True,
        )
    )
    monday_2300 = datetime(2026, 8, 31, 23, 0, 0)
    mgr._mark_fired("nightly", "2026-08-31T10:00")
    state = mgr._load_state()
    state["jobs"]["nightly"]["last_exit_code"] = 0
    mgr._save_state(state)
    assert mgr.tick(now=monday_2300, dry_run=True) == []

    state["jobs"]["nightly"]["last_exit_code"] = 1
    mgr._save_state(state)
    assert mgr.tick(now=monday_2300, dry_run=True) == ["nightly"]


def test_crontab_roundtrip_yaml(tmp_path: Path):
    mgr = _mgr(tmp_path)
    mgr.add_job(
        CronJob(
            name="channel_touch_15m",
            schedule=["45 9 * * 1-5", "0 16 * * 1-5"],
            command=r"crons\channel_touch_15m.bat",
            timezone="America/New_York",
        )
    )
    raw = yaml.safe_load(mgr.crontab_path.read_text(encoding="utf-8"))
    assert raw["jobs"][0]["schedule"] == ["45 9 * * 1-5", "0 16 * * 1-5"]
    assert raw["jobs"][0]["timezone"] == "America/New_York"


def test_repo_crontab_loads():
    mgr = CronManager()
    jobs = mgr.jobs()
    names = {j.name for j in jobs}
    assert "channel_touch_nightly" in names
    assert "after_rth_ib_backfill" in names
    assert "stop_ib_5m_backfill" in names
    assert "backfill_ib_5m_universe" in names
    nightly = mgr.job("channel_touch_nightly")
    assert nightly.schedule == ["0 23 * * 1-5"]
    assert nightly.skip_if_ran_today is True
    after = mgr.job("after_rth_ib_backfill")
    assert after.schedule == ["30 16 * * 1-5"]
    assert after.timezone == "America/New_York"
    assert after.enabled is True
    stop = mgr.job("stop_ib_5m_backfill")
    assert stop.schedule == ["15 9 * * 1-5"]
    assert stop.timezone == "America/New_York"
    assert mgr.job("backfill_ib_15m_universe").enabled is False


def test_stop_job_when_not_running(tmp_path: Path):
    mgr = _mgr(tmp_path)
    mgr.add_job(CronJob(name="idle", schedule=["* * * * *"], command="echo idle", enabled=True))
    assert mgr.stop_job("idle") == 0
