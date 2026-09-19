"""Feed freshness status for the /hot API table (no live Alpaca/IB)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from utils.scanning.channel_touch_feed_status import (
    JOB_ALPACA_1D,
    JOB_ALPACA_LAST,
    JOB_AFTER_RTH,
    JOB_IB_15M,
    JOB_IB_5M,
    JOB_IB_UNIVERSE,
    STATUS_FAILED,
    STATUS_MISSING,
    STATUS_STALE,
    STATUS_UP_TO_DATE,
    STATUS_UPDATING,
    STATUS_WAITING,
    as_of_behind_last_close,
    build_feeds,
    in_5m_yield_window,
    last_completed_rth_15m,
    last_completed_rth_5m,
    last_completed_session_date,
)

ET = ZoneInfo("America/New_York")


def _job(name, *, running=False, exit_code=0, last_fired=None, next_run=None):
    return {
        "name": name,
        "enabled": True,
        "running": running,
        "last_fired": last_fired,
        "last_finished": None,
        "last_exit_code": exit_code,
        "next_run": next_run,
    }


def _jobs(**overrides):
    base = {
        JOB_ALPACA_LAST: _job(JOB_ALPACA_LAST, last_fired="2026-09-02T15:08"),
        JOB_IB_15M: _job(JOB_IB_15M, last_fired="2026-09-02T15:00", exit_code=1),
        JOB_IB_UNIVERSE: _job(JOB_IB_UNIVERSE, last_fired="2026-09-02T02:30", exit_code=0),
        JOB_AFTER_RTH: _job(JOB_AFTER_RTH, last_fired="2026-09-01T16:30", exit_code=0),
        JOB_IB_5M: _job(JOB_IB_5M, last_fired="2026-09-02T03:00", exit_code=0),
        JOB_ALPACA_1D: _job(
            JOB_ALPACA_1D,
            last_fired="2026-09-01T17:00",
            next_run="2026-09-02T17:00-04:00",
        ),
    }
    base.update(overrides)
    return base


def _feeds(**kwargs):
    kwargs.setdefault("five_m", {})
    kwargs.setdefault("jobs", _jobs())
    return build_feeds(**kwargs)


def test_last_completed_rth_15m_slots():
    wed_1508 = datetime(2026, 9, 2, 15, 8, tzinfo=ET)
    assert last_completed_rth_15m(wed_1508) == datetime(2026, 9, 2, 14, 45, tzinfo=ET)
    wed_1500 = datetime(2026, 9, 2, 15, 0, tzinfo=ET)
    assert last_completed_rth_15m(wed_1500) == datetime(2026, 9, 2, 14, 45, tzinfo=ET)
    wed_1600 = datetime(2026, 9, 2, 16, 0, tzinfo=ET)
    assert last_completed_rth_15m(wed_1600) == datetime(2026, 9, 2, 15, 45, tzinfo=ET)
    wed_0944 = datetime(2026, 9, 2, 9, 44, tzinfo=ET)
    assert last_completed_rth_15m(wed_0944) == datetime(2026, 9, 1, 15, 45, tzinfo=ET)
    wed_0945 = datetime(2026, 9, 2, 9, 45, tzinfo=ET)
    assert last_completed_rth_15m(wed_0945) == datetime(2026, 9, 2, 9, 30, tzinfo=ET)
    saturday = datetime(2026, 9, 5, 12, 0, tzinfo=ET)
    assert last_completed_rth_15m(saturday) == datetime(2026, 9, 4, 15, 45, tzinfo=ET)


def test_last_completed_session_date_before_and_after_close():
    wed_1508 = datetime(2026, 9, 2, 15, 8, tzinfo=ET)
    assert last_completed_session_date(wed_1508).isoformat() == "2026-09-01"
    wed_close = datetime(2026, 9, 2, 16, 0, tzinfo=ET)
    assert last_completed_session_date(wed_close).isoformat() == "2026-09-02"
    saturday = datetime(2026, 9, 5, 12, 0, tzinfo=ET)
    assert last_completed_session_date(saturday).isoformat() == "2026-09-04"


def test_as_of_behind_last_close_friday_session_on_saturday():
    saturday = datetime(2026, 9, 19, 3, 9, tzinfo=ET)
    behind, got, expected = as_of_behind_last_close("2026-09-17", now=saturday)
    assert behind is True
    assert got.isoformat() == "2026-09-17"
    assert expected.isoformat() == "2026-09-18"
    behind, got, expected = as_of_behind_last_close("2026-09-18", now=saturday)
    assert behind is False
    assert expected.isoformat() == "2026-09-18"


def test_feeds_match_hot_dashboard_staleness():
    """Wed 15:08 ET: Alpaca live, IB 15m stuck at Tue 15:45, 1d still Mon."""
    now = datetime(2026, 9, 2, 15, 8, 51, tzinfo=ET)
    feeds = _feeds(
        price_ts="2026-09-02 19:08:51",
        as_of_15m="2026-09-01 19:45:00",
        as_of_1d="2026-08-31",
        now=now,
    )
    by_id = {row["id"]: row for row in feeds}
    assert by_id["alpaca_last"]["status"] == STATUS_UP_TO_DATE
    assert by_id["ib_15m"]["status"] == STATUS_FAILED
    assert by_id["ib_15m"]["name"] == "IB 15m hot list"
    assert "channel_touch_15m" in by_id["ib_15m"]["detail"]
    assert "14:45" in by_id["ib_15m"]["detail"]
    assert by_id["ib_15m"]["expected_as_of"] == "2026-09-02 18:45:00"
    assert by_id["ib_15m_universe"]["status"] == STATUS_WAITING
    assert "02:30" in by_id["ib_15m_universe"]["cadence"]
    assert by_id["alpaca_1d"]["status"] == STATUS_STALE
    assert by_id["alpaca_1d"]["expected_as_of"] == "2026-09-01"
    assert "EOD" in by_id["alpaca_1d"]["role"] or "EOD" in by_id["alpaca_1d"]["detail"]
    assert "17:00 ET" in by_id["alpaca_1d"]["detail"]
    assert by_id["ib_5m"]["name"] == "IB 5m universe"
    assert by_id["ib_5m"]["status"] == STATUS_WAITING
    assert "16:30 ET" in by_id["ib_5m"]["cadence"]
    assert "Yielded" in by_id["ib_5m"]["detail"]


def test_ib_15m_updating_when_job_running():
    now = datetime(2026, 9, 2, 15, 8, tzinfo=ET)
    feeds = _feeds(
        price_ts="2026-09-02 19:08:00",
        as_of_15m="2026-09-01 19:45:00",
        as_of_1d="2026-09-01",
        now=now,
        jobs=_jobs(**{JOB_IB_15M: _job(JOB_IB_15M, running=True, exit_code=1)}),
    )
    assert feeds[1]["status"] == STATUS_UPDATING


def test_1d_waiting_when_caught_up_during_rth():
    now = datetime(2026, 9, 2, 15, 8, tzinfo=ET)
    feeds = _feeds(
        price_ts="2026-09-02 19:08:00",
        as_of_15m="2026-09-02 18:45:00",
        as_of_1d="2026-09-01",
        now=now,
        jobs=_jobs(**{JOB_IB_15M: _job(JOB_IB_15M, exit_code=0)}),
    )
    by_id = {row["id"]: row for row in feeds}
    assert by_id["ib_15m"]["status"] == STATUS_UP_TO_DATE
    assert by_id["alpaca_1d"]["status"] == STATUS_WAITING
    assert "17:00 ET" in by_id["alpaca_1d"]["detail"]


def test_alpaca_last_waiting_after_rth():
    now = datetime(2026, 9, 2, 17, 0, tzinfo=ET)
    feeds = _feeds(
        price_ts="2026-09-02 20:00:00",
        as_of_15m="2026-09-02 19:45:00",
        as_of_1d="2026-09-02",
        now=now,
        jobs=_jobs(**{JOB_IB_15M: _job(JOB_IB_15M, exit_code=0)}),
    )
    by_id = {row["id"]: row for row in feeds}
    assert by_id["alpaca_last"]["status"] == STATUS_WAITING
    assert by_id["alpaca_1d"]["status"] == STATUS_UP_TO_DATE


def test_last_completed_rth_5m_slots():
    wed_1508 = datetime(2026, 9, 2, 15, 8, tzinfo=ET)
    assert last_completed_rth_5m(wed_1508) == datetime(2026, 9, 2, 15, 0, tzinfo=ET)
    wed_1600 = datetime(2026, 9, 2, 16, 0, tzinfo=ET)
    assert last_completed_rth_5m(wed_1600) == datetime(2026, 9, 2, 15, 55, tzinfo=ET)
    wed_0935 = datetime(2026, 9, 2, 9, 35, tzinfo=ET)
    assert last_completed_rth_5m(wed_0935) == datetime(2026, 9, 2, 9, 30, tzinfo=ET)
    wed_0934 = datetime(2026, 9, 2, 9, 34, tzinfo=ET)
    assert last_completed_rth_5m(wed_0934) == datetime(2026, 9, 1, 15, 55, tzinfo=ET)
    saturday = datetime(2026, 9, 5, 12, 0, tzinfo=ET)
    assert last_completed_rth_5m(saturday) == datetime(2026, 9, 4, 15, 55, tzinfo=ET)


def test_5m_yield_window_weekdays_only():
    monday_open = datetime(2026, 9, 7, 9, 30, tzinfo=ET)
    monday_pre = datetime(2026, 9, 7, 9, 14, tzinfo=ET)
    monday_yield = datetime(2026, 9, 7, 9, 15, tzinfo=ET)
    monday_after = datetime(2026, 9, 7, 16, 30, tzinfo=ET)
    saturday = datetime(2026, 9, 5, 12, 0, tzinfo=ET)
    assert in_5m_yield_window(monday_open) is True
    assert in_5m_yield_window(monday_yield) is True
    assert in_5m_yield_window(monday_pre) is False
    assert in_5m_yield_window(monday_after) is False
    assert in_5m_yield_window(saturday) is False


def _five_m_progress():
    return {
        "symbol": "RYTM",
        "year": 2024,
        "n_bars_this_symbol": 18648,
        "window_first": "2024-12-31 18:00:00+00:00",
        "window_last": "2024-12-31 20:55:00+00:00",
        "queue_i": 2410,
        "queue_n": 6905,
        "updated": "2026-09-19T07:08:08+00:00",
    }


def test_ib_5m_updating_when_lock_held_on_weekend():
    now = datetime(2026, 9, 5, 10, 0, tzinfo=ET)
    feeds = _feeds(
        price_ts="2026-09-04 20:00:00",
        as_of_15m="2026-09-04 19:45:00",
        as_of_1d="2026-09-04",
        now=now,
        five_m={
            "progress": _five_m_progress(),
            "lock_held": True,
            "coverage": {"n": 472, "last_max": "2026-09-17 19:55:00+00:00"},
            "n_failed": 3,
        },
    )
    row = {r["id"]: r for r in feeds}["ib_5m"]
    assert row["status"] == STATUS_UPDATING
    assert row["as_of"] == "2026-09-17 19:55:00+00:00"
    assert row["expected_as_of"] == "2026-09-04 19:55:00"
    assert "RYTM" in row["detail"]
    assert "2410/6905" in row["detail"]
    assert "failed skip 3" in row["detail"]
    assert row["job_running"] is True


def test_ib_5m_updating_during_after_rth_15m_phase():
    now = datetime(2026, 9, 4, 16, 40, tzinfo=ET)
    feeds = _feeds(
        now=now,
        jobs=_jobs(**{JOB_AFTER_RTH: _job(JOB_AFTER_RTH, running=True, exit_code=0)}),
        five_m={"progress": {}, "lock_held": False, "coverage": {}, "n_failed": 0},
    )
    row = {r["id"]: r for r in feeds}["ib_5m"]
    assert row["status"] == STATUS_UPDATING
    assert "8822" in row["detail"]
    assert row["job"] == JOB_AFTER_RTH


def test_ib_5m_failed_if_lock_held_during_yield():
    now = datetime(2026, 9, 7, 10, 0, tzinfo=ET)
    feeds = _feeds(
        now=now,
        five_m={
            "progress": _five_m_progress(),
            "lock_held": True,
            "coverage": {"n": 472, "last_max": "2026-09-17 19:55:00+00:00"},
            "n_failed": 0,
        },
    )
    row = {r["id"]: r for r in feeds}["ib_5m"]
    assert row["status"] == STATUS_FAILED
    assert "yield" in row["detail"]


def test_ib_5m_behind_when_idle_with_unfinished_queue():
    now = datetime(2026, 9, 5, 10, 0, tzinfo=ET)
    feeds = _feeds(
        now=now,
        five_m={
            "progress": _five_m_progress(),
            "lock_held": False,
            "coverage": {"n": 472, "last_max": "2026-09-17 19:55:00+00:00"},
            "n_failed": 0,
        },
    )
    row = {r["id"]: r for r in feeds}["ib_5m"]
    assert row["status"] == STATUS_STALE
    assert "incomplete" in row["detail"]


def test_ib_5m_missing_outside_yield_with_empty_snapshot():
    now = datetime(2026, 9, 5, 10, 0, tzinfo=ET)
    feeds = _feeds(now=now)
    row = {r["id"]: r for r in feeds}["ib_5m"]
    assert row["status"] == STATUS_MISSING
    assert row["expected_as_of"] == "2026-09-04 19:55:00"
