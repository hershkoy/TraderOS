"""Feed freshness status for the /hot API table (no live Alpaca/IB)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from utils.scanning.channel_touch_feed_status import (
    JOB_ALPACA_1D,
    JOB_ALPACA_LAST,
    JOB_IB_15M,
    STATUS_FAILED,
    STATUS_STALE,
    STATUS_UP_TO_DATE,
    STATUS_UPDATING,
    STATUS_WAITING,
    build_feeds,
    last_completed_rth_15m,
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
        JOB_ALPACA_1D: _job(
            JOB_ALPACA_1D,
            last_fired="2026-09-01T23:00",
            next_run="2026-09-02T23:00+03:00",
        ),
    }
    base.update(overrides)
    return base


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


def test_feeds_match_hot_dashboard_staleness():
    """Wed 15:08 ET: Alpaca live, IB 15m stuck at Tue 15:45, 1d still Mon."""
    now = datetime(2026, 9, 2, 15, 8, 51, tzinfo=ET)
    feeds = build_feeds(
        price_ts="2026-09-02 19:08:51",
        as_of_15m="2026-09-01 19:45:00",
        as_of_1d="2026-08-31",
        now=now,
        jobs=_jobs(),
    )
    by_id = {row["id"]: row for row in feeds}
    assert by_id["alpaca_last"]["status"] == STATUS_UP_TO_DATE
    assert by_id["ib_15m"]["status"] == STATUS_FAILED
    assert "14:45" in by_id["ib_15m"]["detail"]
    assert by_id["ib_15m"]["expected_as_of"] == "2026-09-02 18:45:00"
    assert by_id["alpaca_1d"]["status"] == STATUS_STALE
    assert by_id["alpaca_1d"]["expected_as_of"] == "2026-09-01"
    assert "EOD" in by_id["alpaca_1d"]["role"] or "EOD" in by_id["alpaca_1d"]["detail"]
    assert "16:00 ET" in by_id["alpaca_1d"]["detail"]


def test_ib_15m_updating_when_job_running():
    now = datetime(2026, 9, 2, 15, 8, tzinfo=ET)
    feeds = build_feeds(
        price_ts="2026-09-02 19:08:00",
        as_of_15m="2026-09-01 19:45:00",
        as_of_1d="2026-09-01",
        now=now,
        jobs=_jobs(**{JOB_IB_15M: _job(JOB_IB_15M, running=True, exit_code=1)}),
    )
    assert feeds[1]["status"] == STATUS_UPDATING


def test_1d_waiting_when_caught_up_during_rth():
    now = datetime(2026, 9, 2, 15, 8, tzinfo=ET)
    feeds = build_feeds(
        price_ts="2026-09-02 19:08:00",
        as_of_15m="2026-09-02 18:45:00",
        as_of_1d="2026-09-01",
        now=now,
        jobs=_jobs(**{JOB_IB_15M: _job(JOB_IB_15M, exit_code=0)}),
    )
    by_id = {row["id"]: row for row in feeds}
    assert by_id["ib_15m"]["status"] == STATUS_UP_TO_DATE
    assert by_id["alpaca_1d"]["status"] == STATUS_WAITING
    assert "23:00" in by_id["alpaca_1d"]["detail"]


def test_alpaca_last_waiting_after_rth():
    now = datetime(2026, 9, 2, 17, 0, tzinfo=ET)
    feeds = build_feeds(
        price_ts="2026-09-02 20:00:00",
        as_of_15m="2026-09-02 19:45:00",
        as_of_1d="2026-09-02",
        now=now,
        jobs=_jobs(**{JOB_IB_15M: _job(JOB_IB_15M, exit_code=0)}),
    )
    by_id = {row["id"]: row for row in feeds}
    assert by_id["alpaca_last"]["status"] == STATUS_WAITING
    assert by_id["alpaca_1d"]["status"] == STATUS_UP_TO_DATE
