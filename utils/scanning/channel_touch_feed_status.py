"""Expected freshness and status for /hot data feeds (Alpaca last, IB 15m, Alpaca 1d)."""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
PRICE_FRESH_SEC = 120.0

JOB_ALPACA_LAST = "channel_touch_15m_proximity"
JOB_IB_15M = "channel_touch_15m"
JOB_IB_UNIVERSE = "backfill_ib_15m_universe"
JOB_ALPACA_1D = "channel_touch_nightly"

STATUS_UP_TO_DATE = "up_to_date"
STATUS_UPDATING = "updating"
STATUS_WAITING = "waiting"
STATUS_STALE = "stale"
STATUS_FAILED = "failed"
STATUS_MISSING = "missing"

STATUS_LABELS = {
    STATUS_UP_TO_DATE: "Up to date",
    STATUS_UPDATING: "Updating",
    STATUS_WAITING: "Waiting",
    STATUS_STALE: "Behind",
    STATUS_FAILED: "Failed",
    STATUS_MISSING: "Missing",
}


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        ts = value
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    text = str(value).strip().replace("T", " ")
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            if len(text) == 10:
                ts = datetime.strptime(text[:10], "%Y-%m-%d")
            else:
                ts = datetime.fromisoformat(text[:19])
        except ValueError:
            return None
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    return None


def _to_et(now: datetime) -> datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ET)


def previous_weekday(day: date) -> date:
    out = day - timedelta(days=1)
    while out.weekday() >= 5:
        out -= timedelta(days=1)
    return out


def is_rth(now: datetime) -> bool:
    now_et = _to_et(now)
    if now_et.weekday() >= 5:
        return False
    return RTH_OPEN <= now_et.time() < RTH_CLOSE


def last_completed_session_date(now: datetime) -> date:
    """Last US cash session whose daily bar should already exist in Alpaca.

    Before 16:00 ET on a weekday this is the previous weekday. At/after the
    close it is today (the bar may still be settling at Alpaca).
    """
    now_et = _to_et(now)
    day = now_et.date()
    if now_et.weekday() >= 5:
        return previous_weekday(day)
    if now_et.time() < RTH_CLOSE:
        return previous_weekday(day)
    return day


def last_completed_rth_15m(now: datetime) -> datetime:
    """Period-start of the last fully closed US RTH 15m bar (ET-aware).

    IB labels 15m bars at period start. The 14:45 bar covers 14:45-15:00 and
    is complete at 15:00.
    """
    now_et = _to_et(now)
    closed = now_et - timedelta(minutes=15)
    minute = (closed.minute // 15) * 15
    candidate = closed.replace(minute=minute, second=0, microsecond=0)
    session = candidate.date()
    if candidate.weekday() >= 5:
        prev = previous_weekday(session)
        return datetime(prev.year, prev.month, prev.day, 15, 45, tzinfo=ET)
    open_bar = datetime(session.year, session.month, session.day, 9, 30, tzinfo=ET)
    last_bar = datetime(session.year, session.month, session.day, 15, 45, tzinfo=ET)
    if candidate < open_bar:
        prev = previous_weekday(session)
        return datetime(prev.year, prev.month, prev.day, 15, 45, tzinfo=ET)
    if candidate > last_bar:
        return last_bar
    return candidate


def _fmt_utc_naive(ts: datetime) -> str:
    utc = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return utc.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_date(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def _as_of_date(value: Any) -> Optional[date]:
    """Calendar date for daily as_of. Date-only strings are not UTC midnights."""
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    parsed = _parse_ts(value)
    if parsed is None:
        return None
    return parsed.astimezone(ET).date()


def cron_job_snapshot(name: str, *, manager: Any = None) -> Dict[str, Any]:
    """Last run / next run for a CronRunner job. Never raises."""
    out: Dict[str, Any] = {
        "name": name,
        "enabled": None,
        "running": False,
        "last_fired": None,
        "last_finished": None,
        "last_exit_code": None,
        "next_run": None,
    }
    mgr = manager
    if mgr is None:
        try:
            from utils.cron.manager import CronManager

            mgr = CronManager()
        except Exception:
            logger.debug("CronManager unavailable for feed status", exc_info=True)
            return out
    try:
        job = mgr.job(name)
        out["enabled"] = bool(getattr(job, "enabled", True))
    except Exception:
        job = None
    try:
        st = mgr.job_state(name) or {}
        out["running"] = bool(st.get("running"))
        out["last_fired"] = st.get("last_fired")
        out["last_finished"] = st.get("last_finished")
        out["last_exit_code"] = st.get("last_exit_code")
    except Exception:
        logger.debug("job_state failed for %s", name, exc_info=True)
    if job is not None:
        try:
            nxt = mgr.next_run(job)
            if isinstance(nxt, datetime):
                out["next_run"] = nxt.isoformat(timespec="minutes")
        except Exception:
            pass
    return out


def _job_failed(job: Optional[dict]) -> bool:
    if not job:
        return False
    rc = job.get("last_exit_code")
    if rc is None:
        return False
    try:
        return int(rc) != 0
    except (TypeError, ValueError):
        return False


def _feed(
    *,
    feed_id: str,
    name: str,
    role: str,
    cadence: str,
    as_of: Any,
    expected_as_of: Optional[str],
    status: str,
    detail: str,
    job: Optional[dict] = None,
) -> Dict[str, Any]:
    job = job or {}
    return {
        "id": feed_id,
        "name": name,
        "role": role,
        "cadence": cadence,
        "as_of": as_of,
        "expected_as_of": expected_as_of,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "detail": detail,
        "job": job.get("name"),
        "job_enabled": job.get("enabled"),
        "job_running": bool(job.get("running")),
        "job_last_fired": job.get("last_fired"),
        "job_last_finished": job.get("last_finished"),
        "job_last_exit": job.get("last_exit_code"),
        "job_next_run": job.get("next_run"),
    }


def _alpaca_last_feed(
    price_ts: Any,
    *,
    now: datetime,
    job: dict,
) -> Dict[str, Any]:
    role = "Live last price (proximity / hot). Not a fill."
    cadence = "Every RTH minute; dashboard refreshes ~5s while /hot is open"
    parsed = _parse_ts(price_ts)
    if job.get("running"):
        return _feed(
            feed_id="alpaca_last",
            name="Alpaca last",
            role=role,
            cadence=cadence,
            as_of=price_ts,
            expected_as_of=None,
            status=STATUS_UPDATING,
            detail="Proximity job is fetching last prices.",
            job=job,
        )
    if parsed is None:
        return _feed(
            feed_id="alpaca_last",
            name="Alpaca last",
            role=role,
            cadence=cadence,
            as_of=price_ts,
            expected_as_of=None,
            status=STATUS_MISSING,
            detail="No last-price timestamp yet.",
            job=job,
        )
    age = (now - parsed).total_seconds()
    if is_rth(now):
        if age <= PRICE_FRESH_SEC:
            status, detail = STATUS_UP_TO_DATE, "Quotes are current."
        else:
            status, detail = STATUS_STALE, "Quotes are older than 2 minutes during RTH."
    else:
        status, detail = (
            STATUS_WAITING,
            "RTH closed. Last quotes stay until the next session.",
        )
    return _feed(
        feed_id="alpaca_last",
        name="Alpaca last",
        role=role,
        cadence=cadence,
        as_of=price_ts,
        expected_as_of=None,
        status=status,
        detail=detail,
        job=job,
    )


def _ib_15m_feed(
    as_of: Any,
    *,
    now: datetime,
    job: dict,
) -> Dict[str, Any]:
    role = "LIVE IB hist on the armed/hot list only (job channel_touch_15m, client 8826). Not overnight universe backfill."
    cadence = "Every US RTH 15m close"
    expected = last_completed_rth_15m(now)
    expected_s = _fmt_utc_naive(expected)
    parsed = _parse_ts(as_of)
    if job.get("running"):
        return _feed(
            feed_id="ib_15m",
            name="IB 15m hot list",
            role=role,
            cadence=cadence,
            as_of=as_of,
            expected_as_of=expected_s,
            status=STATUS_UPDATING,
            detail="Bar-close job is pulling IB hist on the armed/hot list.",
            job=job,
        )
    if parsed is None:
        return _feed(
            feed_id="ib_15m",
            name="IB 15m hot list",
            role=role,
            cadence=cadence,
            as_of=as_of,
            expected_as_of=expected_s,
            status=STATUS_MISSING,
            detail="No IB 15m as_of. Fills cannot be evaluated.",
            job=job,
        )
    got_et = parsed.astimezone(ET).replace(second=0, microsecond=0)
    exp_et = expected.replace(second=0, microsecond=0)
    behind = got_et < exp_et
    if behind and _job_failed(job):
        detail = (
            "Expected last closed bar %s. Job channel_touch_15m last exit %s "
            "(as_of did not advance). This job fires every RTH 15m close; "
            "overnight backfill_ib_15m_universe is a different full-universe job."
        ) % (exp_et.strftime("%Y-%m-%d %H:%M ET"), job.get("last_exit_code"))
        status = STATUS_FAILED
    elif behind:
        detail = (
            "Expected last closed bar %s. Hot-list IB hist is channel_touch_15m "
            "at each RTH close, not the 02:30 universe backfill."
        ) % exp_et.strftime("%Y-%m-%d %H:%M ET")
        status = STATUS_STALE
    elif not is_rth(now):
        status = STATUS_WAITING
        detail = "RTH closed. Last bar should be the prior session 15:45 ET until 9:45 ET."
    else:
        status = STATUS_UP_TO_DATE
        detail = "Last closed 15m bar matches the expected RTH slot."
    return _feed(
        feed_id="ib_15m",
        name="IB 15m hot list",
        role=role,
        cadence=cadence,
        as_of=as_of,
        expected_as_of=expected_s,
        status=status,
        detail=detail,
        job=job,
    )


def _ib_universe_feed(
    as_of: Any,
    *,
    now: datetime,
    job: dict,
) -> Dict[str, Any]:
    role = "Overnight catch-up of ALL ~1478 IB 15m names (job backfill_ib_15m_universe, client 8822). Not the hot list."
    cadence = "02:30 local daily"
    if job.get("running"):
        return _feed(
            feed_id="ib_15m_universe",
            name="IB 15m universe",
            role=role,
            cadence=cadence,
            as_of=as_of,
            expected_as_of=None,
            status=STATUS_UPDATING,
            detail="Overnight full-universe backfill is running.",
            job=job,
        )
    nxt = job.get("next_run") or "02:30 local"
    last = job.get("last_fired") or "never"
    rc = job.get("last_exit_code")
    if _job_failed(job):
        status = STATUS_FAILED
        detail = (
            "Last overnight run %s exit %s. This job does not run during RTH. "
            "Hot-list IB hist is channel_touch_15m every 15m. Next %s."
        ) % (last, rc, nxt)
    else:
        status = STATUS_WAITING
        detail = (
            "Idle until 02:30 local (full universe, not hot list). "
            "Last run %s exit %s. In-session bars come from channel_touch_15m."
        ) % (last, rc if rc is not None else "-")
    return _feed(
        feed_id="ib_15m_universe",
        name="IB 15m universe",
        role=role,
        cadence=cadence,
        as_of=as_of,
        expected_as_of=None,
        status=status,
        detail=detail,
        job=job,
    )


def _alpaca_1d_feed(
    as_of_1d: Any,
    *,
    now: datetime,
    job: dict,
) -> Dict[str, Any]:
    role = "Daily H2 armed book. EOD scan only - not a live API."
    cadence = "Weekdays 23:00 local (CronRunner channel_touch_nightly)"
    expected = last_completed_session_date(now)
    expected_s = _fmt_date(expected)
    got = _as_of_date(as_of_1d)
    if job.get("running"):
        return _feed(
            feed_id="alpaca_1d",
            name="Alpaca 1d",
            role=role,
            cadence=cadence,
            as_of=as_of_1d,
            expected_as_of=expected_s,
            status=STATUS_UPDATING,
            detail="Nightly scan is running (Alpaca daily bars + detector).",
            job=job,
        )
    if got is None:
        return _feed(
            feed_id="alpaca_1d",
            name="Alpaca 1d",
            role=role,
            cadence=cadence,
            as_of=as_of_1d,
            expected_as_of=expected_s,
            status=STATUS_MISSING,
            detail="No 1d as_of. Run channel_touch_nightly.",
            job=job,
        )
    nxt = job.get("next_run") or "23:00 local"
    if got < expected:
        if _job_failed(job):
            status = STATUS_FAILED
            detail = (
                "Expected last close %s. Nightly last exit %s. "
                "1d is EOD-only; next run %s. Today's bar is not a live feed."
            ) % (expected_s, job.get("last_exit_code"), nxt)
        else:
            status = STATUS_STALE
            detail = (
                "Expected last close %s (not today until after cash close). "
                "Nightly at 23:00 local is 16:00 ET - Alpaca daily often still has "
                "the prior session at that minute. Next run %s."
            ) % (expected_s, nxt)
    elif is_rth(now) or _to_et(now).time() < RTH_CLOSE:
        status = STATUS_WAITING
        detail = (
            "Caught up through last close %s. 1d does not update intraday; "
            "today's bar is scanned at 23:00 local tonight."
        ) % expected_s
    else:
        status = STATUS_UP_TO_DATE
        detail = "Last completed session matches the nightly as_of."
    return _feed(
        feed_id="alpaca_1d",
        name="Alpaca 1d",
        role=role,
        cadence=cadence,
        as_of=as_of_1d,
        expected_as_of=expected_s,
        status=status,
        detail=detail,
        job=job,
    )


def build_feeds(
    *,
    price_ts: Any = None,
    as_of_15m: Any = None,
    as_of_1d: Any = None,
    now: Optional[datetime] = None,
    jobs: Optional[Dict[str, dict]] = None,
    manager: Any = None,
) -> List[Dict[str, Any]]:
    """Status rows for the /hot API table."""
    now_ts = now or datetime.now(timezone.utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=timezone.utc)
    snaps = dict(jobs or {})
    needed = (JOB_ALPACA_LAST, JOB_IB_15M, JOB_IB_UNIVERSE, JOB_ALPACA_1D)
    if jobs is None:
        for name in needed:
            snaps[name] = cron_job_snapshot(name, manager=manager)
    else:
        for name in needed:
            snaps.setdefault(name, {"name": name, "running": False})
    return [
        _alpaca_last_feed(price_ts, now=now_ts, job=snaps[JOB_ALPACA_LAST]),
        _ib_15m_feed(as_of_15m, now=now_ts, job=snaps[JOB_IB_15M]),
        _ib_universe_feed(as_of_15m, now=now_ts, job=snaps[JOB_IB_UNIVERSE]),
        _alpaca_1d_feed(as_of_1d, now=now_ts, job=snaps[JOB_ALPACA_1D]),
    ]


def feeds_fingerprint(feeds: Sequence[dict]) -> tuple:
    """Stable hub key: ignore next_run clock drift."""
    slim = []
    for row in feeds:
        slim.append(
            (
                row.get("id"),
                row.get("status"),
                row.get("as_of"),
                row.get("expected_as_of"),
                row.get("job_running"),
                row.get("job_last_exit"),
                row.get("job_last_fired"),
            )
        )
    return tuple(slim)
