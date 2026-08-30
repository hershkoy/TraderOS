"""Unit tests for five-field cron matching."""
from datetime import datetime

import pytest

from utils.cron.cronexpr import cron_matches, next_match, parse_schedule, validate_cron


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def test_star_matches_any_minute():
    assert cron_matches("* * * * *", _dt("2026-08-30T15:07:00"))


def test_exact_minute_hour():
    expr = "0 23 * * *"
    assert cron_matches(expr, _dt("2026-08-30T23:00:00"))
    assert not cron_matches(expr, _dt("2026-08-30T23:01:00"))
    assert not cron_matches(expr, _dt("2026-08-30T22:00:00"))


def test_list_and_range_and_step():
    expr = "0,15,30,45 9-16 * * 1-5"
    assert cron_matches(expr, _dt("2026-08-31T09:15:00"))  # Monday
    assert cron_matches(expr, _dt("2026-08-31T16:45:00"))
    assert not cron_matches(expr, _dt("2026-08-31T09:16:00"))
    assert not cron_matches(expr, _dt("2026-08-30T09:15:00"))  # Sunday


def test_step_every_15():
    expr = "*/15 * * * *"
    assert cron_matches(expr, _dt("2026-08-30T12:00:00"))
    assert cron_matches(expr, _dt("2026-08-30T12:45:00"))
    assert not cron_matches(expr, _dt("2026-08-30T12:07:00"))


def test_dow_names_and_sunday_seven():
    assert cron_matches("0 9 * * MON", _dt("2026-08-31T09:00:00"))
    assert cron_matches("0 9 * * 0", _dt("2026-08-30T09:00:00"))
    assert cron_matches("0 9 * * 7", _dt("2026-08-30T09:00:00"))
    assert not cron_matches("0 9 * * 7", _dt("2026-08-31T09:00:00"))


def test_month_name():
    assert cron_matches("0 0 1 AUG *", _dt("2026-08-01T00:00:00"))
    assert not cron_matches("0 0 1 AUG *", _dt("2026-07-01T00:00:00"))


def test_dom_or_dow_when_both_restricted():
    # 1st of month OR Monday
    expr = "0 0 1 * 1"
    assert cron_matches(expr, _dt("2026-09-01T00:00:00"))  # Tuesday 1st
    assert cron_matches(expr, _dt("2026-08-31T00:00:00"))  # Monday not 1st
    assert not cron_matches(expr, _dt("2026-09-02T00:00:00"))  # Wednesday 2nd


def test_validate_rejects_bad_expr():
    with pytest.raises(ValueError):
        validate_cron("* * *")
    with pytest.raises(ValueError):
        validate_cron("60 * * * *")
    with pytest.raises(ValueError):
        parse_schedule("")


def test_next_match_finds_following_minute():
    nxt = next_match("0 23 * * 1-5", _dt("2026-08-30T12:00:00"))
    assert nxt == _dt("2026-08-31T23:00:00")


def test_schedule_list_any_match():
    from utils.cron.cronexpr import any_cron_matches

    sched = ["45 9 * * 1-5", "0 16 * * 1-5"]
    assert any_cron_matches(sched, _dt("2026-08-31T09:45:00"))
    assert any_cron_matches(sched, _dt("2026-08-31T16:00:00"))
    assert not any_cron_matches(sched, _dt("2026-08-31T10:00:00"))
