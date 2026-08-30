"""Five-field cron matching (minute hour day month weekday).

Supports *, lists, ranges, and steps. Day-of-month and day-of-week follow
classic cron OR semantics when both fields are restricted (not *).
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Sequence, Union

CronLike = Union[str, Sequence[str]]

_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

_DOWS = {
    "SUN": 0,
    "MON": 1,
    "TUE": 2,
    "WED": 3,
    "THU": 4,
    "FRI": 5,
    "SAT": 6,
}

_FIELD_BOUNDS = (
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 7),    # day of week (7 == Sunday)
)


def parse_schedule(schedule: CronLike) -> List[str]:
    """Normalize a schedule to a list of 5-field cron strings."""
    if isinstance(schedule, str):
        parts = [schedule.strip()]
    else:
        parts = [str(item).strip() for item in schedule]
    if not parts or any(not p for p in parts):
        raise ValueError("schedule must be a non-empty cron string or list")
    for expr in parts:
        validate_cron(expr)
    return parts


def validate_cron(expr: str) -> None:
    """Raise ValueError if expr is not a valid 5-field cron string."""
    fields = _split_fields(expr)
    names = ("minute", "hour", "day-of-month", "month", "day-of-week")
    for i, (field, (lo, hi), name) in enumerate(zip(fields, _FIELD_BOUNDS, names)):
        aliases = _MONTHS if i == 3 else (_DOWS if i == 4 else None)
        _validate_field(field, lo, hi, name, aliases)


def cron_matches(expr: str, dt: datetime) -> bool:
    """True if the 5-field cron expression matches dt (minute resolution)."""
    fields = _split_fields(expr)
    minute, hour, dom, month, dow = fields
    if not _field_matches(minute, dt.minute, 0, 59, None):
        return False
    if not _field_matches(hour, dt.hour, 0, 23, None):
        return False
    if not _field_matches(month, dt.month, 1, 12, _MONTHS):
        return False

    cron_dow = (dt.weekday() + 1) % 7  # Mon=1 ... Sat=6 Sun=0
    dom_star = _is_unrestricted(dom)
    dow_star = _is_unrestricted(dow)
    dom_ok = _field_matches(dom, dt.day, 1, 31, None)
    dow_ok = _field_matches(dow, cron_dow, 0, 7, _DOWS, wrap_seven=True)

    if dom_star and dow_star:
        return True
    if not dom_star and not dow_star:
        return dom_ok or dow_ok
    if dom_star:
        return dow_ok
    return dom_ok


def any_cron_matches(schedule: CronLike, dt: datetime) -> bool:
    """True if any expression in the schedule matches dt."""
    for expr in parse_schedule(schedule):
        if cron_matches(expr, dt):
            return True
    return False


def next_match(schedule: CronLike, after: datetime, limit_minutes: int = 8 * 24 * 60) -> datetime:
    """Return the next matching minute strictly after `after`, or raise."""
    from datetime import timedelta

    parse_schedule(schedule)  # validate
    cursor = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(limit_minutes):
        if any_cron_matches(schedule, cursor):
            return cursor
        cursor += timedelta(minutes=1)
    raise ValueError("no matching minute within search window")


def _split_fields(expr: str) -> List[str]:
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(
            "cron must have 5 fields (minute hour day month weekday), got %r" % expr
        )
    return fields


def _is_unrestricted(field: str) -> bool:
    return field == "*"


def _validate_field(field: str, lo: int, hi: int, name: str, aliases) -> None:
    try:
        _expand_field(field, lo, hi, aliases, wrap_seven=(name == "day-of-week"))
    except ValueError as exc:
        raise ValueError("invalid %s field %r: %s" % (name, field, exc)) from exc


def _field_matches(
    field: str,
    value: int,
    lo: int,
    hi: int,
    aliases,
    wrap_seven: bool = False,
) -> bool:
    allowed = _expand_field(field, lo, hi, aliases, wrap_seven=wrap_seven)
    if wrap_seven and value == 0 and 7 in allowed:
        return True
    return value in allowed


def _expand_field(
    field: str,
    lo: int,
    hi: int,
    aliases,
    wrap_seven: bool = False,
) -> set:
    out = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty list item")
        out.update(_expand_part(part, lo, hi, aliases, wrap_seven))
    return out


def _expand_part(
    part: str,
    lo: int,
    hi: int,
    aliases,
    wrap_seven: bool,
) -> Iterable[int]:
    step = 1
    if "/" in part:
        base, step_s = part.split("/", 1)
        step = _parse_int(step_s, aliases)
        if step <= 0:
            raise ValueError("step must be > 0")
    else:
        base = part

    if base == "*":
        start, end = lo, hi
    elif "-" in base:
        a, b = base.split("-", 1)
        start = _parse_bound(a, lo, hi, aliases, wrap_seven)
        end = _parse_bound(b, lo, hi, aliases, wrap_seven)
        if start > end:
            raise ValueError("range start > end")
    else:
        start = _parse_bound(base, lo, hi, aliases, wrap_seven)
        end = hi if "/" in part else start

    values = set()
    n = start
    while n <= end:
        values.add(n)
        n += step
    if wrap_seven and (0 in values or 7 in values):
        values.add(0)
        values.add(7)
    return values


def _parse_bound(token: str, lo: int, hi: int, aliases, _wrap_seven: bool) -> int:
    n = _parse_int(token, aliases)
    if n < lo or n > hi:
        raise ValueError("%s out of range %s-%s" % (n, lo, hi))
    return n


def _parse_int(token: str, aliases) -> int:
    key = token.strip().upper()
    if aliases and key in aliases:
        return aliases[key]
    try:
        return int(token, 10)
    except ValueError as exc:
        raise ValueError("not an integer or name: %r" % token) from exc
