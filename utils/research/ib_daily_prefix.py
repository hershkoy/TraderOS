"""Plan IB daily prefix windows.

The 1d panel is ALPACA from each name's listing (often ~2020-2022) plus an IB
prefix written strictly before the first stored bar. A single IB historical
request should stay around 10 years. Longer gaps walk backward from the first
bar already in TimescaleDB so a partial run can resume.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple
from zoneinfo import ZoneInfo

# IB lifted the hard 1-year cap on daily bars, but a 20-year request is a soft
# throttle risk. 10 years is about 2,500 daily bars, under the 3,000-bar cap
# used by fetch_max_from_ib.
DEFAULT_CHUNK_YEARS = 10
# Research loaders treat 2006-01-01 as complete history (--complete-history).
DEFAULT_HISTORY_START = "2006-01-01"


def iter_prefix_chunks(
    start: datetime,
    end: datetime,
    *,
    chunk_years: int = DEFAULT_CHUNK_YEARS,
) -> Iterator[Tuple[datetime, datetime]]:
    """Yield [chunk_start, chunk_end) windows from ``end`` backward to ``start``.

    The first window ends at ``end`` (the earliest bar already stored). Filling
    that edge first keeps MIN(ts) moving backward, so a resumed run does not
    skip a hole in the middle.
    """
    if chunk_years < 1:
        raise ValueError("chunk_years must be >= 1")
    if end <= start:
        return
    span = timedelta(days=int(chunk_years * 365))
    cursor = end
    while cursor > start:
        chunk_start = start if cursor - span <= start else cursor - span
        yield chunk_start, cursor
        if chunk_start <= start:
            break
        cursor = chunk_start


NY_TZ = ZoneInfo("America/New_York")
# Live 15m (client 8826) owns Gateway on weekdays from 09:15 through 16:30 ET.
RTH_YIELD_START = (9, 15)
RTH_YIELD_END = (16, 30)


def in_rth_yield_window(now: Optional[datetime] = None) -> bool:
    """True during Mon-Fri 09:15-16:30 America/New_York."""
    when = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    ny = when.astimezone(NY_TZ)
    if ny.weekday() >= 5:
        return False
    start = ny.replace(hour=RTH_YIELD_START[0], minute=RTH_YIELD_START[1], second=0, microsecond=0)
    end = ny.replace(hour=RTH_YIELD_END[0], minute=RTH_YIELD_END[1], second=0, microsecond=0)
    return start <= ny < end


def next_rth_yield_dt(now: Optional[datetime] = None) -> datetime:
    """Next weekday 09:15 America/New_York strictly after ``now``."""
    when = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    ny = when.astimezone(NY_TZ)
    day = ny.date()
    for i in range(0, 10):
        candidate_day = day + timedelta(days=i)
        candidate = datetime(
            candidate_day.year,
            candidate_day.month,
            candidate_day.day,
            RTH_YIELD_START[0],
            RTH_YIELD_START[1],
            tzinfo=NY_TZ,
        )
        if candidate.weekday() >= 5:
            continue
        if candidate > ny:
            return candidate
    raise RuntimeError("could not find next RTH yield datetime")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_cmdline(pid: int) -> str:
    if pid <= 0:
        return ""
    if sys.platform == "win32":
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine" % pid,
                ],
                capture_output=True,
                text=True,
                timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return (completed.stdout or "").strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    try:
        raw = Path("/proc/%d/cmdline" % pid).read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


class PidLock:
    """One backfill process. A lock whose pid is dead or recycled is stolen."""

    def __init__(self, path: Path, needle: str):
        self.path = path
        self.needle = needle.lower().replace("/", "\\")
        self.held = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(data.get("pid") or 0)
            except (OSError, ValueError, json.JSONDecodeError):
                pid = 0
            if pid and pid != os.getpid() and self._holds(pid):
                return False
            try:
                self.path.unlink()
            except OSError:
                pass
        payload = {"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.held = True
        return True

    def release(self) -> None:
        if not self.held:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self.held = False

    def _holds(self, pid: int) -> bool:
        if not _pid_alive(pid):
            return False
        cmd = _process_cmdline(pid).lower().replace("/", "\\")
        if not cmd:
            return True
        return self.needle in cmd


def make_should_stop(
    *,
    stop_file: Path,
    until: Optional[datetime],
    allow_rth: bool,
    flag: dict,
) -> Callable[[], bool]:
    def _check() -> bool:
        if flag.get("stop"):
            return True
        if stop_file.exists():
            return True
        now = datetime.now(timezone.utc)
        if until is not None and now >= until.astimezone(timezone.utc):
            return True
        if not allow_rth and in_rth_yield_window(now):
            return True
        return False

    return _check


def duration_str(start: datetime, end: datetime) -> str:
    """IB durationStr covering [start, end], padded a few days. Years if > 365d."""
    days = max(1, (end - start).days + 7)
    if days > 365:
        years = max(1, (days + 364) // 365)
        return f"{years} Y"
    return f"{days} D"
