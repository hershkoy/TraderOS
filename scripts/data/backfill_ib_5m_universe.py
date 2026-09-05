#!/usr/bin/env python3
"""
IB 5m backfill for the stored IB 15m universe.

Newest calendar year first for all symbols (default 2025-01-01 through now,
then 2024 ... 2020) so a whole-universe simulation can start before older
years finish. Resume is MAX(ts) inside that year window, not a skip-forever
file. Ctrl+C, --stop-file, --until, or the RTH yield window all stop after
the current window so the next run continues.

Do not run during US RTH: live 15m monitoring owns Gateway (client 8826).
Default client 8823. 15m universe is 8822; single-symbol 5m is 8824.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_5m_universe.py --inventory
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_5m_universe.py --dry-run --limit 2
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_5m_universe.py --sleep 1 --ib-client-id 8823
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import (  # noqa: E402
    _prepare_ib_duration_from_days,
    cleanup_ib_connection,
    create_ib_contract_with_primary_exchange,
    fetch_ib_historical_window,
    get_ib_connection,
    ib_step_back_minutes,
    prepare_nautilus_dataframe,
    save_to_timescaledb,
    set_ib_client_id,
)
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.db.market_data_coverage import (  # noqa: E402
    load_ib_coverage,
    load_ib_coverage_range,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_ib_5m_universe")
logging.getLogger("ib_insync").setLevel(logging.WARNING)
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

NY_TZ = ZoneInfo("America/New_York")
RTH_YIELD_START = (9, 15)
RTH_YIELD_END = (16, 30)

DEFAULT_LOCK = ROOT / "logs" / "data" / "ib_5m_universe.lock"
DEFAULT_STOP = ROOT / "logs" / "data" / "ib_5m_universe.stop"
DEFAULT_FAILED = ROOT / "logs" / "data" / "ib_5m_universe_failed.txt"
DEFAULT_PROGRESS = ROOT / "logs" / "data" / "ib_5m_universe_progress.json"
DEFAULT_INVENTORY = ROOT / "reports" / "ascending_channels" / "ib_5m_coverage.csv"
DEFAULT_CLIENT_ID = 8823
TIMEFRAME = "5m"
BAR_MINUTES = 5
DEFAULT_YEAR_FROM = 2025
DEFAULT_YEAR_TO = 2020
YEAR_SLACK_DAYS = 7


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IB 5m universe backfill (resume-safe, yields to RTH)")
    ap.add_argument(
        "--inventory",
        action="store_true",
        help="Write 5m coverage CSV (symbol, first_ts, last_ts) and exit (no IB)",
    )
    ap.add_argument(
        "--ib-client-id",
        type=int,
        default=DEFAULT_CLIENT_ID,
        help="Gateway client ID (default 8823; keep distinct from 8822/8826)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between IB windows (pacing)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max symbols to fetch this run")
    ap.add_argument("--dry-run", action="store_true", help="Qualify + fetch, do not insert")
    ap.add_argument(
        "--fresh-hours",
        type=float,
        default=36.0,
        help="Skip symbols whose 5m last bar is newer than this and caught up to 15m",
    )
    ap.add_argument(
        "--overlap-bars",
        type=int,
        default=2,
        help="Rewind this many 5m bars from last_ts to overlap upsert",
    )
    ap.add_argument(
        "--batch-days",
        type=int,
        default=7,
        help="IB window length in calendar days (default 7 = 1 W, official 5m max)",
    )
    ap.add_argument(
        "--stop-file",
        type=Path,
        default=DEFAULT_STOP,
        help="If this file exists, stop after the current IB window",
    )
    ap.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK,
    )
    ap.add_argument(
        "--failed-file",
        type=Path,
        default=DEFAULT_FAILED,
        help="Symbols that failed qualify/empty; skipped until --reset-failed",
    )
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=DEFAULT_PROGRESS,
    )
    ap.add_argument(
        "--inventory-out",
        type=Path,
        default=DEFAULT_INVENTORY,
    )
    ap.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional newline list; default = all IB 15m symbols in DB",
    )
    ap.add_argument(
        "--until",
        default="",
        help="Stop at this local/ISO datetime. Empty = next weekday 09:15 America/New_York",
    )
    ap.add_argument(
        "--no-until",
        action="store_true",
        help="Do not auto-stop at next 09:15 ET (still honors stop-file, RTH guard, Ctrl+C)",
    )
    ap.add_argument(
        "--allow-rth",
        action="store_true",
        help="Allow running during Mon-Fri 09:15-16:30 ET (default: refuse / stop)",
    )
    ap.add_argument(
        "--reset-failed",
        action="store_true",
        help="Ignore and rewrite the failed-symbol skip list",
    )
    ap.add_argument(
        "--ib-port",
        type=int,
        default=4001,
        help="Gateway port (default 4001; skips detect_ib_port client 98)",
    )
    ap.add_argument(
        "--skip-if-job-running",
        default="",
        help="Exit 0 if this CronRunner job still holds its lock (hourly watchdog)",
    )
    ap.add_argument(
        "--year-from",
        type=int,
        default=DEFAULT_YEAR_FROM,
        help="Newest calendar year to fill first (default 2025)",
    )
    ap.add_argument(
        "--year-to",
        type=int,
        default=DEFAULT_YEAR_TO,
        help="Oldest calendar year to fill last (default 2020)",
    )
    ap.add_argument(
        "--years",
        default="",
        help="Optional comma list overriding --year-from/--year-to "
        "(e.g. 2025,2024,2023,2022,2021,2020)",
    )
    ap.add_argument(
        "--no-through-now",
        action="store_true",
        help="Cap the newest year at Jan 1 of the next year (excludes 2026 YTD from the 2025 pass)",
    )
    ap.add_argument(
        "--no-year-slice",
        action="store_true",
        help="Old behavior: fill each symbol from first 15m bar to now before the next symbol",
    )
    return ap.parse_args(argv)


def _as_utc(ts) -> Optional[pd.Timestamp]:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)) or pd.isna(ts):
        return None
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def in_rth_yield_window(now: Optional[datetime] = None) -> bool:
    """True during Mon-Fri 09:15-16:30 America/New_York (live 15m owns IB)."""
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
    """Next Mon-Fri 09:15 America/New_York strictly after now."""
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


def parse_until(value: str, *, now: Optional[datetime] = None) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def needs_backfill(
    last_5m: Optional[pd.Timestamp],
    last_15m: Optional[pd.Timestamp],
    *,
    now: Optional[datetime] = None,
    fresh_hours: float = 36.0,
) -> bool:
    """True if 5m is missing, behind 15m history, or stale vs now."""
    ts5 = _as_utc(last_5m)
    if ts5 is None:
        return True
    now_ts = _as_utc(now or datetime.now(timezone.utc))
    age_h = (now_ts - ts5).total_seconds() / 3600.0
    ts15 = _as_utc(last_15m)
    if ts15 is not None and ts5 + timedelta(minutes=BAR_MINUTES * 2) < ts15:
        return True
    return age_h > float(fresh_hours)


def rewind_start(last_ts: pd.Timestamp, overlap_bars: int) -> datetime:
    ts = _as_utc(last_ts)
    delta = timedelta(minutes=BAR_MINUTES * max(0, int(overlap_bars)))
    return (ts - delta).to_pydatetime()


def parse_year_list(years_csv: str, year_from: int, year_to: int) -> List[int]:
    """Newest-first year order. --years overrides --year-from/--year-to."""
    raw = str(years_csv or "").strip()
    if raw:
        out = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not out:
            raise ValueError("empty --years")
        return out
    newest = int(year_from)
    oldest = int(year_to)
    if newest < oldest:
        raise ValueError("year-from must be >= year-to (newest year first)")
    return list(range(newest, oldest - 1, -1))


def year_slice_bounds(
    year: int,
    *,
    newest_year: int,
    now: datetime,
    through_now: bool,
) -> Tuple[datetime, datetime]:
    """UTC [Jan 1, next Jan 1). Newest year optionally extends through now (2026 YTD)."""
    start = datetime(int(year), 1, 1, tzinfo=timezone.utc)
    when = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    if through_now and int(year) == int(newest_year):
        return start, when
    return start, datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc)


def _window_start_bound(
    window_start: datetime,
    first_15m: Optional[pd.Timestamp],
) -> datetime:
    bound = window_start
    if bound.tzinfo is None:
        bound = bound.replace(tzinfo=timezone.utc)
    f15 = _as_utc(first_15m)
    if f15 is not None:
        fdt = f15.to_pydatetime()
        if fdt.tzinfo is None:
            fdt = fdt.replace(tzinfo=timezone.utc)
        if fdt > bound:
            return fdt
    return bound


def year_window_complete(
    first_in: Optional[pd.Timestamp],
    last_in: Optional[pd.Timestamp],
    *,
    window_start: datetime,
    window_end: datetime,
    first_15m: Optional[pd.Timestamp],
    last_15m: Optional[pd.Timestamp] = None,
    now: Optional[datetime] = None,
    slack_days: int = YEAR_SLACK_DAYS,
    through_now: bool = False,
    fresh_hours: float = 36.0,
) -> bool:
    """True if 5m already covers this year window (gap at start counts as incomplete)."""
    last_ts = _as_utc(last_in)
    if last_ts is None:
        return False
    first_ts = _as_utc(first_in)
    start_bound = _window_start_bound(window_start, first_15m)
    slack = timedelta(days=max(1, int(slack_days)))
    if first_ts is None or first_ts > _as_utc(start_bound) + slack:
        return False
    if through_now:
        return not needs_backfill(
            last_ts, last_15m, now=now, fresh_hours=float(fresh_hours)
        )
    end_ts = _as_utc(window_end)
    return last_ts >= end_ts - slack


def year_fetch_start(
    last_in: Optional[pd.Timestamp],
    first_in: Optional[pd.Timestamp],
    *,
    window_start: datetime,
    first_15m: Optional[pd.Timestamp],
    overlap_bars: int,
    slack_days: int = YEAR_SLACK_DAYS,
) -> datetime:
    """Resume inside the year, or restart from the window start if the front is missing."""
    start_bound = _window_start_bound(window_start, first_15m)
    last_ts = _as_utc(last_in)
    if last_ts is None:
        return start_bound
    first_ts = _as_utc(first_in)
    slack = timedelta(days=max(1, int(slack_days)))
    if first_ts is None or first_ts > _as_utc(start_bound) + slack:
        return start_bound
    return rewind_start(last_ts, overlap_bars)


def symbol_year_job(
    symbol: str,
    first_15m: Optional[pd.Timestamp],
    last_15m: Optional[pd.Timestamp],
    first_in: Optional[pd.Timestamp],
    last_in: Optional[pd.Timestamp],
    *,
    year: int,
    window_start: datetime,
    window_end: datetime,
    through_now: bool,
    now: datetime,
    overlap_bars: int,
    fresh_hours: float,
) -> Optional[Tuple[str, int, datetime, datetime]]:
    """One (symbol, year) fetch job, or None if this window is skippable."""
    f15 = _as_utc(first_15m)
    l15 = _as_utc(last_15m)
    if f15 is not None and f15 >= _as_utc(window_end):
        return None
    if l15 is not None and l15 < _as_utc(window_start):
        return None
    if year_window_complete(
        first_in,
        last_in,
        window_start=window_start,
        window_end=window_end,
        first_15m=first_15m,
        last_15m=last_15m,
        now=now,
        through_now=through_now,
        fresh_hours=float(fresh_hours),
    ):
        return None
    start_dt = year_fetch_start(
        last_in,
        first_in,
        window_start=window_start,
        first_15m=first_15m,
        overlap_bars=int(overlap_bars),
    )
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if start_dt >= window_end:
        return None
    return (str(symbol).upper(), int(year), start_dt, window_end)


def load_skip_list(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def append_skip_list(path: Path, symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(str(symbol).upper() + "\n")


def load_symbol_5m_last(symbol: str) -> Optional[pd.Timestamp]:
    """MAX(ts) for one IB 5m symbol (index-friendly; not a universe GROUP BY)."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")
    sql = """
        SELECT MAX(ts) FROM market_data
        WHERE provider = %s AND timeframe = %s AND symbol = %s
    """
    cur = client.connection.cursor()
    try:
        cur.execute(sql, ("IB", TIMEFRAME, str(symbol).upper()))
        row = cur.fetchone()
    finally:
        cur.close()
    if not row or row[0] is None:
        return None
    return _as_utc(row[0])


def load_symbol_5m_range(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """MIN/MAX(ts) for one IB 5m symbol inside [start, end)."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")
    sql = """
        SELECT MIN(ts), MAX(ts) FROM market_data
        WHERE provider = %s AND timeframe = %s AND symbol = %s
          AND ts >= %s AND ts < %s
    """
    cur = client.connection.cursor()
    try:
        cur.execute(
            sql,
            ("IB", TIMEFRAME, str(symbol).upper(), start_dt, end_dt),
        )
        row = cur.fetchone()
    finally:
        cur.close()
    if not row or (row[0] is None and row[1] is None):
        return None, None
    return _as_utc(row[0]), _as_utc(row[1])


def _symbols_from_file(path: Path) -> List[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def coverage_summary(df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"n": 0}
    last = pd.to_datetime(df["last_ts"], utc=True, errors="coerce")
    first = pd.to_datetime(df["first_ts"], utc=True, errors="coerce")
    return {
        "n": int(len(df)),
        "first_min": str(first.min()) if first.notna().any() else None,
        "last_min": str(last.min()) if last.notna().any() else None,
        "last_max": str(last.max()) if last.notna().any() else None,
        "last_median": str(last.median()) if last.notna().any() else None,
    }


def write_inventory(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote coverage %s (%d symbols)", path, len(df))
    return path


def write_progress(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _ns_ts(df: pd.DataFrame, first: bool) -> str:
    if "ts_event" not in df.columns:
        return "?"
    val = int(df["ts_event"].min() if first else df["ts_event"].max())
    return str(pd.Timestamp(val, unit="ns", tz="UTC"))


def clear_stop_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            logger.info("Cleared stop file %s", path)
    except OSError as exc:
        logger.warning("Could not clear stop file %s: %s", path, exc)


class StopRequested(Exception):
    """Cooperative stop after the current IB window."""


class PidLock:
    def __init__(self, path: Path):
        self.path = path
        self.held = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(data.get("pid") or 0)
            except (OSError, ValueError, json.JSONDecodeError):
                pid = 0
            if pid and pid != os.getpid() and _pid_holds_5m_lock(pid):
                logger.error("Another 5m backfill is running (pid %s, lock %s)", pid, self.path)
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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
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
    """Best-effort command line for a live pid (stale-lock detection after reboot)."""
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


def _pid_holds_5m_lock(pid: int) -> bool:
    """True only if pid is alive and still looks like this backfill (not a recycled pid)."""
    if not _pid_alive(pid):
        return False
    cmd = _process_cmdline(pid).lower().replace("/", "\\")
    if not cmd:
        return True
    return "scripts\\data\\backfill_ib_5m_universe.py" in cmd


_TRANSIENT_IB_MARKERS = (
    "connection refused",
    "connectionrefused",
    "refused the network",
    "not connected",
    "winerror 1225",
    "10054",
    "10053",
    "10060",
    "timed out",
    "timeout",
    "forcibly closed",
    "api connection failed",
    "make sure api port",
)


def is_transient_ib_error(exc: BaseException) -> bool:
    """Gateway down / handshake drop: retry on next run, do not poison the failed list."""
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(marker in text for marker in _TRANSIENT_IB_MARKERS)


def make_should_stop(
    *,
    stop_file: Path,
    until: Optional[datetime],
    allow_rth: bool,
    flag: Dict[str, bool],
) -> Callable[[], bool]:
    def _check() -> bool:
        if flag.get("stop"):
            return True
        if stop_file.exists():
            logger.info("Stop file present: %s", stop_file)
            return True
        now = datetime.now(timezone.utc)
        if until is not None and now >= until.astimezone(timezone.utc):
            logger.info("Reached --until %s", until.isoformat())
            return True
        if not allow_rth and in_rth_yield_window(now):
            logger.info("US RTH yield window (09:15-16:30 ET); stopping 5m backfill")
            return True
        return False

    return _check


def backfill_symbol_forward(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    batch_days: int,
    sleep_s: float,
    dry_run: bool,
    should_stop: Callable[[], bool],
    on_progress: Optional[Callable[[str, int, str, str], None]] = None,
) -> Tuple[int, str]:
    """Fetch 5m forward from start_dt, upsert each window. Returns (bars, status)."""
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    if start_dt >= end_dt:
        return 0, "caught_up"

    ib = get_ib_connection()
    contract = create_ib_contract_with_primary_exchange(symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.error("%s: qualifyContracts returned empty", symbol)
        return 0, "qualify_failed"
    contract = qualified[0]

    probe = fetch_ib_historical_window(
        ib,
        contract,
        timeframe=TIMEFRAME,
        end_dt=end_dt,
        duration_str="1 W",
    )
    if probe is None or probe.empty:
        logger.warning("%s: no recent IB 5m bars; skipping", symbol)
        return 0, "empty"
    if sleep_s > 0:
        time.sleep(sleep_s)

    cursor = start_dt
    n_saved = 0
    empty_streak = 0
    step = timedelta(minutes=ib_step_back_minutes(TIMEFRAME))

    while cursor < end_dt:
        if should_stop():
            raise StopRequested()
        window_end = min(cursor + timedelta(days=max(1, int(batch_days))), end_dt)
        duration_days = max(1, (window_end - cursor).days + 1)
        duration_days = min(duration_days, max(1, int(batch_days)))
        dur_unit = _prepare_ib_duration_from_days(duration_days)
        logger.info(
            "%s: IB 5m window %s -> %s (%s)",
            symbol,
            cursor.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d"),
            dur_unit,
        )
        raw = fetch_ib_historical_window(
            ib,
            contract,
            timeframe=TIMEFRAME,
            end_dt=window_end,
            duration_str=dur_unit,
        )
        if raw is None or raw.empty:
            empty_streak += 1
            logger.warning("%s: no 5m bars in window (empty_streak=%d)", symbol, empty_streak)
            # IB 5m often starts later than 15m. Keep walking while we have saved nothing.
            if n_saved > 0 and empty_streak >= 8:
                logger.error("%s: too many empty 5m windows after data started", symbol)
                return n_saved, "ok"
            cursor = window_end
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue
        empty_streak = 0
        raw = raw[raw["timestamp"] >= cursor]
        raw = raw[raw["timestamp"] <= window_end]
        if raw.empty:
            cursor = window_end
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue
        df = prepare_nautilus_dataframe(raw.copy(), symbol, "IB", TIMEFRAME)
        ts0 = _ns_ts(df, True)
        ts1 = _ns_ts(df, False)
        logger.info("%s: fetched %d 5m bars (%s -> %s)", symbol, len(df), ts0, ts1)
        if dry_run:
            n_saved += int(len(df))
        else:
            if not save_to_timescaledb(df, symbol, "IB", TIMEFRAME):
                logger.error("%s: TimescaleDB insert failed", symbol)
                return n_saved, "insert_failed"
            n_saved += int(len(df))
        last_ts = _as_utc(raw["timestamp"].max())
        if on_progress is not None:
            on_progress(symbol, n_saved, ts0, ts1)
        cursor = (last_ts + step).to_pydatetime()
        if cursor.tzinfo is None:
            cursor = cursor.replace(tzinfo=timezone.utc)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return n_saved, "ok"


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("IB_PORT", str(int(args.ib_port)))

    if str(args.skip_if_job_running).strip() and not args.inventory:
        from utils.cron.manager import CronManager

        alive, pid = CronManager(root=ROOT).lock_status(str(args.skip_if_job_running).strip())
        if alive:
            logger.info(
                "Skip: CronRunner job %s still running (pid %s)",
                args.skip_if_job_running,
                pid,
            )
            return 0

    logger.info("Loading IB 15m coverage from TimescaleDB (5m universe = 15m symbols) ...")
    cov15 = load_ib_coverage("15m")
    summary15 = coverage_summary(cov15)
    logger.info("15m coverage: %s", json.dumps(summary15))
    if cov15.empty:
        logger.error("No IB 15m symbols in market_data")
        return 1

    logger.info("Loading IB 5m coverage ...")
    cov5 = load_ib_coverage(TIMEFRAME)
    summary5 = coverage_summary(cov5)
    logger.info("5m coverage: %s", json.dumps(summary5))
    write_inventory(cov5, args.inventory_out)
    if args.inventory:
        return 0

    if args.symbols_file is not None:
        wanted = set(_symbols_from_file(args.symbols_file))
        cov15 = cov15[cov15["symbol"].isin(wanted)].copy()
        logger.info("Symbols-file filter: %d names", len(cov15))

    if args.reset_failed and args.failed_file.exists():
        args.failed_file.unlink()
        logger.info("Cleared failed file %s", args.failed_file)
    failed = load_skip_list(args.failed_file)
    last5_map = {}
    if not cov5.empty:
        last5_map = {
            str(row["symbol"]).upper(): row["last_ts"] for _, row in cov5.iterrows()
        }

    now = datetime.now(timezone.utc)
    until = None
    if not args.no_until:
        until = parse_until(args.until, now=now) if str(args.until).strip() else next_rth_yield_dt(now)
        logger.info("Will stop at %s", until.isoformat())

    if not args.allow_rth and in_rth_yield_window(now):
        logger.error(
            "Refusing to start during US RTH yield window (09:15-16:30 ET). "
            "Live 15m monitoring needs Gateway. Use --allow-rth only for emergencies."
        )
        return 1

    year_slice = not bool(args.no_year_slice)
    years: List[int] = []
    through_now = not bool(args.no_through_now)
    todo: List[Tuple[str, Optional[int], datetime, datetime]] = []
    skipped_fresh = 0
    skipped_failed = 0
    skipped_year_done = 0
    newest_year = None
    if year_slice:
        try:
            years = parse_year_list(args.years, int(args.year_from), int(args.year_to))
        except ValueError as exc:
            logger.error("Bad year list: %s", exc)
            return 1
        newest_year = years[0]
        skipped_failed = sum(
            1 for _, row in cov15.iterrows() if str(row["symbol"]).upper() in failed
        )
        logger.info(
            "Year order %s (newest year %s through_now=%s)",
            years,
            newest_year,
            through_now,
        )
        for year in years:
            ystart, yend = year_slice_bounds(
                year,
                newest_year=newest_year,
                now=now,
                through_now=through_now,
            )
            logger.info(
                "Loading IB 5m coverage for %s (%s -> %s) ...",
                year,
                ystart.strftime("%Y-%m-%d"),
                yend.strftime("%Y-%m-%d %H:%M"),
            )
            cov_year = load_ib_coverage_range(
                TIMEFRAME,
                ystart,
                yend,
                symbols=[str(s).upper() for s in cov15["symbol"].tolist()],
            )
            range_map = {}
            if not cov_year.empty:
                range_map = {
                    str(r["symbol"]).upper(): (r["first_ts"], r["last_ts"])
                    for _, r in cov_year.iterrows()
                }
            n_before = len(todo)
            for _, row in cov15.iterrows():
                sym = str(row["symbol"]).upper()
                if sym in failed:
                    continue
                first_in, last_in = range_map.get(sym, (None, None))
                job = symbol_year_job(
                    sym,
                    row["first_ts"],
                    row["last_ts"],
                    first_in,
                    last_in,
                    year=year,
                    window_start=ystart,
                    window_end=yend,
                    through_now=through_now and year == newest_year,
                    now=now,
                    overlap_bars=int(args.overlap_bars),
                    fresh_hours=float(args.fresh_hours),
                )
                if job is None:
                    skipped_year_done += 1
                    continue
                todo.append(job)
            logger.info(
                "Year %s queue +%d (total %d)",
                year,
                len(todo) - n_before,
                len(todo),
            )
    else:
        for _, row in cov15.iterrows():
            sym = str(row["symbol"]).upper()
            last_15m = row["last_ts"]
            first_15m = row["first_ts"]
            if sym in failed:
                skipped_failed += 1
                continue
            last_5m = last5_map.get(sym)
            if not needs_backfill(
                last_5m, last_15m, now=now, fresh_hours=float(args.fresh_hours)
            ):
                skipped_fresh += 1
                continue
            start_dt = _as_utc(first_15m).to_pydatetime()
            if last_5m is not None and not pd.isna(last_5m):
                start_dt = rewind_start(last_5m, int(args.overlap_bars))
            todo.append((sym, None, start_dt, now))
    if args.limit and args.limit > 0:
        todo = todo[: int(args.limit)]
    logger.info(
        "Fetch queue %d | skipped_fresh=%d skipped_year_done=%d skipped_failed=%d "
        "year_slice=%s years=%s fresh_hours=%.1f client_id=%s batch_days=%d dry_run=%s",
        len(todo),
        skipped_fresh,
        skipped_year_done,
        skipped_failed,
        year_slice,
        years if year_slice else "full-history",
        float(args.fresh_hours),
        args.ib_client_id,
        int(args.batch_days),
        bool(args.dry_run),
    )
    if not todo:
        logger.info("Nothing to backfill")
        return 0

    lock = PidLock(args.lock_file)
    if not lock.acquire():
        logger.info("Another 5m backfill holds the lock; nothing to do")
        return 0

    flag = {"stop": False}

    def _on_signal(signum, _frame):
        logger.info("Signal %s: will stop after current IB window", signum)
        flag["stop"] = True

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    should_stop = make_should_stop(
        stop_file=args.stop_file,
        until=until,
        allow_rth=bool(args.allow_rth),
        flag=flag,
    )
    if not args.dry_run:
        clear_stop_file(args.stop_file)

    set_ib_client_id(int(args.ib_client_id))
    n_ok = 0
    n_fail = 0
    n_stopped = False
    t0 = time.perf_counter()
    try:
        for i, (sym, year, start_dt, end_dt) in enumerate(todo, start=1):
            if should_stop():
                n_stopped = True
                logger.info("Stopping before %s (%d/%d)", sym, i, len(todo))
                break
            if year is None:
                live_last = load_symbol_5m_last(sym)
                if live_last is not None:
                    start_dt = rewind_start(live_last, int(args.overlap_bars))
                    end_dt = datetime.now(timezone.utc)
            else:
                ystart, yend = year_slice_bounds(
                    year,
                    newest_year=newest_year if newest_year is not None else year,
                    now=now,
                    through_now=through_now and newest_year is not None and year == newest_year,
                )
                first_in, live_last = load_symbol_5m_range(sym, ystart, yend)
                start_dt = year_fetch_start(
                    live_last,
                    first_in,
                    window_start=ystart,
                    first_15m=None,
                    overlap_bars=int(args.overlap_bars),
                )
                end_dt = yend
            logger.info(
                "[%d/%d] %s year=%s 5m_last=%s fetch_from=%s fetch_to=%s",
                i,
                len(todo),
                sym,
                year if year is not None else "full",
                live_last,
                start_dt.isoformat(),
                end_dt.isoformat(),
            )
            if start_dt >= end_dt:
                n_ok += 1
                logger.info("%s: year=%s already caught up", sym, year)
                continue

            def _progress(symbol, n_bars, ts0, ts1, _year=year, _i=i):
                write_progress(
                    args.progress_file,
                    {
                        "symbol": symbol,
                        "year": _year,
                        "n_bars_this_symbol": n_bars,
                        "window_first": ts0,
                        "window_last": ts1,
                        "queue_i": _i,
                        "queue_n": len(todo),
                        "updated": datetime.now(timezone.utc).isoformat(),
                    },
                )

            try:
                n_bars, status = backfill_symbol_forward(
                    sym,
                    start_dt,
                    end_dt,
                    batch_days=int(args.batch_days),
                    sleep_s=float(args.sleep),
                    dry_run=bool(args.dry_run),
                    should_stop=should_stop,
                    on_progress=_progress,
                )
            except StopRequested:
                n_stopped = True
                logger.info("%s: cooperative stop after current window; re-run to continue", sym)
                break
            except Exception as exc:
                logger.exception("%s: backfill failed: %s", sym, exc)
                n_fail += 1
                if not args.dry_run and not is_transient_ib_error(exc):
                    append_skip_list(args.failed_file, sym)
                elif not args.dry_run:
                    logger.warning(
                        "%s: transient IB error; not adding to failed list",
                        sym,
                    )
                continue
            if status in {"ok", "caught_up"}:
                n_ok += 1
                logger.info("%s: done year=%s status=%s bars=%d", sym, year, status, n_bars)
            elif status == "empty" and year is not None:
                n_ok += 1
                logger.info("%s: no IB 5m in %s; continuing other years", sym, year)
            elif status == "qualify_failed" and year is not None and year != newest_year:
                n_ok += 1
                logger.info("%s: qualify failed for %s; not failing older/newer years", sym, year)
            elif status in {"qualify_failed", "empty", "insert_failed"}:
                n_fail += 1
                if not args.dry_run:
                    append_skip_list(args.failed_file, sym)
            else:
                n_ok += 1
    finally:
        cleanup_ib_connection()
        lock.release()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Done ok=%d fail=%d stopped=%s elapsed_sec=%.1f queue=%d",
        n_ok,
        n_fail,
        n_stopped,
        elapsed,
        len(todo),
    )
    write_progress(
        args.progress_file,
        {
            "finished": datetime.now(timezone.utc).isoformat(),
            "ok": n_ok,
            "fail": n_fail,
            "stopped": n_stopped,
            "elapsed_sec": round(elapsed, 1),
            "queue": len(todo),
            "years": years if year_slice else None,
        },
    )
    if n_stopped:
        return 0
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
