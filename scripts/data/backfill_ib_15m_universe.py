#!/usr/bin/env python3
"""
Incremental IB 15m backfill for the stored 15m universe.

IB 15m in TimescaleDB currently ends ~2025-12-02. This walks each IB 15m
symbol, reads MAX(ts), and fetches only the gap to now. Resume-safe.

Do not run next to another IB historical session without a distinct client id
(default 8822; single-symbol backfill uses 8821).

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_15m_universe.py --inventory
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_15m_universe.py --dry-run --limit 5
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_15m_universe.py --sleep 1 --ib-client-id 8822
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import (  # noqa: E402
    _fetch_ib_intraday_batched,
    cleanup_ib_connection,
    create_ib_contract_with_primary_exchange,
    get_ib_connection,
    prepare_nautilus_dataframe,
    save_to_timescaledb,
    set_ib_client_id,
)
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_ib_15m_universe")
logging.getLogger("ib_insync").setLevel(logging.WARNING)
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_RESUME = ROOT / "logs" / "data" / "ib_15m_universe_resume.txt"
DEFAULT_INVENTORY = ROOT / "reports" / "ascending_channels" / "ib_15m_coverage.csv"
DEFAULT_CLIENT_ID = 8822


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Incremental IB 15m universe backfill")
    ap.add_argument(
        "--inventory",
        action="store_true",
        help="Write coverage CSV (symbol, first_ts, last_ts) and exit (no IB)",
    )
    ap.add_argument(
        "--ib-client-id",
        type=int,
        default=DEFAULT_CLIENT_ID,
        help="Gateway client ID (default 8822; keep distinct from other IB sessions)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between symbols (IB pacing)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max symbols to fetch this run")
    ap.add_argument("--dry-run", action="store_true", help="Qualify + fetch, do not insert")
    ap.add_argument(
        "--fresh-hours",
        type=float,
        default=36.0,
        help="Skip symbols whose last bar is newer than this many hours",
    )
    ap.add_argument(
        "--stale-before",
        default="",
        help="Only symbols with last_ts < this UTC date (YYYY-MM-DD). Empty = all stale vs --fresh-hours",
    )
    ap.add_argument(
        "--overlap-bars",
        type=int,
        default=2,
        help="Rewind this many 15m bars from last_ts to overlap upsert",
    )
    ap.add_argument(
        "--resume-file",
        type=Path,
        default=DEFAULT_RESUME,
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
        "--reset-resume",
        action="store_true",
        help="Ignore and rewrite the resume file",
    )
    return ap.parse_args(argv)


def needs_backfill(
    last_ts: Optional[pd.Timestamp],
    *,
    now: Optional[datetime] = None,
    fresh_hours: float = 36.0,
    stale_before: Optional[datetime] = None,
) -> bool:
    """True if this symbol still has an IB 15m gap to fill."""
    if last_ts is None or pd.isna(last_ts):
        return True
    ts = pd.Timestamp(last_ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    else:
        ts = ts.tz_localize("UTC")
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    if stale_before is not None:
        cut = pd.Timestamp(stale_before)
        if cut.tzinfo is None:
            cut = cut.tz_localize("UTC")
        return ts < cut
    age_h = (now_ts - ts).total_seconds() / 3600.0
    return age_h > float(fresh_hours)


def load_resume(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def append_resume(path: Path, symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(str(symbol).upper() + "\n")


def load_ib_15m_coverage() -> pd.DataFrame:
    """Per-symbol first/last IB 15m timestamp. Scoped to provider+timeframe."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")
    sql = """
        SELECT symbol, MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n_bars
        FROM market_data
        WHERE provider = %s AND timeframe = %s
        GROUP BY symbol
        ORDER BY symbol
    """
    cur = client.connection.cursor()
    try:
        cur.execute("SET LOCAL statement_timeout = '120s'")
        cur.execute(sql, ("IB", "15m"))
        rows = cur.fetchall()
    finally:
        cur.close()
    df = pd.DataFrame(rows, columns=["symbol", "first_ts", "last_ts", "n_bars"])
    if df.empty:
        return df
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def _symbols_from_file(path: Path) -> List[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def rewind_start(last_ts: pd.Timestamp, overlap_bars: int) -> datetime:
    ts = pd.Timestamp(last_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    delta = timedelta(minutes=15 * max(0, int(overlap_bars)))
    return (ts - delta).to_pydatetime()


def backfill_gap(
    symbol: str,
    start_dt: datetime,
    *,
    dry_run: bool,
) -> Tuple[int, Optional[str], Optional[str]]:
    ib = get_ib_connection()
    contract = create_ib_contract_with_primary_exchange(symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.error("%s: qualifyContracts returned empty", symbol)
        return 0, None, None
    contract = qualified[0]
    end_dt = datetime.now(timezone.utc)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    raw = _fetch_ib_intraday_batched(symbol, "15m", ib, contract, start_dt, end_dt)
    if raw is None or raw.empty:
        logger.warning("%s: no new 15m bars from IB", symbol)
        return 0, None, None
    df = prepare_nautilus_dataframe(raw, symbol, "IB", "15m")
    ts0 = _ns_ts(df, True)
    ts1 = _ns_ts(df, False)
    logger.info("%s: fetched %d 15m bars (%s -> %s)", symbol, len(df), ts0, ts1)
    if dry_run:
        logger.info("%s: dry-run, skip insert", symbol)
        return int(len(df)), ts0, ts1
    if not save_to_timescaledb(df, symbol, "IB", "15m"):
        logger.error("%s: TimescaleDB insert failed", symbol)
        return 0, ts0, ts1
    return int(len(df)), ts0, ts1


def _ns_ts(df: pd.DataFrame, first: bool) -> str:
    if "ts_event" not in df.columns:
        return "?"
    val = int(df["ts_event"].min() if first else df["ts_event"].max())
    return str(pd.Timestamp(val, unit="ns", tz="UTC"))


def write_inventory(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote coverage %s (%d symbols)", path, len(df))
    return path


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


def main(argv=None) -> int:
    args = parse_args(argv)
    stale_before = None
    if str(args.stale_before).strip():
        stale_before = datetime.strptime(str(args.stale_before).strip(), "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    logger.info("Loading IB 15m coverage from TimescaleDB ...")
    cov = load_ib_15m_coverage()
    summary = coverage_summary(cov)
    logger.info("Coverage: %s", json.dumps(summary))
    write_inventory(cov, args.inventory_out)
    if args.inventory:
        return 0 if not cov.empty else 1

    if cov.empty:
        logger.error("No IB 15m symbols in market_data")
        return 1

    wanted = None
    if args.symbols_file is not None:
        wanted = set(_symbols_from_file(args.symbols_file))
        cov = cov[cov["symbol"].isin(wanted)].copy()
        logger.info("Symbols-file filter: %d names", len(cov))

    if args.reset_resume and args.resume_file.exists():
        args.resume_file.unlink()
        logger.info("Cleared resume file %s", args.resume_file)
    done = load_resume(args.resume_file)

    now = datetime.now(timezone.utc)
    todo: List[Tuple[str, pd.Timestamp]] = []
    skipped_fresh = 0
    skipped_resume = 0
    for _, row in cov.iterrows():
        sym = str(row["symbol"]).upper()
        last_ts = row["last_ts"]
        if sym in done:
            skipped_resume += 1
            continue
        if not needs_backfill(
            last_ts, now=now, fresh_hours=float(args.fresh_hours), stale_before=stale_before
        ):
            skipped_fresh += 1
            continue
        todo.append((sym, pd.Timestamp(last_ts)))
    if args.limit and args.limit > 0:
        todo = todo[: int(args.limit)]
    logger.info(
        "Fetch queue %d | skipped_fresh=%d skipped_resume=%d fresh_hours=%.1f client_id=%s dry_run=%s",
        len(todo),
        skipped_fresh,
        skipped_resume,
        float(args.fresh_hours),
        args.ib_client_id,
        bool(args.dry_run),
    )
    if not todo:
        logger.info("Nothing to backfill")
        return 0

    set_ib_client_id(int(args.ib_client_id))
    n_ok = 0
    n_fail = 0
    t0 = time.perf_counter()
    try:
        for i, (sym, last_ts) in enumerate(todo, start=1):
            start_dt = rewind_start(last_ts, int(args.overlap_bars))
            logger.info(
                "[%d/%d] %s last_ts=%s fetch_from=%s",
                i,
                len(todo),
                sym,
                last_ts,
                start_dt.isoformat(),
            )
            try:
                n_bars, _, _ = backfill_gap(sym, start_dt, dry_run=bool(args.dry_run))
            except Exception as exc:
                logger.exception("%s: backfill failed: %s", sym, exc)
                n_fail += 1
                continue
            if n_bars >= 0:
                n_ok += 1
                if not args.dry_run:
                    append_resume(args.resume_file, sym)
            else:
                n_fail += 1
            if i < len(todo) and float(args.sleep) > 0:
                time.sleep(float(args.sleep))
    finally:
        cleanup_ib_connection()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Done ok=%d fail=%d elapsed_sec=%.1f queue=%d",
        n_ok,
        n_fail,
        elapsed,
        len(todo),
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
