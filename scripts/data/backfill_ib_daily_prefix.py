#!/usr/bin/env python3
"""
Backfill IB daily (1d) history as a prefix for ALPACA symbols that start late.

Writes provider='IB' rows into TimescaleDB (does NOT overwrite ALPACA).
Research loaders can then stitch with:
  load_ohlcv_many(..., provider='ALPACA', fallback_provider='IB', merge_mode='prefix')

Requires IB Gateway on 127.0.0.1:4001.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_daily_prefix.py --limit 5 --start 2018-11-01
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_daily_prefix.py --start 2018-11-01 --workers 1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import cleanup_ib_connection, fetch_max_from_ib  # noqa: E402
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_ib_daily_prefix")
# Quiet noisy loggers
logging.getLogger("utils.data.fetch_data").setLevel(logging.WARNING)


def list_symbols_needing_prefix(
    *,
    target_start: datetime,
    min_alpaca_start: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Tuple[str, datetime]]:
    """
    Return (symbol, alpaca_min_ts) for ALPACA 1d names whose history starts after target_start.
    Prefer symbols that have little/no IB 1d before alpaca_min.
    """
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB connection failed")
    sql = """
        WITH alp AS (
            SELECT symbol, MIN(ts) AS amin
            FROM market_data
            WHERE provider = 'ALPACA' AND timeframe = '1d'
            GROUP BY symbol
            HAVING MIN(ts) > %s
        ),
        ib AS (
            SELECT symbol, MIN(ts) AS imin
            FROM market_data
            WHERE provider = 'IB' AND timeframe = '1d'
            GROUP BY symbol
        )
        SELECT a.symbol, a.amin
        FROM alp a
        LEFT JOIN ib i ON i.symbol = a.symbol
        WHERE i.symbol IS NULL OR i.imin > %s
        ORDER BY a.amin ASC, a.symbol ASC
    """
    cur = client.connection.cursor()
    try:
        cur.execute(sql, (target_start, target_start))
        rows = cur.fetchall()
    finally:
        cur.close()
    if min_alpaca_start is not None:
        rows = [r for r in rows if r[1] >= min_alpaca_start]
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    return [(str(s), ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts) for s, ts in rows]


def backfill_symbol(symbol: str, start: datetime, end: Optional[datetime] = None) -> int:
    """Fetch IB daily from start and insert. Returns bars saved."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB connection failed")

    df = fetch_max_from_ib(symbol, "1d", start_date=start)
    if df is None or df.empty:
        logger.warning("%s: no IB daily bars", symbol)
        return 0

    idx = df.index
    if getattr(idx, "tz", None) is not None:
        df = df.copy()
        df.index = idx.tz_convert("UTC").tz_localize(None)
    if end is not None:
        df = df.loc[df.index <= end]
    df = df.loc[df.index >= start]
    if df.empty:
        logger.warning("%s: IB bars outside window", symbol)
        return 0

    ok = client.insert_market_data(df, symbol, "IB", "1d")
    if not ok:
        logger.error("%s: insert_market_data failed (%d bars)", symbol, len(df))
        return 0
    logger.info("%s: saved %d IB 1d bars (%s -> %s)", symbol, len(df), df.index.min().date(), df.index.max().date())
    return int(len(df))


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill IB 1d prefix for late-starting ALPACA symbols")
    ap.add_argument("--start", default="2018-11-01", help="Target history start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Optional end date for IB fetch window")
    ap.add_argument("--limit", type=int, default=None, help="Max symbols (smoke test)")
    ap.add_argument("--symbols", default="", help="Comma list override")
    ap.add_argument("--sleep", type=float, default=1.0, help="Pause between IB symbols (pacing)")
    ap.add_argument("--dry-run", action="store_true", help="List symbols only")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    if args.symbols.strip():
        needed = [(s.strip().upper(), start) for s in args.symbols.split(",") if s.strip()]
    else:
        logger.info("Finding ALPACA 1d symbols starting after %s ...", args.start)
        t0 = time.perf_counter()
        needed = list_symbols_needing_prefix(target_start=start, limit=args.limit)
        logger.info("Found %d symbols needing IB prefix (%.1fs)", len(needed), time.perf_counter() - t0)

    if not needed:
        print("No symbols need IB prefix backfill")
        return 0

    print(f"Will process {len(needed)} symbols (dry_run={args.dry_run})")
    for s, amin in needed[:10]:
        print(f"  {s} alpaca_min={amin}")
    if len(needed) > 10:
        print(f"  ... +{len(needed)-10} more")

    if args.dry_run:
        return 0

    saved = 0
    failed = 0
    bars_total = 0
    try:
        for i, (sym, _amin) in enumerate(needed, start=1):
            try:
                n = backfill_symbol(sym, start, end=end)
                if n > 0:
                    saved += 1
                    bars_total += n
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                logger.error("%s failed: %s", sym, exc)
            if args.sleep and i < len(needed):
                time.sleep(float(args.sleep))
            if i % 25 == 0:
                logger.info("Progress %d/%d saved=%d failed=%d bars=%d", i, len(needed), saved, failed, bars_total)
    finally:
        try:
            cleanup_ib_connection()
        except Exception:
            pass

    print(f"\nDone: saved_symbols={saved} failed={failed} bars={bars_total}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
