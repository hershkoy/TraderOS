#!/usr/bin/env python3
"""
Backfill IB 5m bars for named symbols into TimescaleDB.

Dedicated Gateway client ID so this can run next to another IB historical
session. Prefer the universe script for the full 15m name list.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_5m_symbol.py --symbols SPY --ib-client-id 8824 --since 2018-01-01
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_ib_5m_universe import (  # noqa: E402
    StopRequested,
    backfill_symbol_forward,
)
from utils.data.fetch_data import cleanup_ib_connection, set_ib_client_id  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_ib_5m_symbol")
logging.getLogger("ib_insync").setLevel(logging.WARNING)

DEFAULT_CLIENT_ID = 8824


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill IB 5m OHLCV into TimescaleDB")
    ap.add_argument("--symbols", default="SPY", help="Comma list (default SPY)")
    ap.add_argument("--since", default="2018-01-01", help="UTC start date YYYY-MM-DD")
    ap.add_argument(
        "--ib-client-id",
        type=int,
        default=DEFAULT_CLIENT_ID,
        help="Gateway client ID (default 8824; keep distinct from 8822/8823/8826)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Qualify + fetch, do not insert")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between IB windows")
    ap.add_argument("--batch-days", type=int, default=7, help="IB window length (default 7)")
    ap.add_argument("--ib-port", type=int, default=4001)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("IB_PORT", str(int(args.ib_port)))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols")
        return 1
    start_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    set_ib_client_id(int(args.ib_client_id))
    logger.info(
        "IB 5m backfill symbols=%s since=%s client_id=%s dry_run=%s",
        symbols,
        args.since,
        args.ib_client_id,
        bool(args.dry_run),
    )
    n_ok = 0
    try:
        for sym in symbols:
            try:
                n, status = backfill_symbol_forward(
                    sym,
                    start_dt,
                    datetime.now(timezone.utc),
                    batch_days=int(args.batch_days),
                    sleep_s=float(args.sleep),
                    dry_run=bool(args.dry_run),
                    should_stop=lambda: False,
                )
            except StopRequested:
                logger.info("%s: stopped", sym)
                break
            if status in {"ok", "caught_up"} and n >= 0:
                n_ok += 1
                logger.info("%s: %s bars=%d", sym, status, n)
            else:
                logger.error("%s: status=%s bars=%d", sym, status, n)
    finally:
        cleanup_ib_connection()
    logger.info("Done: %d/%d symbols saved", n_ok, len(symbols))
    return 0 if n_ok == len(symbols) else 1


if __name__ == "__main__":
    raise SystemExit(main())
