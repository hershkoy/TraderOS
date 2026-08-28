#!/usr/bin/env python3
"""
Backfill IB 15m bars for named symbols into TimescaleDB.

Uses yearly IB batches (``_fetch_ib_intraday_batched``) and a dedicated Gateway
client ID so this can run next to another IB historical session.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_15m_symbol.py --symbols SPY --ib-client-id 8821 --since 2018-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_ib_15m_symbol")
logging.getLogger("ib_insync").setLevel(logging.WARNING)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill IB 15m OHLCV into TimescaleDB")
    ap.add_argument("--symbols", default="SPY", help="Comma list (default SPY)")
    ap.add_argument("--since", default="2018-01-01", help="UTC start date YYYY-MM-DD")
    ap.add_argument(
        "--ib-client-id",
        type=int,
        default=8821,
        help="Gateway client ID (default 8821; keep distinct from other IB sessions)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Qualify + fetch, do not insert")
    return ap.parse_args(argv)


def backfill_one(symbol: str, start_dt: datetime, *, dry_run: bool) -> int:
    ib = get_ib_connection()
    contract = create_ib_contract_with_primary_exchange(symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.error("%s: qualifyContracts returned empty", symbol)
        return 0
    contract = qualified[0]
    logger.info(
        "%s: qualified conId=%s primaryExchange=%s",
        symbol,
        contract.conId,
        contract.primaryExchange,
    )
    end_dt = datetime.now(timezone.utc)
    raw = _fetch_ib_intraday_batched(symbol, "15m", ib, contract, start_dt, end_dt)
    if raw is None or raw.empty:
        logger.error("%s: no 15m bars from IB", symbol)
        return 0
    df = prepare_nautilus_dataframe(raw, symbol, "IB", "15m")
    ts0 = pd_min_ts(df)
    ts1 = pd_max_ts(df)
    logger.info("%s: fetched %d 15m bars (%s -> %s)", symbol, len(df), ts0, ts1)
    if dry_run:
        logger.info("%s: dry-run, skip insert", symbol)
        return int(len(df))
    if not save_to_timescaledb(df, symbol, "IB", "15m"):
        logger.error("%s: TimescaleDB insert failed", symbol)
        return 0
    return int(len(df))


def pd_min_ts(df) -> str:
    import pandas as pd

    if "ts_event" not in df.columns:
        return "?"
    return str(pd.Timestamp(int(df["ts_event"].min()), unit="ns", tz="UTC"))


def pd_max_ts(df) -> str:
    import pandas as pd

    if "ts_event" not in df.columns:
        return "?"
    return str(pd.Timestamp(int(df["ts_event"].max()), unit="ns", tz="UTC"))


def main(argv=None) -> int:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols")
        return 1
    start_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    set_ib_client_id(int(args.ib_client_id))
    logger.info(
        "IB 15m backfill symbols=%s since=%s client_id=%s dry_run=%s",
        symbols,
        args.since,
        args.ib_client_id,
        bool(args.dry_run),
    )
    n_ok = 0
    try:
        for sym in symbols:
            n = backfill_one(sym, start_dt, dry_run=bool(args.dry_run))
            if n > 0:
                n_ok += 1
    finally:
        cleanup_ib_connection()
    logger.info("Done: %d/%d symbols saved", n_ok, len(symbols))
    return 0 if n_ok == len(symbols) else 1


if __name__ == "__main__":
    raise SystemExit(main())
