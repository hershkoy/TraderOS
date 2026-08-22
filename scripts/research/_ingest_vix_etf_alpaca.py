"""Ingest VIX ETF proxies via Alpaca (IB index/HMDS unavailable)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import fetch_from_alpaca, save_to_timescaledb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_vix_etf")

SYMS = ["VIXY", "VXX", "VXZ", "VIXM", "UVXY", "SVXY"]


def main() -> int:
    ok = 0
    for sym in SYMS:
        try:
            df = fetch_from_alpaca(sym, "max", "1d", start_date="2010-01-01")
        except Exception as exc:
            logger.error("%s alpaca fetch failed: %s", sym, exc)
            continue
        if df is None or getattr(df, "empty", True):
            logger.error("No data %s", sym)
            continue
        logger.info("%s rows=%d", sym, len(df))
        if save_to_timescaledb(df, sym, "ALPACA", "1d"):
            ok += 1
            logger.info("Saved %s", sym)
    logger.info("Saved %d/%d", ok, len(SYMS))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
