"""Ingest Phase 6 macro ETFs into TimescaleDB via Alpaca (IB fallback)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import fetch_from_alpaca, fetch_from_ib, save_to_timescaledb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_p6")

SYMS = ["TLT", "GLD", "USO", "UUP", "IEF", "DBC"]


def main() -> int:
    ok = 0
    for sym in SYMS:
        df = None
        provider = "ALPACA"
        try:
            df = fetch_from_alpaca(sym, "max", "1d", start_date="2010-01-01")
        except Exception as exc:
            logger.warning("Alpaca %s failed: %s", sym, exc)
        if df is None or getattr(df, "empty", True):
            try:
                provider = "IB"
                df = fetch_from_ib(sym, "max", "1d", start_date="2010-01-01")
            except Exception as exc:
                logger.warning("IB %s failed: %s", sym, exc)
                df = None
        if df is None or getattr(df, "empty", True):
            logger.error("No data for %s", sym)
            continue
        logger.info("%s via %s rows=%d", sym, provider, len(df))
        if save_to_timescaledb(df, sym, provider, "1d"):
            ok += 1
    logger.info("Saved %d/%d symbols", ok, len(SYMS))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
