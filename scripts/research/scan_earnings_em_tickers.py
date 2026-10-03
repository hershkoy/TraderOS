#!/usr/bin/env python3
"""Phase 1: ATM 15:45 expected-move containment for liquid large caps.

Stock facts only (inside band / broke downside / opened green). Trailing
stats and the walk-forward structure are causal (prior reports of that name).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\scan_earnings_em_tickers.py
  python scripts\\research\\scan_earnings_em_tickers.py --symbols NVDA,AAPL --start 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.earnings_em import (  # noqa: E402
    DEFAULT_UNIVERSE,
    attach_trailing,
    fetch_earnings,
    inventory_event,
    load_quote_cache,
    load_stock,
    save_quote_cache,
    yahoo_splits,
)
from utils.research.forward_factor import list_expirations  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("scan_earnings_em_tickers")

DEFAULT_OUT = ROOT / "reports" / "research" / "earnings_em"


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _symbol_blurb(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "amc": sum(1 for r in rows if r.get("amc")),
        "bmo": sum(1 for r in rows if not r.get("amc")),
        "inside_em": sum(1 for r in rows if r.get("inside_em")),
        "broke_down": sum(1 for r in rows if r.get("broke_down")),
        "opened_green": sum(1 for r in rows if r.get("opened_green")),
        "broke_up": sum(1 for r in rows if r.get("broke_up")),
        "wf_pcs": sum(1 for r in rows if r.get("wf_structure") == "pcs"),
        "wf_ic": sum(1 for r in rows if r.get("wf_structure") == "ic"),
        "wf_skip": sum(1 for r in rows if r.get("wf_structure") == "skip"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--start", default="2023-02-01")
    parser.add_argument("--end", default="2026-09-01")
    parser.add_argument("--limit-earn", type=int, default=24)
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT / date.today().isoformat()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = DEFAULT_OUT / "quote_cache.json"
    cache = load_quote_cache(cache_path)

    t0 = time.perf_counter()
    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal is not up on %s", client.base_url)
        return 1

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for sym in symbols:
        events = [
            e
            for e in fetch_earnings(sym, limit=args.limit_earn)
            if start <= e["earn_date"] <= end
        ]
        LOG.info("%s earnings %d", sym, len(events))
        if not events:
            continue
        splits = yahoo_splits(sym)
        LOG.info("%s splits %s", sym, splits)
        stock = load_stock(sym, start, end + timedelta(days=7), splits=splits)
        LOG.info("%s sessions %d", sym, len(stock))
        expirations = list_expirations(client, sym)
        LOG.info("%s expirations %d", sym, len(expirations))
        for event in events:
            try:
                row = inventory_event(
                    client,
                    cache,
                    symbol=sym,
                    event=event,
                    stock=stock,
                    expirations=expirations,
                )
            except Exception as exc:
                LOG.exception("%s %s failed: %s", sym, event["earn_date"], exc)
                row = None
            if row is None:
                skipped[sym] = skipped.get(sym, 0) + 1
            else:
                rows.append(row)
            save_quote_cache(cache_path, cache)

    tagged = attach_trailing(rows)
    import pandas as pd

    frame = pd.DataFrame(tagged)
    frame.to_csv(outdir / "inventory.csv", index=False)
    by_sym = {}
    for row in tagged:
        by_sym.setdefault(row["symbol"], []).append(row)
    summary: Dict[str, Any] = {
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "n": len(tagged),
        "symbols": symbols,
        "skipped": skipped,
        "by_symbol": {s: _symbol_blurb(by_sym.get(s, [])) for s in symbols},
        "wf_pcs": sum(1 for r in tagged if r.get("wf_structure") == "pcs"),
        "wf_ic": sum(1 for r in tagged if r.get("wf_structure") == "ic"),
        "wf_skip": sum(1 for r in tagged if r.get("wf_structure") == "skip"),
    }
    (outdir / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    LOG.info(
        "inventory n=%d pcs=%s ic=%s skip=%s in %.1fs -> %s",
        summary["n"],
        summary["wf_pcs"],
        summary["wf_ic"],
        summary["wf_skip"],
        summary["wall_clock_s"],
        outdir,
    )
    for s in symbols:
        LOG.info("  %s %s", s, summary["by_symbol"][s])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
