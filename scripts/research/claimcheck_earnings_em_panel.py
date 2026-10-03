#!/usr/bin/env python3
"""Phase 2: walk-forward expected-move PCS / iron condor on liquid large caps.

Reads Phase 1 inventory (causal wf_structure). Prices only the next report
when the trailing screen assigns pcs or ic. Default holdout prices 2025-01-01
forward so 2023-24 is the trail window, not the P&L sample.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\scan_earnings_em_tickers.py
  python scripts\\research\\claimcheck_earnings_em_panel.py
  python scripts\\research\\claimcheck_earnings_em_panel.py --price-start 2025-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.earnings_em import (  # noqa: E402
    DEFAULT_UNIVERSE,
    attach_trailing,
    book_stats,
    load_quote_cache,
    load_stock,
    price_structure,
    save_quote_cache,
    yahoo_splits,
)
from utils.research.forward_factor import list_expirations  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_earnings_em_panel")

DEFAULT_OUT = ROOT / "reports" / "research" / "earnings_em"


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _latest_inventory(outdir_root: Path) -> Optional[Path]:
    dated = sorted(
        [p for p in outdir_root.glob("*/inventory.csv") if p.is_file()],
        reverse=True,
    )
    return dated[0] if dated else None


def _subset(rows: Sequence[Dict[str, Any]], structure: str) -> Dict[str, Any]:
    slim = [r for r in rows if r.get("wf_structure") == structure and r.get("net_mid") is not None]
    out = {
        "mid": book_stats(slim, "net_mid"),
        "mid_slip0.05": book_stats(slim, "net_mid_slip0.05"),
        "mid_slip0.1": book_stats(slim, "net_mid_slip0.1"),
        "natural": book_stats(slim, "net_natural"),
    }
    held = [r for r in slim if r.get("spot_held") is not None]
    if held:
        out["spot_held"] = "%d/%d" % (sum(1 for r in held if r.get("spot_held")), len(held))
    credits = [float(r["credit_mid"]) for r in slim if r.get("credit_mid") is not None]
    if credits:
        out["avg_credit"] = round(sum(credits) / len(credits), 3)
    return out


def _fmt(label: str, stats: Dict[str, Any]) -> str:
    mid = stats.get("mid") or {}
    nat = stats.get("natural") or {}
    slip = stats.get("mid_slip0.1") or {}
    return (
        "%s: n=%s spot_held=%s avgCredit=%s | mid WR=%s%% mean$=%s PF=%s meanRoR=%s total$=%s | "
        "slip10 WR=%s%% total$=%s PF=%s | natural WR=%s%% mean$=%s PF=%s total$=%s"
        % (
            label,
            mid.get("n"),
            stats.get("spot_held"),
            stats.get("avg_credit"),
            mid.get("win_rate_pct"),
            mid.get("mean_pnl"),
            mid.get("pf"),
            mid.get("mean_ror_pct"),
            mid.get("total_pnl"),
            slip.get("win_rate_pct"),
            slip.get("total_pnl"),
            slip.get("pf"),
            nat.get("win_rate_pct"),
            nat.get("mean_pnl"),
            nat.get("pf"),
            nat.get("total_pnl"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="")
    parser.add_argument("--price-start", default="2025-01-01")
    parser.add_argument("--price-end", default="2026-09-01")
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()
    price_start = _parse_date(args.price_start)
    price_end = _parse_date(args.price_end)
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT / date.today().isoformat()
    outdir.mkdir(parents=True, exist_ok=True)
    inv_path = Path(args.inventory) if args.inventory else _latest_inventory(DEFAULT_OUT)
    if inv_path is None or not inv_path.exists():
        LOG.error("no inventory.csv; run scan_earnings_em_tickers.py first")
        return 1

    t0 = time.perf_counter()
    raw = pd.read_csv(inv_path)
    if raw.empty:
        LOG.error("empty inventory %s", inv_path)
        return 1
    records = raw.to_dict(orient="records")
    for rec in records:
        rec["amc"] = bool(rec.get("amc"))
        rec["inside_em"] = bool(rec.get("inside_em"))
        rec["broke_down"] = bool(rec.get("broke_down"))
        rec["opened_green"] = bool(rec.get("opened_green"))
        rec["broke_up"] = bool(rec.get("broke_up"))
        rec["earn_date"] = str(rec.get("earn_date"))[:10]
    tagged = attach_trailing(records)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal is not up on %s", client.base_url)
        return 1
    cache_path = DEFAULT_OUT / "quote_cache.json"
    cache = load_quote_cache(cache_path)

    candidates = [
        r
        for r in tagged
        if r.get("wf_structure") in ("pcs", "ic")
        and price_start <= date.fromisoformat(r["earn_date"]) <= price_end
    ]
    LOG.info("inventory %s candidates %d (price %s -> %s)", inv_path, len(candidates), price_start, price_end)

    stock_cache: Dict[str, pd.DataFrame] = {}
    exp_cache: Dict[str, List] = {}
    priced: List[Dict[str, Any]] = []
    missing = 0
    for row in candidates:
        sym = str(row["symbol"]).upper()
        if sym not in stock_cache:
            splits = yahoo_splits(sym)
            start = date.fromisoformat(min(r["earn_date"] for r in tagged if r["symbol"] == sym))
            stock_cache[sym] = load_stock(sym, start - timedelta(days=10), price_end + timedelta(days=7), splits=splits)
            exp_cache[sym] = list_expirations(client, sym)
        try:
            got = price_structure(
                client,
                cache,
                symbol=sym,
                inventory=row,
                structure=str(row["wf_structure"]),
                stock=stock_cache[sym],
                expirations=exp_cache[sym],
            )
        except Exception as exc:
            LOG.exception("%s %s price failed: %s", sym, row["earn_date"], exc)
            got = None
        save_quote_cache(cache_path, cache)
        if got is None or got.get("missing_quote"):
            missing += 1
            LOG.info("%s %s %s missing quotes", sym, row["earn_date"], row["wf_structure"])
            continue
        priced.append(got)
        LOG.info(
            "%s %s %s mid=$%s nat=$%s held=%s",
            sym,
            row["earn_date"],
            row["wf_structure"],
            got.get("net_mid"),
            got.get("net_natural"),
            got.get("spot_held"),
        )

    frame = pd.DataFrame(priced)
    frame.to_csv(outdir / "wf_trades.csv", index=False)
    pcs = _subset(priced, "pcs")
    ic = _subset(priced, "ic")
    all_rows = {
        "mid": book_stats(priced, "net_mid"),
        "mid_slip0.1": book_stats(priced, "net_mid_slip0.1"),
        "natural": book_stats(priced, "net_natural"),
    }
    summary: Dict[str, Any] = {
        "inventory": str(inv_path),
        "price_start": price_start.isoformat(),
        "price_end": price_end.isoformat(),
        "universe": list(DEFAULT_UNIVERSE),
        "candidates": len(candidates),
        "priced": len(priced),
        "missing_quote": missing,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "pcs": pcs,
        "ic": ic,
        "all": all_rows,
        "by_symbol": {},
    }
    for sym in sorted({r["symbol"] for r in priced}):
        subset = [r for r in priced if r["symbol"] == sym]
        summary["by_symbol"][sym] = {
            "n": len(subset),
            "pcs": sum(1 for r in subset if r.get("wf_structure") == "pcs"),
            "ic": sum(1 for r in subset if r.get("wf_structure") == "ic"),
            "mid": book_stats(subset, "net_mid"),
            "natural": book_stats(subset, "net_natural"),
        }
    (outdir / "wf_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info(
        "ALL %s",
        _fmt(
            "wf",
            {
                "mid": all_rows["mid"],
                "mid_slip0.1": all_rows["mid_slip0.1"],
                "natural": all_rows["natural"],
            },
        ),
    )
    if (pcs.get("mid") or {}).get("n"):
        LOG.info(_fmt("PCS", pcs))
    if (ic.get("mid") or {}).get("n"):
        LOG.info(_fmt("IC", ic))
    LOG.info("wrote %s in %.1fs", outdir, summary["wall_clock_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
