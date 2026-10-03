#!/usr/bin/env python3
"""Claim-check a 45 DTE SPX bull put credit spread on Theta EOD quotes.

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_spx_bull_put.py --variants baseline
  python scripts\\research\\claimcheck_spx_bull_put.py --marks mid --package-slip 0.05 --max-entries 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.spx_bull_put import (  # noqa: E402
    CLAIMED,
    book_stats,
    load_index_eod,
    parse_package_slip,
    run_bull_put_book,
    trades_to_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_spx_bull_put")

DEFAULT_OUT = ROOT / "reports" / "research" / "spx_bull_put"

# Shared-cache books. Slip variants replay the same mid quotes.
VARIANT_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "baseline": [
        {"marks": "mid", "package_slip": 0.05, "use_stop": True, "max_concurrent": 1},
        {"marks": "mid", "package_slip": 0.0, "use_stop": True, "max_concurrent": 1},
        {"marks": "mid", "package_slip": 0.10, "use_stop": True, "max_concurrent": 1},
        {"marks": "mid", "package_slip": 0.15, "use_stop": True, "max_concurrent": 1},
        {"marks": "ask_bid", "package_slip": 0.0, "use_stop": True, "max_concurrent": 1},
        {"marks": "mid", "package_slip": 0.05, "use_stop": False, "max_concurrent": 1},
        {"marks": "mid", "package_slip": 0.05, "use_stop": True, "max_concurrent": 4},
    ],
}


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _tag(cfg: Dict[str, Any]) -> str:
    mark = str(cfg["marks"])
    slip = parse_package_slip(mark, float(cfg.get("package_slip") or 0.0))
    base = "ask_bid" if mark in ("ask_bid", "natural", "askbid") else "mid"
    if slip > 0 and base == "mid":
        base = f"mid_slip{slip:g}"
    stop = "sl2x" if cfg.get("use_stop", True) else "nostop"
    parts = [base, stop]
    scan = str(cfg.get("scan") or "monday")
    if scan != "monday":
        parts.append(scan)
    conc = int(cfg.get("max_concurrent") or 1)
    if conc > 1:
        parts.append(f"c{conc}")
    return "_".join(parts)


def _fmt(label: str, s: Dict[str, Any]) -> str:
    return (
        f"{label}: n={s.get('n')} WR={s.get('win_rate_pct')}% "
        f"meanRoR={s.get('mean_ror_pct')}% medRoR={s.get('med_ror_pct')}% "
        f"PF={s.get('pf')} PnL=${s.get('total_pnl')} "
        f"Sharpe~={s.get('sharpe_proxy')} avgHold={s.get('avg_hold')}d "
        f"avgCredit={s.get('avg_credit')} medCross={s.get('med_cross')} "
        f"MDD={s.get('mdd_pct_1lot')}% exits={s.get('exit_reasons')} years={s.get('years')}"
    )


def _count_reasons(skipped: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in skipped:
        r = str(s.get("reason") or "unknown")
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-08-29")
    ap.add_argument("--underlier", default="SPX")
    ap.add_argument("--opt-symbol", default="SPXW", help="SPXW Friday weeklies, or SPX for monthlies")
    ap.add_argument("--marks", default="mid")
    ap.add_argument("--width", type=float, default=25.0)
    ap.add_argument("--dte", type=int, default=45)
    ap.add_argument("--dte-tol", type=int, default=5)
    ap.add_argument("--min-dte-exit", type=int, default=21)
    ap.add_argument("--target-delta", type=float, default=0.22)
    ap.add_argument("--min-credit", type=float, default=1.0)
    ap.add_argument("--package-slip", type=float, default=0.05)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--scan", default="monday", choices=["monday", "daily"])
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--max-entries", type=int, default=0, help="Stop after N fills (0 = no cap)")
    ap.add_argument("--variants", default="", help="Named suite: baseline")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if args.variants:
        suite = VARIANT_SUITES.get(args.variants.strip().lower())
        if not suite:
            LOG.error("Unknown --variants %s; choose from %s", args.variants, list(VARIANT_SUITES))
            return 1
        configs = suite
    else:
        marks = [m.strip() for m in args.marks.split(",") if m.strip()]
        configs = [
            {
                "marks": m,
                "package_slip": args.package_slip,
                "use_stop": not args.no_stop,
                "scan": args.scan,
                "max_concurrent": args.max_concurrent,
            }
            for m in marks
        ]

    outdir = args.outdir or (DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d"))
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not ready at %s", client.base_url)
        return 1

    LOG.info("Loading spot %s %s..%s", args.underlier, start, end)
    spots = load_index_eod(client, args.underlier, start - timedelta(days=10), end)
    LOG.info("Spot rows %s", 0 if spots is None else len(spots))
    LOG.info("Loading %s expirations", args.opt_symbol)
    exps = list_expirations(client, args.opt_symbol)
    LOG.info("Expirations %s", len(exps))
    cache: Dict[Tuple[Any, ...], Any] = {}
    strike_cache: Dict[date, List[float]] = {}

    t0 = time.time()
    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "underlier": args.underlier,
        "opt_symbol": args.opt_symbol,
        "structure": {
            "width": args.width,
            "dte": args.dte,
            "dte_tol": args.dte_tol,
            "target_delta": args.target_delta,
            "min_dte_exit": args.min_dte_exit,
            "min_credit": args.min_credit,
            "tp_frac": 0.50,
            "sl_mult": 2.0,
            "commission_per_leg": 0.65,
            "legs_round_turn": 4,
        },
        "variants": args.variants or None,
        "books": {},
        "wall_s": None,
        "notes": [
            "EOD marks only: a 50% or 2x touch that reverts before the close is missed.",
            "Stop price is 2x fill credit (loss about 1x credit), not a 2x-credit loss.",
            "1-lot equity starts at $100k and adds closed-trade P&L. Not a Sharpe-vs-SPY book.",
            "medCross is half the two bid-ask widths at entry (cost to cross vs mid).",
        ],
    }

    for cfg in configs:
        tag = _tag(cfg)
        LOG.info("=== book %s ===", tag)
        book = run_bull_put_book(
            start=start,
            end=end,
            underlier=args.underlier,
            opt_symbol=args.opt_symbol,
            mark_mode=str(cfg["marks"]),
            width=args.width,
            target_dte=args.dte,
            dte_tol=args.dte_tol,
            min_dte_exit=args.min_dte_exit,
            target_delta=args.target_delta,
            min_credit=args.min_credit,
            use_stop=bool(cfg.get("use_stop", True)),
            package_slip=float(cfg.get("package_slip") or 0.0),
            scan=str(cfg.get("scan") or "monday"),
            max_concurrent=int(cfg.get("max_concurrent") or 1),
            max_entries=int(args.max_entries),
            client=client,
            spots=spots,
            exps=exps,
            cache=cache,
            strike_cache=strike_cache,
        )
        stats = book_stats(book.trades)
        summary["books"][tag] = {
            "stats": stats,
            "n_skipped": len(book.skipped),
            "skip_reasons": _count_reasons(book.skipped),
            "label": book.label,
            "config": {
                "mark": cfg["marks"],
                "package_slip": float(cfg.get("package_slip") or 0.0),
                "use_stop": bool(cfg.get("use_stop", True)),
                "scan": str(cfg.get("scan") or "monday"),
                "max_concurrent": int(cfg.get("max_concurrent") or 1),
            },
        }
        LOG.info(_fmt(tag, stats))
        stem = f"{args.opt_symbol}_w{int(args.width)}_{tag}"
        trades_to_frame(book.trades).to_csv(outdir / f"{stem}_trades.csv", index=False)
        with (outdir / f"{stem}_skipped.json").open("w", encoding="utf-8") as f:
            json.dump(book.skipped[:8000], f, indent=2, default=str)

    summary["wall_s"] = round(time.time() - t0, 1)
    out_name = "summary_baseline.json" if args.variants == "baseline" else "summary.json"
    with (outdir / out_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOG.info("Wrote %s (wall %.1fs)", outdir, summary["wall_s"])
    _print_verdict(summary)
    return 0


def _print_verdict(summary: Dict[str, Any]) -> None:
    print("\n=== 45 DTE SPX bull put vs sketch ===")
    print(
        "Rules: Friday 40-50 DTE, ~22 delta short, 25-pt wing, "
        "TP 50% credit, SL debit>=2x credit, time-stop 21 DTE, $0.65 x 4 legs."
    )
    for tag, body in summary.get("books", {}).items():
        s = body.get("stats") or {}
        print(_fmt(tag, s), f"| skips={body.get('n_skipped')}")
    print(f"wall_s={summary.get('wall_s')}")


if __name__ == "__main__":
    raise SystemExit(main())
