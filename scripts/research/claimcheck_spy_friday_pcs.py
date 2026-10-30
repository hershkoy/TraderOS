#!/usr/bin/env python3
"""Claim-check Nathan's SPY Friday 7 DTE put credit spread (HbV4sTEyoBM).

Data: OptionsDX SPY EOD through 2023-12-29, Theta from 2024-01-02.
Requires Theta Terminal on :25503 for the post-2023 window:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_spy_friday_pcs.py --variants claim
  python scripts\\research\\claimcheck_spy_friday_pcs.py --start 2016-01-01 --end 2023-12-29 --no-theta
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.spy_friday_pcs import (  # noqa: E402
    CLAIMED,
    book_stats,
    make_hybrid_source,
    phase5_summary,
    run_friday_pcs_book,
    trades_to_frame,
)
from utils.research.spy_wheel import OptionsDXSource  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_spy_friday_pcs")

DEFAULT_OUT = ROOT / "reports" / "research" / "spy_friday_pcs"

VARIANT_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "claim": [
        {
            "marks": "mid",
            "tp_profit_frac": 0.25,
            "sl_mult": 4.0,
            "label": "claim_tp25_sl300",
        },
        {
            "marks": "mid",
            "tp_profit_frac": 0.50,
            "sl_mult": 4.0,
            "label": "claim_tp50_sl300",
        },
        {
            "marks": "mid",
            "tp_profit_frac": 0.25,
            "sl_mult": 4.0,
            "use_stop": False,
            "label": "tp25_nostop",
        },
        {
            "marks": "ask_bid",
            "tp_profit_frac": 0.25,
            "sl_mult": 4.0,
            "label": "natural_tp25_sl300",
        },
        {
            "marks": "ask_bid",
            "tp_profit_frac": 0.50,
            "sl_mult": 4.0,
            "label": "natural_tp50_sl300",
        },
    ],
    "tp50_slip": [
        {
            "marks": "mid",
            "package_slip": 0.0,
            "tp_profit_frac": 0.50,
            "sl_mult": 4.0,
            "label": "tp50_mid",
        },
        {
            "marks": "mid",
            "package_slip": 0.05,
            "tp_profit_frac": 0.50,
            "sl_mult": 4.0,
            "label": "tp50_mid_slip05",
        },
        {
            "marks": "mid",
            "package_slip": 0.10,
            "tp_profit_frac": 0.50,
            "sl_mult": 4.0,
            "label": "tp50_mid_slip10",
        },
        {
            "marks": "ask_bid",
            "package_slip": 0.0,
            "tp_profit_frac": 0.50,
            "sl_mult": 4.0,
            "label": "tp50_natural",
        },
    ],
    "odx_only": [
        {
            "marks": "mid",
            "tp_profit_frac": 0.25,
            "sl_mult": 4.0,
            "label": "odx_tp25_sl300",
        },
        {
            "marks": "mid",
            "tp_profit_frac": 0.50,
            "sl_mult": 4.0,
            "label": "odx_tp50_sl300",
        },
    ],
}


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt(label: str, s: Dict[str, Any]) -> str:
    return (
        f"{label}: n={s.get('n')} WR={s.get('win_rate_pct')}% "
        f"meanRoR={s.get('mean_ror_pct')}% PF={s.get('pf')} "
        f"PnL=${s.get('total_pnl')} RoC={s.get('return_on_capital_pct')}% "
        f"CAGR={s.get('cagr')} MDD={s.get('mdd_pct')}% Sharpe={s.get('sharpe')} "
        f"avgCredit={s.get('avg_credit')} avgDelta={s.get('avg_delta')} "
        f"exits={s.get('exit_reasons')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--marks", default="mid")
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--dte", type=int, default=7)
    ap.add_argument("--dte-tol", type=int, default=2)
    ap.add_argument("--target-delta", type=float, default=0.40)
    ap.add_argument("--tp-profit-frac", type=float, default=0.25)
    ap.add_argument(
        "--sl-mult",
        type=float,
        default=4.0,
        help="Close when debit >= sl_mult * credit (4.0 = 300pct of credit lost)",
    )
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--min-credit", type=float, default=0.05)
    ap.add_argument("--margin-buffer", type=float, default=1.5)
    ap.add_argument(
        "--variants",
        default="claim",
        help="Named suite: claim, tp50_slip, odx_only, or empty",
    )
    ap.add_argument(
        "--package-slip",
        type=float,
        default=0.0,
        help="Mid package slip in points (clamped to NBBO); ignored for ask_bid",
    )
    ap.add_argument("--no-theta", action="store_true", help="OptionsDX only (clip end to 2023-12-29)")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "cache" / "spy_friday_pcs_theta",
    )
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if args.no_theta and end > date(2023, 12, 29):
        end = date(2023, 12, 29)
        LOG.info("Clipping end to OptionsDX window %s", end)

    if args.variants:
        suite = VARIANT_SUITES.get(args.variants.strip().lower())
        if not suite:
            LOG.error("Unknown --variants %s; choose from %s", args.variants, list(VARIANT_SUITES))
            return 1
        configs = suite
    else:
        configs = [
            {
                "marks": m.strip(),
                "tp_profit_frac": args.tp_profit_frac,
                "sl_mult": args.sl_mult,
                "use_stop": not args.no_stop,
            }
            for m in args.marks.split(",")
            if m.strip()
        ]

    outdir = args.outdir or (DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d"))
    outdir.mkdir(parents=True, exist_ok=True)

    if args.no_theta or args.variants.strip().lower() == "odx_only":
        source = OptionsDXSource()
        LOG.info("Quote source: OptionsDX only")
    else:
        source = make_hybrid_source(cache_dir=args.cache_dir)
        LOG.info("Quote source: OptionsDX + Theta (cache %s)", args.cache_dir)

    t0 = time.time()
    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "structure": {
            "width": args.width,
            "dte": args.dte,
            "dte_tol": args.dte_tol,
            "target_delta": args.target_delta,
            "min_credit": args.min_credit,
            "margin_buffer": args.margin_buffer,
            "commission": 0.0,
        },
        "variants": args.variants or None,
        "books": {},
        "wall_s": None,
        "notes": [
            "EOD marks only: intraday TP/SL touches that revert before the close are missed.",
            "SL mult 4.0 = debit 4x credit = lose 300pct of credit (video 'stop loss at 300pct').",
            "TP profit frac 0.25 = buy back at 75pct of credit.",
            "Capital = peak defined risk (width-credit)*100 * margin_buffer (video 1.5x peak margin).",
            "OptionsDX covers through 2023-12-29; Theta carries 2024+.",
            "Mid package slip is clamped to NBBO (never worse than natural ask/bid).",
        ],
    }

    for cfg in configs:
        label = str(cfg.get("label") or cfg.get("marks"))
        LOG.info("=== book %s ===", label)
        book = run_friday_pcs_book(
            start=start,
            end=end,
            source=source,
            mark_mode=str(cfg.get("marks", "mid")),
            package_slip=float(cfg.get("package_slip", args.package_slip)),
            target_delta=float(args.target_delta),
            width=float(args.width),
            target_dte=int(args.dte),
            dte_tol=int(args.dte_tol),
            tp_profit_frac=float(cfg.get("tp_profit_frac", args.tp_profit_frac)),
            sl_mult=float(cfg.get("sl_mult", args.sl_mult)),
            use_stop=bool(cfg.get("use_stop", not args.no_stop)),
            min_credit=float(args.min_credit),
            margin_buffer=float(args.margin_buffer),
            label=label,
        )
        stats = book_stats(book.trades, capital=book.capital, equity=book.equity)
        p5 = phase5_summary(book)
        LOG.info("%s", _fmt(label, stats))
        if p5.get("phase5"):
            LOG.info("Phase-5: %s", p5["phase5"])
        if stats.get("years"):
            LOG.info("Years: %s", stats["years"])

        csv_path = outdir / f"{label}_trades.csv"
        trades_to_frame(book.trades).to_csv(csv_path, index=False)
        if book.equity is not None and not book.equity.empty:
            eq_path = outdir / f"{label}_equity.csv"
            eq = book.equity.rename("strategy").to_frame()
            if book.spy_equity is not None and not book.spy_equity.empty:
                eq["spy"] = book.spy_equity.reindex(eq.index)
            eq.to_csv(eq_path)
        summary["books"][label] = {
            "stats": stats,
            "phase5": p5.get("phase5"),
            "strat_perf": p5.get("strat_perf"),
            "spy_perf": p5.get("spy_perf"),
            "notes": book.notes,
            "n_skipped": len(book.skipped),
            "skip_reasons": _count_reasons(book.skipped),
            "trades_csv": str(csv_path),
        }

    summary["wall_s"] = round(time.time() - t0, 1)
    out_json = outdir / f"summary_{args.variants or 'custom'}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %s (wall %ss)", out_json, summary["wall_s"])
    return 0


def _count_reasons(skipped: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in skipped:
        r = str(s.get("reason") or "unknown")
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
