#!/usr/bin/env python3
"""Claim-check a VIX spike call credit spread on Theta EOD quotes.

Cash VIX is Yahoo ^VIX (Theta index history is 403 on this plan).
VIX option NBBO is Theta Terminal on :25503.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\pipeline\\thetadata_terminal.py
  python scripts\\research\\claimcheck_vix_spike_call.py --variants baseline
  python scripts\\research\\claimcheck_vix_spike_call.py --max-entries 2 --start 2024-07-01 --end 2024-09-01
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
from utils.research.vix_spike_call import (  # noqa: E402
    CLAIMED,
    book_stats,
    cached_strikes,
    eod_mark_map,
    load_vix_spot,
    parity_forward,
    run_vix_spike_book,
    trades_to_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_vix_spike_call")

DEFAULT_OUT = ROOT / "reports" / "research" / "vix_spike_call"

# Slip variants share the same strikes. Forward books reprice the short strike.
VARIANT_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "baseline": [
        {"arm": "cross", "ref": "spot", "trigger": 32.0, "marks": "mid", "package_slip": 0.10},
        {"arm": "cross", "ref": "spot", "trigger": 32.0, "marks": "mid", "package_slip": 0.0},
        {"arm": "cross", "ref": "spot", "trigger": 32.0, "marks": "ask_bid", "package_slip": 0.0},
        {"arm": "cross", "ref": "forward", "trigger": 32.0, "marks": "mid", "package_slip": 0.10},
        {"arm": "cross", "ref": "forward", "trigger": 32.0, "marks": "mid", "package_slip": 0.0},
        {"arm": "cross", "ref": "forward", "trigger": 32.0, "marks": "ask_bid", "package_slip": 0.0},
        {"arm": "cross", "ref": "forward", "trigger": 30.0, "marks": "mid", "package_slip": 0.10},
        {"arm": "cross", "ref": "forward", "trigger": 35.0, "marks": "mid", "package_slip": 0.10},
        {"arm": "cross", "ref": "forward", "trigger": 40.0, "marks": "mid", "package_slip": 0.10},
        {"arm": "elevated", "ref": "forward", "trigger": 32.0, "marks": "mid", "package_slip": 0.10},
    ],
}


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _tag(cfg: Dict[str, Any]) -> str:
    arm = "elev" if str(cfg.get("arm")) == "elevated" else "cross"
    ref = "fwd" if str(cfg.get("ref")) == "forward" else "spot"
    trig = f"t{float(cfg.get('trigger') or 32):g}"
    mark = str(cfg.get("marks") or "mid")
    slip = float(cfg.get("package_slip") or 0.0)
    if mark in ("ask_bid", "natural", "askbid"):
        px = "ask_bid"
    elif slip > 0:
        px = f"mid_slip{slip:g}"
    else:
        px = "mid"
    return "_".join([arm, trig, ref, px])


def _fmt(label: str, stats: Dict[str, Any]) -> str:
    return (
        f"{label}: n={stats.get('n')} /yr={stats.get('per_year')} "
        f"WR={stats.get('win_rate_pct')}% meanRoR={stats.get('mean_ror_pct')}% "
        f"medRoR={stats.get('med_ror_pct')}% PF={stats.get('pf')} "
        f"PnL=${stats.get('total_pnl')} avgWin={stats.get('avg_win')} "
        f"avgLoss={stats.get('avg_loss')} Sharpe~={stats.get('sharpe_proxy')} "
        f"avgCredit={stats.get('avg_credit')} medCross={stats.get('med_cross')} "
        f"medRisk={stats.get('med_max_risk')} "
        f"budgetPnL=${stats.get('budget_pnl')} budgetMDD={stats.get('budget_mdd_pct')}% "
        f"exits={stats.get('exit_reasons')} years={stats.get('years')}"
    )


def _count_reasons(skipped: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in skipped:
        key = str(row.get("reason") or "unknown")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-08-29")
    parser.add_argument("--trigger", type=float, default=32.0)
    parser.add_argument("--otm", type=float, default=10.0)
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--min-credit", type=float, default=0.80)
    parser.add_argument("--marks", default="mid")
    parser.add_argument("--package-slip", type=float, default=0.10)
    parser.add_argument("--arm", default="cross", choices=["cross", "elevated"])
    parser.add_argument("--ref", default="spot", choices=["spot", "forward"])
    parser.add_argument("--risk-budget", type=float, default=15000.0)
    parser.add_argument("--max-entries", type=int, default=0, help="Stop after N fills (0 = no cap)")
    parser.add_argument("--variants", default="", help="Named suite: baseline")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if args.variants:
        suite = VARIANT_SUITES.get(args.variants.strip().lower())
        if not suite:
            LOG.error("Unknown --variants %s; choose from %s", args.variants, list(VARIANT_SUITES))
            return 1
        configs = suite
    else:
        configs = [
            {
                "arm": args.arm,
                "ref": args.ref,
                "trigger": args.trigger,
                "marks": args.marks,
                "package_slip": args.package_slip,
            }
        ]

    outdir = args.outdir or (DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d"))
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not ready at %s", client.base_url)
        return 1

    LOG.info("Loading Yahoo ^VIX %s..%s", start, end)
    spots = load_vix_spot(start - timedelta(days=10), end)
    LOG.info("VIX rows %s", 0 if spots is None else len(spots))
    exps = list_expirations(client, "VIX")
    LOG.info("VIX expirations %s", len(exps))

    eod_cache: Dict[Tuple[Any, ...], Any] = {}
    strike_cache: Dict[date, List[float]] = {}

    def mark_fn(exp: date, strike: float, right: str, d0: date, d1: date):
        return eod_mark_map(client, eod_cache, exp, strike, right, d0, d1)

    def strike_fn(exp: date) -> List[float]:
        return cached_strikes(client, strike_cache, exp)

    def forward_fn(exp: date, day: date, spot: float, dte: int) -> Optional[float]:
        strikes = strike_fn(exp)
        return parity_forward(client, eod_cache, strikes, exp, day, spot, dte)

    t0 = time.time()
    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "structure": {
            "otm": args.otm,
            "width": args.width,
            "dte_min": 30,
            "dte_max": 60,
            "min_credit": args.min_credit,
            "tp_remain": 0.40,
            "expiry_week_days": 3,
            "commission_per_leg": 0.65,
            "legs_round_turn": 4,
            "risk_budget": args.risk_budget,
        },
        "variants": args.variants or None,
        "books": {},
        "wall_s": None,
        "notes": [
            "Cash VIX is Yahoo ^VIX. Theta index history returns 403 on this plan.",
            "Spot reference matches the pasted strike rule (round((VIX+10)/5)*5).",
            "Forward reference is put-call parity on the nearest listed strike, then the same +10 rule.",
            "A leg with bid-ask wider than 3 points is dropped. Ask stuck at 10 against a dime bid is not a fill.",
            "Put-call parity is ignored when the call spread plus the put spread is wider than 2 points.",
            "Expiry exit is the EOD mark with <=3 calendar days left. It is not the VIX special opening quotation.",
            "EOD marks miss an intraday 60% decay that reverts before the close.",
            "1-lot equity starts at $100k. budget_* sizes each trade to a $15k defined-risk cap.",
            "Trade Sharpe uses actual entries per year, not 365/hold.",
        ],
    }

    for cfg in configs:
        tag = _tag(cfg)
        LOG.info("=== book %s ===", tag)
        book = run_vix_spike_book(
            spots,
            exps,
            start=start,
            end=end,
            mark_fn=mark_fn,
            strike_fn=strike_fn,
            forward_fn=forward_fn,
            trigger=float(cfg["trigger"]),
            otm=float(args.otm),
            width=float(args.width),
            min_credit=float(args.min_credit),
            mark_mode=str(cfg["marks"]),
            package_slip=float(cfg.get("package_slip") or 0.0),
            arm_mode=str(cfg["arm"]),
            ref_mode=str(cfg["ref"]),
            max_entries=int(args.max_entries),
            label=tag,
        )
        stats = book_stats(book.trades, risk_budget=float(args.risk_budget))
        summary["books"][tag] = {
            "stats": stats,
            "n_skipped": len(book.skipped),
            "skip_reasons": _count_reasons(book.skipped),
            "label": book.label,
            "config": {
                "arm": cfg["arm"],
                "ref": cfg["ref"],
                "trigger": float(cfg["trigger"]),
                "marks": cfg["marks"],
                "package_slip": float(cfg.get("package_slip") or 0.0),
            },
        }
        LOG.info(_fmt(tag, stats))
        frame = trades_to_frame(book.trades)
        frame.to_csv(outdir / f"VIX_{tag}_trades.csv", index=False)
        with (outdir / f"VIX_{tag}_skipped.json").open("w", encoding="utf-8") as handle:
            json.dump(book.skipped[:8000], handle, indent=2, default=str)

    summary["wall_s"] = round(time.time() - t0, 1)
    out_name = "summary_baseline.json" if args.variants == "baseline" else "summary.json"
    with (outdir / out_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOG.info("Wrote %s (wall %.1fs)", outdir, summary["wall_s"])
    _print_verdict(summary)
    return 0


def _print_verdict(summary: Dict[str, Any]) -> None:
    print("\n=== VIX spike call credit spread ===")
    print(
        "Rules: VIX cross, 30-60 DTE Wednesday monthly, short call +10 pts, "
        "10-wide, TP debit<=40% credit, no stop, expiry-week EOD, $0.65 x 4."
    )
    for tag, body in summary.get("books", {}).items():
        stats = body.get("stats") or {}
        print(_fmt(tag, stats), f"| skips={body.get('n_skipped')} {body.get('skip_reasons')}")
    print(f"wall_s={summary.get('wall_s')}")


if __name__ == "__main__":
    raise SystemExit(main())
