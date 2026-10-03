#!/usr/bin/env python3
"""Claim-check cash-flow screen v2 (1y hold + FCF yield + SH yield + cash OP + mom).

Usage (Windows CMD)::

    venv\\Scripts\\activate
    set PYTHONPATH=.
    python scripts\\research\\claimcheck_cashflow_four_filters_v2.py
    python scripts\\research\\claimcheck_cashflow_four_filters_v2.py --no-momentum
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.cashflow_four_filters_v2 import (  # noqa: E402
    HOLD_YEARS,
    run_claimcheck_v2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_cf4_v2")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "cashflow_four_filters_v2"


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt(label: str, s: dict) -> str:
    return (
        f"{label}: n={s.get('n')} syms={s.get('n_symbols')} "
        f"med={s.get('median_ret_pct')}% mean={s.get('mean_ret_pct')}% "
        f"WR={s.get('win_rate_pct')}% PF={s.get('pf')} "
        f"spyMed={s.get('spy_median_ret_pct')}% excessMed={s.get('excess_median_pct')}% "
        f"buy={s.get('buy_from')}..{s.get('buy_to')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp", default="Q", choices=["Q", "FY", "BOTH"])
    ap.add_argument("--hold-years", type=int, default=HOLD_YEARS)
    ap.add_argument("--fcf-yield-min", type=float, default=0.05)
    ap.add_argument("--sh-yield-min", type=float, default=0.01)
    ap.add_argument("--cash-op-roa-min", type=float, default=0.06)
    ap.add_argument("--mom-min", type=float, default=0.0)
    ap.add_argument("--no-momentum", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--no-dedupe-year", action="store_true")
    ap.add_argument("--min-filed", default=None)
    ap.add_argument("--max-filed", default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    outdir = args.outdir or (DEFAULT_OUT / date.today().isoformat())
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    LOG.info(
        "v2 claim-check fp=%s hold=%dy mom=%s fcfY>=%.2f shY>=%.2f cashOP>=%.2f",
        args.fp,
        args.hold_years,
        not args.no_momentum,
        args.fcf_yield_min,
        args.sh_yield_min,
        args.cash_op_roa_min,
    )
    result = run_claimcheck_v2(
        fp_mode=args.fp,
        hold_years=args.hold_years,
        require_full_hold=not args.allow_partial,
        dedupe_year=not args.no_dedupe_year,
        require_momentum=not args.no_momentum,
        fcf_yield_min=args.fcf_yield_min,
        sh_yield_min=args.sh_yield_min,
        cash_op_roa_min=args.cash_op_roa_min,
        mom_min=args.mom_min,
        min_filed=_parse_date(args.min_filed),
        max_filed=_parse_date(args.max_filed),
    )
    elapsed = time.perf_counter() - t0

    primary = result.stats_unique_year
    LOG.info("Screen rows: %d", result.screen_n)
    LOG.info("%s", _fmt("all passes", result.stats))
    LOG.info("%s", _fmt("unique symbol-year", primary))

    trades = result.trades
    stem = f"trades_{args.fp.lower()}_h{args.hold_years}"
    if args.no_momentum:
        stem += "_nomom"
    trades_path = outdir / f"{stem}.csv"
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
        LOG.info("Wrote %s", trades_path)

    med = primary.get("median_ret_pct")
    spy_med = primary.get("spy_median_ret_pct")
    verdict = "no_promote"
    if primary.get("n", 0) < 30:
        note = "n<30 thin sample"
    elif med is None:
        note = "no median"
    elif spy_med is not None and med > spy_med + 5 and primary.get("n", 0) >= 30:
        verdict = "beats_spy_median"
        note = "beats SPY median on 1y unique-year book; not Sharpe>1 / occupancy promote"
    elif spy_med is not None and med > spy_med:
        verdict = "edge_weak"
        note = "median above SPY but <5pp; weak"
    else:
        note = "does not beat SPY median (or insufficient edge)"
    LOG.info("Verdict: %s (%s)", verdict, note)

    summary = {
        "version": "v2",
        "fp_mode": args.fp,
        "params": result.params,
        "require_full_hold": not args.allow_partial,
        "dedupe_year": not args.no_dedupe_year,
        "screen_n": result.screen_n,
        "stats_all": result.stats,
        "stats_unique_year": primary,
        "verdict": verdict,
        "verdict_note": note,
        "elapsed_s": round(elapsed, 2),
        "trades_csv": str(trades_path) if not trades.empty else None,
    }
    sum_path = outdir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %s elapsed=%.1fs", sum_path, elapsed)

    status = (
        ROOT
        / "docs"
        / "status_log"
        / "edge_hunt"
        / f"{date.today().isoformat()}_cashflow_four_filters_v2.md"
    )
    lines = [
        "# Cash-flow screen v2 claim-check",
        "",
        f"**Verdict: {verdict}** — {note}",
        "",
        "## Spec",
        "",
        "- Hold **1y** (unique symbol-year rebalance)",
        f"- CapEx/OCF < 35%",
        f"- FCF yield (FCF/EV) > {args.fcf_yield_min:.0%}",
        f"- Shareholder yield (buybacks+dividends)/mkt_cap > {args.sh_yield_min:.0%}",
        f"- Cash OP ROA = (OpInc - (NI - OCF)) / Assets > {args.cash_op_roa_min:.0%}",
        f"- 12m momentum {'OFF' if args.no_momentum else f'> {args.mom_min:.0%}'}",
        "- SIC exclude FIN/EN/UTIL/MAT/RE",
        "",
        f"- Unique-year: {_fmt('', primary).lstrip(': ')}",
        f"- All passes: {_fmt('', result.stats).lstrip(': ')}",
        f"- Wall-clock: {elapsed:.1f}s",
        f"- Trades: `{trades_path if not trades.empty else 'n/a'}`",
        "",
        "## By buy year",
        "",
    ]
    for y, st in sorted((primary.get("by_buy_year") or {}).items()):
        lines.append(
            f"- {y}: n={st['n']} med={st['median_ret_pct']}% mean={st['mean_ret_pct']}%"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- EV = mkt_cap + debt - cash (cash from companyfacts; floor EV at 0.5x mkt)",
            "- Dividends/OpInc/NI enriched from local `data/sec/companyfacts` cache",
            "- Not a Sharpe>1 / occupancy promote gate",
            "",
        ]
    )
    status.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
