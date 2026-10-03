#!/usr/bin/env python3
"""Claim-check YouTube cash-flow four-filter screen (5y median return).

Claim: median ~86% over 5y vs market ~66% (ex FIN/EN/UTIL/MAT/RE).

Usage (Windows CMD)::

    venv\\Scripts\\activate
    set PYTHONPATH=.
    python scripts\\research\\claimcheck_cashflow_four_filters.py
    python scripts\\research\\claimcheck_cashflow_four_filters.py --fp FY
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

from utils.research.cashflow_four_filters import CLAIMED, run_claimcheck  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_cashflow_four_filters")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "cashflow_four_filters"


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
    ap.add_argument("--hold-years", type=int, default=5)
    ap.add_argument("--min-filed", default=None)
    ap.add_argument("--max-filed", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="Keep trades without full hold")
    ap.add_argument("--no-dedupe-year", action="store_true")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    outdir = args.outdir or (DEFAULT_OUT / date.today().isoformat())
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    LOG.info(
        "Running four-filter claim-check fp=%s hold=%dy full_hold=%s dedupe_year=%s",
        args.fp,
        args.hold_years,
        not args.allow_partial,
        not args.no_dedupe_year,
    )
    result = run_claimcheck(
        fp_mode=args.fp,
        hold_years=args.hold_years,
        require_full_hold=not args.allow_partial,
        dedupe_year=not args.no_dedupe_year,
        min_filed=_parse_date(args.min_filed),
        max_filed=_parse_date(args.max_filed),
    )
    elapsed = time.perf_counter() - t0

    LOG.info("Screen rows (all-Q/FY after f3): %d", result.screen_n)
    LOG.info("%s", _fmt("all passes", result.stats))
    LOG.info("%s", _fmt("unique symbol-year", result.stats_unique_year))
    LOG.info(
        "Claim: median %.0f%% vs market %.0f%% over %dy",
        CLAIMED["median_5y_pct"],
        CLAIMED["market_median_5y_pct"],
        CLAIMED["hold_years"],
    )

    trades = result.trades
    trades_path = outdir / f"trades_{args.fp.lower()}_h{args.hold_years}.csv"
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
        LOG.info("Wrote %s", trades_path)

    summary = {
        "fp_mode": args.fp,
        "hold_years": args.hold_years,
        "require_full_hold": not args.allow_partial,
        "dedupe_year": not args.no_dedupe_year,
        "screen_n": result.screen_n,
        "stats_all": result.stats,
        "stats_unique_year": result.stats_unique_year,
        "claimed": result.claimed,
        "elapsed_s": round(elapsed, 2),
        "trades_csv": str(trades_path) if not trades.empty else None,
    }
    # Verdict vs claim (unique-year book is primary)
    primary = result.stats_unique_year
    med = primary.get("median_ret_pct")
    spy_med = primary.get("spy_median_ret_pct")
    verdict = "no_promote"
    note = ""
    if primary.get("n", 0) < 30:
        note = "n<30 thin sample"
    elif med is None:
        note = "no median"
    elif med >= CLAIMED["median_5y_pct"] - 5 and (
        spy_med is None or med >= spy_med
    ):
        # near claim and beats SPY median — still not Sharpe>1 promote
        verdict = "claim_near"
        note = "median near claim / beats SPY median on this book; not edge-hunt promote"
    elif spy_med is not None and med > spy_med + 5:
        verdict = "beats_spy_median"
        note = "beats SPY median but below video claim; research-interesting only"
    else:
        note = "below claim and/or does not clear SPY median by enough"
    summary["verdict"] = verdict
    summary["verdict_note"] = note
    LOG.info("Verdict: %s (%s)", verdict, note)

    sum_path = outdir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %s elapsed=%.1fs", sum_path, elapsed)

    # Status log
    status = ROOT / "docs" / "status_log" / "edge_hunt" / f"{date.today().isoformat()}_cashflow_four_filters_claimcheck.md"
    lines = [
        "# Cash-flow four-filter claim-check",
        "",
        f"**Verdict: {verdict}** — {note}",
        "",
        f"- Claim: median **{CLAIMED['median_5y_pct']:.0f}%** / 5y vs market **{CLAIMED['market_median_5y_pct']:.0f}%**",
        f"- Book: fp={args.fp}, hold={args.hold_years}y, full_hold={not args.allow_partial}, unique symbol-year={not args.no_dedupe_year}",
        f"- Unique-year: {_fmt('', primary).strip(': ')}",
        f"- All passes: {_fmt('', result.stats).strip(': ')}",
        f"- Wall-clock: {elapsed:.1f}s",
        f"- Trades: `{trades_path if not trades.empty else 'n/a'}`",
        f"- Summary: `{sum_path}`",
        "",
        "## By buy year (unique symbol-year)",
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
            "- Entry = next 1d close on/after SEC `filed`; exit = close on/after filed+5y",
            "- Filter 3 mkt_cap recomputed as shares x close as-of filed when needed",
            "- SIC sector exclusions applied; not a Sharpe>1 / occupancy promote",
            "",
        ]
    )
    status.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
