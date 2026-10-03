#!/usr/bin/env python3
"""Claim-check cash-flow screen v3 (top-N + rel mom + CapEx exception + portfolio).

Usage (Windows CMD)::

    venv\\Scripts\\activate
    set PYTHONPATH=.
    python scripts\\research\\claimcheck_cashflow_four_filters_v3.py
    python scripts\\research\\claimcheck_cashflow_four_filters_v3.py --top-n 25
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

from utils.research.cashflow_four_filters_v3 import (  # noqa: E402
    TOP_N,
    run_claimcheck_v3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_cf4_v3")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "cashflow_four_filters_v3"


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt_trades(s: dict) -> str:
    return (
        f"n={s.get('n')} syms={s.get('n_symbols')} "
        f"med={s.get('median_ret_pct')}% mean={s.get('mean_ret_pct')}% "
        f"WR={s.get('win_rate_pct')}% PF={s.get('pf')} "
        f"spyMed={s.get('spy_median_ret_pct')}% "
        f"rspMed={s.get('rsp_median_ret_pct')}% "
        f"xsSPY={s.get('excess_median_pct')}% xsRSP={s.get('excess_median_vs_rsp_pct')}%"
    )


def _fmt_port(s: dict) -> str:
    return (
        f"years={s.get('n_years')} CAGR={s.get('cagr_pct')}% "
        f"total={s.get('total_return_pct')}% "
        f"medYear={s.get('median_year_ret_pct')}% "
        f"SPY_CAGR={s.get('SPY_cagr_pct')}% "
        f"RSP_CAGR={s.get('RSP_cagr_pct')}% "
        f"xsSPY={s.get('excess_cagr_vs_SPY_pct')}% "
        f"xsRSP={s.get('excess_cagr_vs_RSP_pct')}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp", default="Q", choices=["Q", "FY", "BOTH"])
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--hold-years", type=int, default=1)
    ap.add_argument("--fcf-yield-min", type=float, default=0.05)
    ap.add_argument("--sh-yield-min", type=float, default=0.01)
    ap.add_argument("--cash-op-roa-min", type=float, default=0.06)
    ap.add_argument("--no-rel-momentum", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--min-filed", default=None)
    ap.add_argument("--max-filed", default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    outdir = args.outdir or (DEFAULT_OUT / date.today().isoformat())
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    LOG.info(
        "v3 top_n=%d hold=%dy rel_mom=%s",
        args.top_n,
        args.hold_years,
        not args.no_rel_momentum,
    )
    result = run_claimcheck_v3(
        fp_mode=args.fp,
        top_n=args.top_n,
        hold_years=args.hold_years,
        require_full_hold=not args.allow_partial,
        require_rel_momentum=not args.no_rel_momentum,
        fcf_yield_min=args.fcf_yield_min,
        sh_yield_min=args.sh_yield_min,
        cash_op_roa_min=args.cash_op_roa_min,
        min_filed=_parse_date(args.min_filed),
        max_filed=_parse_date(args.max_filed),
        yahoo_dir=outdir / "yahoo",
        refresh_yahoo=args.refresh_yahoo,
    )
    elapsed = time.perf_counter() - t0

    LOG.info("Eligible=%d selected=%d", result.eligible_n, result.selected_n)
    LOG.info("Trades: %s", _fmt_trades(result.trade_stats))
    LOG.info("Portfolio: %s", _fmt_port(result.portfolio_stats))

    trades_path = outdir / f"trades_top{args.top_n}_h{args.hold_years}.csv"
    if not result.trades.empty:
        result.trades.to_csv(trades_path, index=False)
        LOG.info("Wrote %s", trades_path)
    port_path = outdir / f"portfolio_years_top{args.top_n}.csv"
    if not result.portfolio_years.empty:
        result.portfolio_years.to_csv(port_path, index=False)
        LOG.info("Wrote %s", port_path)

    # Verdict: portfolio CAGR vs SPY and RSP; trade median secondary
    ps = result.portfolio_stats
    ts = result.trade_stats
    verdict = "no_promote"
    xs_spy = ps.get("excess_cagr_vs_SPY_pct")
    xs_rsp = ps.get("excess_cagr_vs_RSP_pct")
    n_years = ps.get("n_years") or 0
    if n_years < 5:
        note = "portfolio n_years<5 thin"
    elif xs_spy is not None and xs_spy > 0 and xs_rsp is not None and xs_rsp > 0:
        verdict = "beats_spy_and_rsp_cagr"
        note = "EW portfolio CAGR beats SPY and RSP; still not Sharpe>1 / occupancy promote"
    elif xs_spy is not None and xs_spy > 0:
        verdict = "beats_spy_cagr_only"
        note = "EW portfolio beats SPY CAGR but not RSP (or RSP missing)"
    elif (
        ts.get("n", 0) >= 30
        and ts.get("excess_median_pct") is not None
        and ts["excess_median_pct"] > 5
        and ts.get("excess_median_vs_rsp_pct") is not None
        and ts["excess_median_vs_rsp_pct"] > 0
    ):
        verdict = "trade_median_edge"
        note = "trade median beats SPY+RSP but portfolio CAGR does not"
    else:
        note = "portfolio CAGR and/or trade median do not clear SPY+RSP"
    LOG.info("Verdict: %s (%s)", verdict, note)

    summary = {
        "version": "v3",
        "params": result.params,
        "eligible_n": result.eligible_n,
        "selected_n": result.selected_n,
        "trade_stats": result.trade_stats,
        "portfolio_stats": result.portfolio_stats,
        "verdict": verdict,
        "verdict_note": note,
        "elapsed_s": round(elapsed, 2),
        "trades_csv": str(trades_path) if not result.trades.empty else None,
        "portfolio_csv": str(port_path) if not result.portfolio_years.empty else None,
    }
    sum_path = outdir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %s elapsed=%.1fs", sum_path, elapsed)

    tag = f"top{args.top_n}"
    if args.no_rel_momentum:
        tag += "_norel"
    status = (
        ROOT
        / "docs"
        / "status_log"
        / "edge_hunt"
        / f"{date.today().isoformat()}_cashflow_four_filters_v3_{tag}.md"
    )
    rel = "off" if args.no_rel_momentum else "on (stock 12-1 > SPY 12-1)"
    lines = [
        f"# Cash-flow screen v3 ({tag})",
        "",
        f"**Verdict: {verdict}** — {note}",
        "",
        "## Spec",
        "",
        f"- Top-{args.top_n} / year by composite rank(FCF yield)+rank(cash OP ROA)+rank(mom)",
        f"- Relative 12-1 momentum: {rel}",
        "- CapEx/OCF < 35% OR cash_op_roa > 15%",
        "- Soft floors: FCF yield > 5%, SH yield > 1%, cash OP ROA > 6%",
        "- Hold 1y; EW annual portfolio rebalance + per-trade book",
        "- Portfolio CAGR uses years with n_names>=10 (thin cohorts excluded from headline)",
        "- Benchmarks: SPY (TimescaleDB) + RSP (Yahoo)",
        "",
        f"- Eligible / selected: {result.eligible_n} / {result.selected_n}",
        f"- Trades: {_fmt_trades(result.trade_stats)}",
        f"- Portfolio: {_fmt_port(result.portfolio_stats)}",
        f"- Wall-clock: {elapsed:.1f}s",
        "",
        "## Portfolio years",
        "",
    ]
    if not result.portfolio_years.empty:
        for _, r in result.portfolio_years.iterrows():
            lines.append(
                f"- {int(r['year'])}: n={int(r['n_names'])} "
                f"port={r['port_ret_pct']:.1f}% "
                f"SPY={r.get('SPY_ret_pct')} RSP={r.get('RSP_ret_pct')}"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Primary gate is **equal-weight portfolio CAGR vs SPY and RSP** "
            "(n_names>=10 years), not unweighted trade-median vs SPY-median.",
            "",
        ]
    )
    status.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
