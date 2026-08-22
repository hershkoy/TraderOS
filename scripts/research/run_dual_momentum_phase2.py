#!/usr/bin/env python3
"""
Phase 2 edge hunt: dual momentum vs SPY buy-and-hold.

Uses IB SPY/EFA (+ optional BIL) and TRADINGVIEW SHY daily bars.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_dual_momentum_phase2.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from spy_benchmark_screen import (
    _clip_dates,
    _format_elapsed,
    format_table,
    load_spy,
    load_symbol,
    passes_gates,
    perf_stats,
    strategy_abs_momentum_sma200,
    strategy_buy_hold,
    strategy_dual_momentum,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dual_momentum_phase2")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 dual momentum vs SPY")
    ap.add_argument("--start", default="2011-01-03", help="Eval start (needs ~1y warmup before)")
    ap.add_argument("--end", default="2025-11-26")
    ap.add_argument("--spy-provider", default="IB")
    ap.add_argument("--efa-provider", default="IB")
    ap.add_argument("--shy-provider", default="TRADINGVIEW")
    ap.add_argument("--bil-provider", default="IB")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase2",
    )
    args = ap.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    load_start = (pd.Timestamp(args.start) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    t_all = time.perf_counter()
    timings = {}

    t0 = time.perf_counter()
    spy_full = load_spy(load_start, args.end, provider=args.spy_provider)
    efa_full = load_symbol("EFA", load_start, args.end, args.efa_provider)
    shy_full = load_symbol("SHY", load_start, args.end, args.shy_provider)
    bil_full = load_symbol("BIL", load_start, args.end, args.bil_provider)
    timings["load"] = time.perf_counter() - t0

    # Normalize timestamps so IB (intraday stamp) aligns with TRADINGVIEW (midnight)
    for df in (spy_full, efa_full, shy_full, bil_full):
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.sort_index(inplace=True)
        # Drop duplicate calendar days if any
        keep = ~df.index.duplicated(keep="last")
        df.drop(index=df.index[~keep], inplace=True)

    spy = _clip_dates(spy_full, args.start, args.end)
    eq_bh, inv_bh = strategy_buy_hold(spy)
    spy_stats = perf_stats(
        "SPY_buy_hold",
        eq_bh,
        inv_bh,
        notes=f"100% SPY ({args.spy_provider})",
    )
    spy_range_note = (
        f"SPY {args.spy_provider} / EFA {args.efa_provider} / SHY {args.shy_provider} / "
        f"BIL {args.bil_provider}; eval {spy.index[0].date()} -> {spy.index[-1].date()} "
        f"(n={len(spy)}; load_start={load_start})"
    )
    logger.info(spy_range_note)

    all_stats = [spy_stats]
    curves = {"SPY_buy_hold": eq_bh}

    # Absolute momentum baseline on same long window
    t0 = time.perf_counter()
    eq_abs, inv_abs = strategy_abs_momentum_sma200(spy_full)
    eq_abs = eq_abs.reindex(eq_bh.index).ffill().bfill()
    if float(eq_abs.iloc[0]) != 0:
        eq_abs = eq_abs / eq_abs.iloc[0]
    inv_abs = inv_abs.reindex(eq_bh.index).fillna(0.0)
    all_stats.append(
        perf_stats(
            "SPY_SMA200_abs_mom",
            eq_abs,
            inv_abs,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes="Month-end close>SMA200; cash=0%",
        )
    )
    curves["SPY_SMA200_abs_mom"] = eq_abs
    timings["abs"] = time.perf_counter() - t0

    # Dual + SHY
    t0 = time.perf_counter()
    eq_d, inv_d, notes = strategy_dual_momentum(spy_full, efa_full, shy_full)
    eq_d = eq_d.reindex(eq_bh.index).ffill().bfill()
    if float(eq_d.iloc[0]) != 0:
        eq_d = eq_d / eq_d.iloc[0]
    inv_d = inv_d.reindex(eq_bh.index).fillna(0.0)
    all_stats.append(
        perf_stats(
            "DualMom_SPY_EFA_SHY",
            eq_d,
            inv_d,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes=notes + f"; EFA={args.efa_provider} SHY={args.shy_provider}",
        )
    )
    curves["DualMom_SPY_EFA_SHY"] = eq_d
    timings["dual_shy"] = time.perf_counter() - t0

    # Dual + BIL
    t0 = time.perf_counter()
    eq_b, inv_b, notes_b = strategy_dual_momentum(spy_full, efa_full, bil_full)
    eq_b = eq_b.reindex(eq_bh.index).ffill().bfill()
    if float(eq_b.iloc[0]) != 0:
        eq_b = eq_b / eq_b.iloc[0]
    inv_b = inv_b.reindex(eq_bh.index).fillna(0.0)
    all_stats.append(
        perf_stats(
            "DualMom_SPY_EFA_BIL",
            eq_b,
            inv_b,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes=notes_b + f"; EFA={args.efa_provider} BIL={args.bil_provider}",
        )
    )
    curves["DualMom_SPY_EFA_BIL"] = eq_b
    timings["dual_bil"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_all

    # Patch write_status_log to Phase 2 filename by calling with modified approach:
    # write_status_log hardcodes phase1 — write our own file after calling helpers.
    curves_path = args.reports_dir / "equity_curves.csv"
    pd.DataFrame(curves).to_csv(curves_path)
    stats_path = args.reports_dir / "scorecard.json"
    stats_path.write_text(json.dumps([asdict(s) for s in all_stats], indent=2), encoding="utf-8")

    # Write Phase 2 scorecard directly
    args.outdir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y-%m-%d")
    log_path = args.outdir / f"{ts}_edge_hunt_phase2_scorecard.md"
    winners = []
    for s in all_stats:
        if s.name == spy_stats.name:
            continue
        ok, gate = passes_gates(s, spy_stats)
        if ok:
            winners.append((s, gate))

    lines = [
        "# Edge hunt Phase 2 - dual momentum vs SPY buy-and-hold",
        "",
        f"Date: {ts}",
        "",
        "## Window / data",
        "",
        f"- Evaluation window: `{spy_stats.start}` -> `{spy_stats.end}`",
        f"- {spy_range_note}",
        "- Rule: month-end pick SPY vs EFA by ~12m return; if winner 12m <= 0 hold SHY/BIL.",
        "- Gate: Sharpe >= SPY and/or better MDD with CAGR within ~2pp (or higher CAGR).",
        "",
        "## Scorecard",
        "",
        format_table(all_stats, spy_stats),
        "",
        "## Notes per candidate",
        "",
    ]
    for s in all_stats:
        lines.append(f"- **{s.name}**: {s.notes or 'n/a'}")
    lines.append("")
    lines.append("## Timings")
    lines.append("")
    for k, v in timings.items():
        lines.append(f"- {k}: {_format_elapsed(v)}")
    lines.append("")
    lines.append("## Promotion")
    lines.append("")
    if winners:
        winners_sorted = sorted(winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
        best, gate = winners_sorted[0]
        lines.append(f"**Promoted (Phase 2):** `{best.name}` - {gate}")
        lines.append("")
        lines.append(
            f"CAGR={best.cagr:.2%} Sharpe={best.sharpe:.2f} MDD={best.max_drawdown:.2%} "
            f"vs SPY CAGR={spy_stats.cagr:.2%} Sharpe={spy_stats.sharpe:.2f} MDD={spy_stats.max_drawdown:.2%}."
        )
        if len(winners) > 1:
            lines.append("")
            lines.append(
                "Other passers: " + ", ".join(f"`{s.name}` ({g})" for s, g in winners_sorted[1:])
            )
    else:
        lines.append("**No Phase-2 candidate cleared gates.**")
        near = [
            s
            for s in all_stats
            if s.name != spy_stats.name
            and abs(s.max_drawdown) < abs(spy_stats.max_drawdown) - 0.05
        ]
        if near:
            lines.append("")
            lines.append("Near-miss (better MDD, CAGR still short):")
            for s in near:
                lines.append(
                    f"- `{s.name}`: CAGR={s.cagr:.2%} Sharpe={s.sharpe:.2f} MDD={s.max_drawdown:.2%}"
                )
    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Phase 2 scorecard ===")
    print(format_table(all_stats, spy_stats))
    for s in all_stats:
        note = (s.notes or "").encode("ascii", "replace").decode("ascii")
        print(f"  note[{s.name}]: {note}")
    print(f"\nWrote {log_path}")
    print(f"Wrote {curves_path}")
    print(f"Total: {_format_elapsed(timings['total'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
