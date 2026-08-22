#!/usr/bin/env python3
"""
Phase 3: SPY core + Weekly BigVol satellite blend vs SPY buy-and-hold.

Default 70/30 SPY/BigVol; also scores 80/20 and 60/40.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_spy_bigvol_blend.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from spy_benchmark_screen import (
    DEFAULT_BIGVOL_SETUPS,
    _clip_dates,
    _format_elapsed,
    format_table,
    load_spy,
    passes_gates,
    perf_stats,
    strategy_bigvol_portfolio,
    strategy_buy_hold,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("spy_bigvol_blend")


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def blend_equities(
    spy_eq: pd.Series,
    sleeve_eq: pd.Series,
    spy_weight: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Fixed capital split: spy_weight in SPY path, (1-spy_weight) in BigVol sleeve.
    Both legs start at 1.0; combined starts at 1.0.
    Idle sleeve cash stays in the sleeve (0% yield) — already in sleeve_eq.
    """
    w_spy = float(spy_weight)
    w_bv = 1.0 - w_spy
    idx = spy_eq.index
    spy_n = (spy_eq / spy_eq.iloc[0]).reindex(idx).ffill()
    bv = sleeve_eq.reindex(idx)
    first = bv.first_valid_index()
    if first is not None:
        bv.loc[:first] = bv.loc[first]
    bv = bv.ffill().fillna(1.0)
    bv_n = bv / bv.iloc[0]
    combined = w_spy * spy_n + w_bv * bv_n
    # Invested flag: always fully allocated at capital level (SPY core is always long)
    invested = pd.Series(1.0, index=idx)
    return combined, invested


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 SPY + BigVol blend")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2025-11-26")
    ap.add_argument("--spy-provider", default="IB", help="Prefer IB for longer SPY; falls back handled by load")
    ap.add_argument("--bigvol-setups", type=Path, default=DEFAULT_BIGVOL_SETUPS)
    ap.add_argument("--bigvol-alloc", type=float, default=0.10)
    ap.add_argument("--bigvol-max-pos", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["0.70", "0.80", "0.60"],
        help="SPY weights to test (rest goes to BigVol sleeve)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase3",
    )
    args = ap.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    try:
        spy = load_spy(args.start, args.end, provider=args.spy_provider)
    except RuntimeError:
        logger.warning("SPY provider %s failed; falling back to ALPACA", args.spy_provider)
        spy = load_spy(args.start, args.end, provider="ALPACA")
    spy = _normalize_ohlc(spy)
    spy = _clip_dates(spy, args.start, args.end)
    timings["load_spy"] = time.perf_counter() - t0

    eq_bh, inv_bh = strategy_buy_hold(spy)
    spy_stats = perf_stats(
        "SPY_buy_hold",
        eq_bh,
        inv_bh,
        notes=f"100% SPY",
    )
    spy_range_note = (
        f"SPY bars {spy.index[0].date()} -> {spy.index[-1].date()} (n={len(spy)}); "
        f"BigVol setups={args.bigvol_setups.name}; sleeve alloc={args.bigvol_alloc:.0%} "
        f"max_pos={args.bigvol_max_pos}; idle sleeve cash earns 0%"
    )
    logger.info(spy_range_note)

    t0 = time.perf_counter()
    if not args.bigvol_setups.exists():
        raise FileNotFoundError(args.bigvol_setups)
    eq_bv, inv_bv, bv_notes = strategy_bigvol_portfolio(
        args.bigvol_setups,
        args.start,
        args.end,
        alloc_frac=args.bigvol_alloc,
        max_positions=args.bigvol_max_pos,
        workers=args.workers,
    )
    eq_bv.index = pd.DatetimeIndex(eq_bv.index).tz_localize(None).normalize()
    inv_bv.index = pd.DatetimeIndex(inv_bv.index).tz_localize(None).normalize()
    eq_bv = eq_bv[~eq_bv.index.duplicated(keep="last")].sort_index()
    inv_bv = inv_bv[~inv_bv.index.duplicated(keep="last")].sort_index()
    timings["bigvol_sleeve"] = time.perf_counter() - t0

    # Pure sleeve on SPY calendar for reference
    eq_bv_aligned = eq_bv.reindex(eq_bh.index)
    first = eq_bv_aligned.first_valid_index()
    if first is not None:
        eq_bv_aligned.loc[:first] = eq_bv_aligned.loc[first]
    eq_bv_aligned = eq_bv_aligned.ffill().fillna(1.0)
    if float(eq_bv_aligned.iloc[0]) != 0:
        eq_bv_aligned = eq_bv_aligned / eq_bv_aligned.iloc[0]
    inv_bv_aligned = inv_bv.reindex(eq_bh.index).fillna(0.0)

    all_stats = [spy_stats]
    curves: Dict[str, pd.Series] = {"SPY_buy_hold": eq_bh}

    sleeve_stats = perf_stats(
        "BigVol_sleeve_100pct",
        eq_bv_aligned,
        inv_bv_aligned,
        spy_cagr=spy_stats.cagr,
        spy_sharpe=spy_stats.sharpe,
        notes=bv_notes,
    )
    all_stats.append(sleeve_stats)
    curves["BigVol_sleeve_100pct"] = eq_bv_aligned

    split_weights = [float(x) for x in args.splits]
    for w_spy in split_weights:
        name = f"Blend_SPY{int(round(w_spy * 100))}_BV{int(round((1 - w_spy) * 100))}"
        t0 = time.perf_counter()
        eq_c, inv_c = blend_equities(eq_bh, eq_bv_aligned, w_spy)
        st = perf_stats(
            name,
            eq_c,
            inv_c,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes=(
                f"spy_weight={w_spy:.0%} bigvol_weight={1 - w_spy:.0%}; "
                f"sleeve_invested_mean={float(inv_bv_aligned.mean()):.0%}"
            ),
        )
        timings[name] = time.perf_counter() - t0
        all_stats.append(st)
        curves[name] = eq_c

    timings["total"] = time.perf_counter() - t_all

    curves_path = args.reports_dir / "equity_curves.csv"
    pd.DataFrame(curves).to_csv(curves_path)
    stats_path = args.reports_dir / "scorecard.json"
    stats_path.write_text(json.dumps([asdict(s) for s in all_stats], indent=2), encoding="utf-8")

    # Scorecard markdown
    args.outdir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y-%m-%d")
    log_path = args.outdir / f"{ts}_edge_hunt_phase3_scorecard.md"

    winners = []
    for s in all_stats:
        if s.name == spy_stats.name:
            continue
        ok, gate = passes_gates(s, spy_stats)
        if ok:
            winners.append((s, gate))

    # Prefer blend names for promotion ranking
    blend_winners = [w for w in winners if w[0].name.startswith("Blend_")]
    rank_pool = blend_winners if blend_winners else winners

    lines: List[str] = [
        "# Edge hunt Phase 3 - SPY core + Weekly BigVol satellite",
        "",
        f"Date: {ts}",
        "",
        "## Window / data",
        "",
        f"- Evaluation window: `{spy_stats.start}` -> `{spy_stats.end}`",
        f"- {spy_range_note}",
        "- Rule: fixed capital split SPY B&H + BigVol sleeve; idle sleeve cash earns 0%.",
        "- Default split 70/30; sensitivity 80/20 and 60/40.",
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

    if rank_pool:
        ranked = sorted(rank_pool, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
        best, gate = ranked[0]
        lines.append(f"**Promoted (Phase 3):** `{best.name}` - {gate}")
        lines.append("")
        lines.append(
            f"CAGR={best.cagr:.2%} Sharpe={best.sharpe:.2f} MDD={best.max_drawdown:.2%} "
            f"vs SPY CAGR={spy_stats.cagr:.2%} Sharpe={spy_stats.sharpe:.2f} MDD={spy_stats.max_drawdown:.2%}."
        )
        if len(ranked) > 1:
            lines.append("")
            lines.append(
                "Other passers: " + ", ".join(f"`{s.name}` ({g})" for s, g in ranked[1:])
            )
        lines.append("")
        lines.append(
            "Same BigVol signal across weights only — treat as capital-allocation result, not a new alpha."
        )
    else:
        lines.append("**No Phase-3 blend cleared gates.**")
        lines.append("")
        lines.append(
            "**Kill blend path** for beating SPY: CAGR still too far below and/or Sharpe does not beat SPY "
            "at 70/30, 80/20, or 60/40."
        )
        lines.append("")
        lines.append("Next (if continuing research at all):")
        lines.append(
            "1. Revisit BigVol *sleeve* risk (higher concurrent risk / denser entries) — not more SPY mixing."
        )
        lines.append(
            "2. Or declare no SPY-beater on this dataset and stop single-strategy edge hunting."
        )

    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")

    # Update README
    readme = args.outdir / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        link = f"| {ts} | [Phase 3 SPY+BigVol blend]({ts}_edge_hunt_phase3_scorecard.md) |"
        if "Phase 3" not in text:
            text = text.replace(
                "| 2026-08-22 | [Phase 2 dual momentum](2026-08-22_edge_hunt_phase2_scorecard.md) |",
                "| 2026-08-22 | [Phase 2 dual momentum](2026-08-22_edge_hunt_phase2_scorecard.md) |\n"
                + link,
            )
            # refresh verdict line if present
            if "Verdict" in text:
                if rank_pool:
                    best, _ = sorted(rank_pool, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)[0]
                    verdict = (
                        f"## Verdict (2026-08-22)\n\n"
                        f"Phase 3 promoted `{best.name}` "
                        f"(CAGR={best.cagr:.2%} Sharpe={best.sharpe:.2f} MDD={best.max_drawdown:.2%}).\n"
                    )
                else:
                    verdict = (
                        "## Verdict (2026-08-22)\n\n"
                        "No Phase 1/2/3 candidate cleared gates vs SPY B&H. "
                        "Blend path killed; BigVol remains a lower-DD sleeve, not a SPY beater.\n"
                    )
                import re

                text = re.sub(
                    r"## Verdict \(2026-08-22\)\n\n.*",
                    verdict.rstrip() + "\n",
                    text,
                    count=1,
                    flags=re.DOTALL,
                )
            # harness line
            if "run_spy_bigvol_blend" not in text:
                text = text.replace(
                    "python scripts\\research\\run_dual_momentum_phase2.py",
                    "python scripts\\research\\run_dual_momentum_phase2.py\n"
                    "python scripts\\research\\run_spy_bigvol_blend.py",
                )
            readme.write_text(text, encoding="utf-8")

    print("\n=== Phase 3 scorecard ===")
    print(format_table(all_stats, spy_stats))
    for s in all_stats:
        note = (s.notes or "").encode("ascii", "replace").decode("ascii")
        print(f"  note[{s.name}]: {note}")
        ok, gate = passes_gates(s, spy_stats) if s.name != spy_stats.name else (True, "BENCHMARK")
        if s.name != spy_stats.name:
            print(f"    gate -> {gate}")
    print(f"\nWrote {log_path}")
    print(f"Wrote {curves_path}")
    print(f"Total: {_format_elapsed(timings['total'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
