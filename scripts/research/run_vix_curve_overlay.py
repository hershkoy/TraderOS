#!/usr/bin/env python3
"""
Phase 6c: VIX term-structure overlay on SPY (now that IB VIX/VIX3M are ingested).

Candidates:
  1) Binary contango filter (risk-off when VIX >= VIX3M)
  2) Binary + min VIX gate (only risk-off if also VIX >= 20 / 25)
  3) Soft curve weight (clip(VIX3M/VIX, 0, 1))
  4) Binary * VolTarget_12pct_20d_cap1
  5) Soft * VolTarget_12pct_20d_cap1

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_vix_curve_overlay.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.cta_trend import load_macro_closes
from utils.research.metrics import (
    EVAL_END,
    EVAL_START,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    PerfStats,
    annual_returns,
    format_annual_table,
    format_elapsed,
    format_table,
    passes_phase5_gates,
    perf_stats,
    window_equity,
)
from utils.research.panel import load_spy_close
from utils.research.report import plot_equity_and_dd
from utils.research.vix_curve import simulate_vix_curve_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vix_curve_overlay")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _spy_bh(spy_close: pd.Series, start: str, end: str) -> Tuple[pd.Series, pd.Series]:
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].dropna()
    eq = s / float(s.iloc[0])
    return eq, pd.Series(1.0, index=eq.index)


def _add_window_stats(
    name: str,
    eq: pd.Series,
    inv: pd.Series | None,
    spy_full: PerfStats,
    spy_is: PerfStats,
    spy_oos: PerfStats,
    notes: str,
) -> List[PerfStats]:
    rows = [
        perf_stats(
            name,
            eq,
            inv,
            spy_cagr=spy_full.cagr,
            spy_sharpe=spy_full.sharpe,
            notes=notes,
        )
    ]
    eq_is, inv_is = window_equity(eq, inv, IS_START, IS_END)
    eq_oos, inv_oos = window_equity(eq, inv, OOS_START, OOS_END)
    rows.append(
        perf_stats(
            f"{name}_IS",
            eq_is,
            inv_is,
            spy_cagr=spy_is.cagr,
            spy_sharpe=spy_is.sharpe,
            notes="IS " + notes,
        )
    )
    rows.append(
        perf_stats(
            f"{name}_OOS",
            eq_oos,
            inv_oos,
            spy_cagr=spy_oos.cagr,
            spy_sharpe=spy_oos.sharpe,
            notes="OOS " + notes,
        )
    )
    return rows


def _write_scorecard(
    out_dir: Path,
    reports_dir: Path,
    spy: PerfStats,
    all_stats: List[PerfStats],
    timings: Dict[str, float],
    curves: Dict[str, pd.Series],
    window_note: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d")
    path = out_dir / f"{ts}_edge_hunt_phase6c_vix_curve.md"

    full_winners = []
    for s in all_stats:
        if s.name.startswith("SPY_buy_hold") or s.name.endswith("_IS") or s.name.endswith("_OOS"):
            continue
        ok, gate = passes_phase5_gates(s, spy)
        if ok:
            full_winners.append((s, gate))

    spy_oos = next((s for s in all_stats if s.name == "SPY_buy_hold_OOS"), None)
    oos_winners = []
    if spy_oos is not None:
        for s in all_stats:
            if s.name.endswith("_OOS") and not s.name.startswith("SPY_buy_hold"):
                ok, gate = passes_phase5_gates(s, spy_oos)
                if ok:
                    oos_winners.append((s, gate))

    annual_map = {n: annual_returns(c) for n, c in curves.items()}
    body = [
        "# Edge hunt Phase 6c - VIX term-structure overlay",
        "",
        f"Date: {ts}",
        "",
        "## Objective",
        "",
        "- Unblock Phase 6 VIX curve overlay using IB `VIX` + `VIX3M` daily closes.",
        "- Contango (VIX < VIX3M) = risk-on SPY; backwardation = risk-off (BIL).",
        "- Also test soft weights and multiply by Phase 6 near-miss vol-target.",
        "- Gate: Sharpe > 1.0, Sharpe >= SPY, MDD better than SPY.",
        "",
        "## Window / data",
        "",
        f"- {window_note}",
        f"- Full eval: `{spy.start}` -> `{spy.end}`",
        "- IS: 2018-01-01 -> 2022-12-31; OOS: 2023-01-01 -> 2025-11-26",
        "",
        "## Scorecard",
        "",
        format_table(all_stats, spy),
        "",
        "## Annual returns",
        "",
        format_annual_table(annual_map),
        "",
        "## Notes",
        "",
    ]
    for s in all_stats:
        body.append(f"- **{s.name}**: {s.notes or 'n/a'}")
    body.extend(["", "## Timings", ""])
    for k, v in timings.items():
        body.append(f"- {k}: {format_elapsed(v)}")
    body.extend(["", "## Promotion", ""])

    if full_winners:
        winners_sorted = sorted(full_winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
        best, gate = winners_sorted[0]
        body.append(f"**Promoted (Phase 6c full sample):** `{best.name}` - {gate}")
        body.append("")
        body.append(
            f"Best passer: CAGR={best.cagr:.2%} Sharpe={best.sharpe:.2f} MDD={best.max_drawdown:.2%} "
            f"vs SPY CAGR={spy.cagr:.2%} Sharpe={spy.sharpe:.2f} MDD={spy.max_drawdown:.2%}."
        )
    else:
        body.append("**No Phase-6c full-sample candidate cleared the risk-adjusted gate.**")
        body.append("")
        near = [
            s
            for s in all_stats
            if not s.name.startswith("SPY_buy_hold")
            and not s.name.endswith("_IS")
            and not s.name.endswith("_OOS")
            and abs(s.max_drawdown) + 1e-12 < abs(spy.max_drawdown)
            and s.sharpe + 1e-12 >= spy.sharpe
        ]
        if near:
            body.append("Near-miss (Sharpe>=SPY and better MDD, but Sharpe<=1.0):")
            for s in sorted(near, key=lambda x: x.sharpe, reverse=True):
                body.append(
                    f"- `{s.name}`: CAGR={s.cagr:.2%} Sharpe={s.sharpe:.2f} MDD={s.max_drawdown:.2%}"
                )
            body.append("")
        body.append(
            "Next: keep Phase 6b `Blend_VT60_BV40` as research near-KEEP; "
            "do not fish VIX thresholds further unless a clear structural edge appears."
        )

    body.extend(["", "### OOS robustness (vs SPY_buy_hold_OOS)", ""])
    if oos_winners:
        ow = sorted(oos_winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
        body.append(
            "OOS passers: "
            + ", ".join(
                f"`{s.name}` Sharpe={s.sharpe:.2f} MDD={s.max_drawdown:.2%}" for s, _ in ow
            )
        )
    else:
        body.append("No OOS candidate beat OOS SPY on the Phase 5/6 gate.")
    body.append("")

    path.write_text("\n".join(body), encoding="utf-8")
    if curves:
        pd.DataFrame(curves).to_csv(reports_dir / "equity_curves.csv")
        plot_equity_and_dd(curves, reports_dir / "equity_drawdown.png")

    (reports_dir / "scorecard.json").write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "spy": spy.name,
                "stats": [
                    {
                        "name": s.name,
                        "cagr": s.cagr,
                        "sharpe": s.sharpe,
                        "max_drawdown": s.max_drawdown,
                        "notes": s.notes,
                    }
                    for s in all_stats
                ],
                "winners": [s.name for s, _ in full_winners],
                "oos_winners": [s.name for s, _ in oos_winners],
                "timings": timings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6c VIX curve overlay vs SPY")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--spy-start", default="2010-01-04")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase6c",
    )
    args = ap.parse_args()

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    spy = load_spy_close(
        datetime.strptime(args.spy_start, "%Y-%m-%d"),
        datetime.strptime(args.end, "%Y-%m-%d"),
    )
    macros = load_macro_closes(
        ["BIL", "VIX", "VIX3M"],
        args.spy_start,
        args.end,
        preferred_provider="IB",
        fallback_provider="ALPACA",
    )
    bil = macros.get("BIL")
    vix = macros.get("VIX")
    vix3m = macros.get("VIX3M")
    timings["load"] = time.perf_counter() - t0

    if vix is None or vix.empty or vix3m is None or vix3m.empty:
        logger.error("Missing VIX/VIX3M in TimescaleDB (IB preferred). Aborting.")
        return 1

    logger.info(
        "Loaded SPY n=%d VIX n=%d (%s->%s) VIX3M n=%d (%s->%s) BIL=%s",
        len(spy),
        len(vix),
        vix.index.min().date(),
        vix.index.max().date(),
        len(vix3m),
        vix3m.index.min().date(),
        vix3m.index.max().date(),
        "yes" if bil is not None and not bil.empty else "no",
    )

    spy_eq, spy_inv = _spy_bh(spy, args.start, args.end)
    spy_full = perf_stats("SPY_buy_hold", spy_eq, spy_inv, notes="100% IB SPY")
    spy_is_eq, spy_is_inv = window_equity(spy_eq, spy_inv, IS_START, IS_END)
    spy_oos_eq, spy_oos_inv = window_equity(spy_eq, spy_inv, OOS_START, OOS_END)
    spy_is = perf_stats("SPY_buy_hold_IS", spy_is_eq, spy_is_inv, notes="IS")
    spy_oos = perf_stats("SPY_buy_hold_OOS", spy_oos_eq, spy_oos_inv, notes="OOS")

    all_stats: List[PerfStats] = [spy_full, spy_is, spy_oos]
    curves: Dict[str, pd.Series] = {spy_full.name: spy_eq}

    variants = [
        ("VIXCurve_binary", {"mode": "binary"}),
        ("VIXCurve_binary_min20", {"mode": "binary", "min_vix": 20.0}),
        ("VIXCurve_binary_min25", {"mode": "binary", "min_vix": 25.0}),
        ("VIXCurve_soft", {"mode": "soft"}),
        (
            "VIXCurve_bin_x_VT12",
            {"mode": "binary", "vol_target": 0.12, "vol_lookback": 20, "vol_cap": 1.0},
        ),
        (
            "VIXCurve_soft_x_VT12",
            {"mode": "soft", "vol_target": 0.12, "vol_lookback": 20, "vol_cap": 1.0},
        ),
        (
            "VIXCurve_bin20_x_VT12",
            {
                "mode": "binary",
                "min_vix": 20.0,
                "vol_target": 0.12,
                "vol_lookback": 20,
                "vol_cap": 1.0,
            },
        ),
    ]

    for name, kwargs in variants:
        t0 = time.perf_counter()
        _eq_g, eq_n, inv, notes = simulate_vix_curve_overlay(
            spy,
            vix,
            vix3m,
            bil,
            eval_start=args.start,
            eval_end=args.end,
            cost_bps_rt=args.cost_bps,
            **kwargs,
        )
        timings[f"sim_{name}"] = time.perf_counter() - t0
        rows = _add_window_stats(name, eq_n, inv, spy_full, spy_is, spy_oos, notes)
        all_stats.extend(rows)
        curves[name] = eq_n
        logger.info(
            "%s Sharpe=%.2f MDD=%.1f%% avg_w=%.2f",
            name,
            rows[0].sharpe,
            100.0 * rows[0].max_drawdown,
            float(inv.mean()),
        )

    timings["total"] = time.perf_counter() - t_all
    window_note = (
        f"SPY IB n={len(spy)}; VIX IB n={len(vix)}; VIX3M IB n={len(vix3m)}; "
        f"BIL={'yes' if bil is not None else 'no'}; cost={args.cost_bps:.0f}bps RT"
    )
    path = _write_scorecard(
        args.outdir,
        args.reports_dir,
        spy_full,
        all_stats,
        timings,
        curves,
        window_note,
    )
    logger.info("Wrote %s (total %.1fs)", path, timings["total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
