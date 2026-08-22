#!/usr/bin/env python3
"""
Phase 6b: Vol-targeted SPY core + frozen BigVol sleeve vs SPY buy-and-hold.

Motivation: Phase 6 VolTarget_12pct near-miss (Sharpe 0.79, MDD -16%) needs
more return path without giving back drawdown control. Replace raw SPY in the
Phase 4 70/30 blend with vol-targeted SPY (frozen BigVol params).

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_voltarget_bigvol_blend.py
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
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from run_spy_bigvol_blend import blend_equities, _normalize_ohlc
from spy_benchmark_screen import (
    DEFAULT_BIGVOL_SETUPS,
    strategy_bigvol_portfolio,
)

from utils.research.cta_trend import load_macro_closes
from utils.research.metrics import (
    EVAL_END,
    EVAL_START,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    PerfStats,
    format_elapsed,
    passes_phase5_gates,
    perf_stats,
    window_equity,
)
from utils.research.panel import load_spy_close
from utils.research.report import plot_equity_and_dd, save_fragment
from utils.research.vol_target import simulate_vol_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voltarget_bigvol")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

# Frozen Phase 4 sleeve
ALLOC = 0.10
MAX_POS = 15
STOP = 0.15
COST_BPS_SLEEVE = 10.0
# Phase 6 near-miss vol-target
TARGET_VOL = 0.12
LOOKBACK = 20
CAP = 1.0
VOL_COST_BPS = 5.0


def _spy_bh(spy_close: pd.Series, start: str, end: str) -> Tuple[pd.Series, pd.Series]:
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].dropna()
    eq = s / float(s.iloc[0])
    return eq, pd.Series(1.0, index=eq.index)


def _align_eq(eq: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    out = eq.reindex(idx)
    first = out.first_valid_index()
    if first is not None:
        out.loc[:first] = out.loc[first]
    return out.ffill().fillna(1.0)


def _write_scorecard(
    out_dir: Path,
    reports_dir: Path,
    spy: PerfStats,
    all_stats: List[PerfStats],
    timings: Dict[str, float],
    curves: Dict[str, pd.Series],
    window_note: str,
) -> Path:
    from utils.research.metrics import annual_returns, format_annual_table, format_table

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d")
    path = out_dir / f"{ts}_edge_hunt_phase6b_voltarget_bigvol.md"

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
        "# Edge hunt Phase 6b - VolTarget SPY + BigVol blend",
        "",
        f"Date: {ts}",
        "",
        "## Objective",
        "",
        "- Replace raw SPY in Phase 4 `Blend_SPY70_BV30` with Phase 6 near-miss `VolTarget_12pct_20d_cap1`.",
        "- BigVol sleeve frozen: alloc=10%, max_pos=15, stop=15%, MA10 exit, 10bps RT.",
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
        best, gate = sorted(full_winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)[0]
        body.append(f"**Promoted (Phase 6b full sample):** `{best.name}` - {gate}")
        body.append(
            f"CAGR={best.cagr:.2%} Sharpe={best.sharpe:.2f} MDD={best.max_drawdown:.2%} "
            f"vs SPY CAGR={spy.cagr:.2%} Sharpe={spy.sharpe:.2f} MDD={spy.max_drawdown:.2%}."
        )
    else:
        body.append("**No Phase-6b full-sample candidate cleared the gate.**")
        near = [
            s
            for s in all_stats
            if not s.name.startswith("SPY_buy_hold")
            and not s.name.endswith("_IS")
            and not s.name.endswith("_OOS")
            and abs(s.max_drawdown) < abs(spy.max_drawdown)
            and s.sharpe >= spy.sharpe
        ]
        if near:
            body.append("")
            body.append("Near-miss (Sharpe>=SPY and better MDD, Sharpe<=1.0):")
            for s in sorted(near, key=lambda x: x.sharpe, reverse=True):
                body.append(
                    f"- `{s.name}`: CAGR={s.cagr:.2%} Sharpe={s.sharpe:.2f} MDD={s.max_drawdown:.2%}"
                )
    body.append("")
    if spy_oos is not None:
        body.append("### OOS vs SPY_buy_hold_OOS")
        body.append("")
        if oos_winners:
            body.append(
                "OOS passers: "
                + ", ".join(f"`{s.name}` Sharpe={s.sharpe:.2f}" for s, _ in oos_winners)
            )
        else:
            body.append("No OOS passer vs OOS SPY.")
        body.append("")

    path.write_text("\n".join(body), encoding="utf-8")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "spy": spy.as_dict(),
        "stats": [s.as_dict() for s in all_stats],
        "timings": timings,
        "winners": [s.name for s, _ in full_winners],
        "oos_winners": [s.name for s, _ in oos_winners],
    }
    (reports_dir / "scorecard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if curves:
        pd.DataFrame(curves).to_csv(reports_dir / "equity_curves.csv")
        plot_equity_and_dd(curves, reports_dir / "equity_drawdown.png")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6b VolTarget + BigVol blend")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--spy-start", default="2010-01-04")
    ap.add_argument("--bigvol-setups", type=Path, default=DEFAULT_BIGVOL_SETUPS)
    ap.add_argument("--weights", nargs="+", type=float, default=[0.70, 0.80, 0.60])
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase6b",
    )
    args = ap.parse_args()

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    spy = load_spy_close(
        datetime.strptime(args.spy_start, "%Y-%m-%d"),
        datetime.strptime(args.end, "%Y-%m-%d"),
    )
    bil_map = load_macro_closes(["BIL"], args.spy_start, args.end)
    bil = bil_map.get("BIL")
    timings["load_spy_bil"] = time.perf_counter() - t0

    spy_eq, spy_inv = _spy_bh(spy, args.start, args.end)
    spy_full = perf_stats("SPY_buy_hold", spy_eq, spy_inv, notes="100% IB SPY")
    spy_is_eq, spy_is_inv = window_equity(spy_eq, spy_inv, IS_START, IS_END)
    spy_oos_eq, spy_oos_inv = window_equity(spy_eq, spy_inv, OOS_START, OOS_END)
    spy_is = perf_stats("SPY_buy_hold_IS", spy_is_eq, spy_is_inv, notes="IS")
    spy_oos = perf_stats("SPY_buy_hold_OOS", spy_oos_eq, spy_oos_inv, notes="OOS")

    all_stats: List[PerfStats] = [spy_full, spy_is, spy_oos]
    curves: Dict[str, pd.Series] = {spy_full.name: spy_eq}

    # Vol-target core
    t0 = time.perf_counter()
    _g, vt_eq, vt_inv, vt_notes = simulate_vol_target(
        spy,
        bil,
        eval_start=args.start,
        eval_end=args.end,
        target_vol=TARGET_VOL,
        lookback=LOOKBACK,
        leverage_cap=CAP,
        cost_bps_rt=VOL_COST_BPS,
        rebalance="daily",
    )
    timings["sim_voltarget"] = time.perf_counter() - t0
    vt_a = _align_eq(vt_eq, spy_eq.index)
    vt_inv_a = vt_inv.reindex(spy_eq.index).fillna(0.0)
    vt_stats = perf_stats(
        "VolTarget_12pct_20d_cap1",
        vt_a,
        vt_inv_a,
        spy_cagr=spy_full.cagr,
        spy_sharpe=spy_full.sharpe,
        notes=vt_notes,
    )
    all_stats.append(vt_stats)
    for label, start, end, spy_ref in (
        ("_IS", IS_START, IS_END, spy_is),
        ("_OOS", OOS_START, OOS_END, spy_oos),
    ):
        eq_w, inv_w = window_equity(vt_a, vt_inv_a, start, end)
        all_stats.append(
            perf_stats(
                f"VolTarget_12pct_20d_cap1{label}",
                eq_w,
                inv_w,
                spy_cagr=spy_ref.cagr,
                spy_sharpe=spy_ref.sharpe,
                notes=label.strip("_") + " " + vt_notes,
            )
        )
    curves["VolTarget_12pct_20d_cap1"] = vt_a
    logger.info(
        "VolTarget CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%%",
        vt_stats.cagr * 100,
        vt_stats.sharpe,
        vt_stats.max_drawdown * 100,
    )

    # BigVol sleeve (frozen Phase 4)
    t0 = time.perf_counter()
    eq_bv, inv_bv, bv_notes = strategy_bigvol_portfolio(
        setups_csv=args.bigvol_setups,
        start=args.start,
        end=args.end,
        alloc_frac=ALLOC,
        max_positions=MAX_POS,
        stop_loss_pct=STOP,
        cost_bps_rt=COST_BPS_SLEEVE,
        workers=4,
    )
    timings["sim_bigvol"] = time.perf_counter() - t0
    eq_bv.index = pd.DatetimeIndex(eq_bv.index).tz_localize(None).normalize()
    eq_bv = eq_bv[~eq_bv.index.duplicated(keep="last")].sort_index()
    bv_a = _align_eq(eq_bv, spy_eq.index)
    bv_stats = perf_stats(
        "BigVol_sleeve_10bps",
        bv_a,
        inv_bv.reindex(spy_eq.index).fillna(0.0) if inv_bv is not None else None,
        spy_cagr=spy_full.cagr,
        spy_sharpe=spy_full.sharpe,
        notes=bv_notes,
    )
    all_stats.append(bv_stats)
    curves["BigVol_sleeve_10bps"] = bv_a

    # Reference: raw SPY 70/30 (Phase 4 incumbent shape)
    eq_p4, _ = blend_equities(spy_eq, bv_a, 0.70)
    p4 = perf_stats(
        "Blend_SPY70_BV30",
        eq_p4,
        pd.Series(1.0, index=eq_p4.index),
        spy_cagr=spy_full.cagr,
        spy_sharpe=spy_full.sharpe,
        notes="Phase4-style raw SPY 70% + BigVol 30%",
    )
    all_stats.append(p4)
    curves["Blend_SPY70_BV30"] = eq_p4

    # VolTarget core blends
    for w in args.weights:
        name = f"Blend_VT{int(round(w * 100))}_BV{int(round((1 - w) * 100))}"
        eq_b, inv_b = blend_equities(vt_a, bv_a, w)
        st = perf_stats(
            name,
            eq_b,
            inv_b,
            spy_cagr=spy_full.cagr,
            spy_sharpe=spy_full.sharpe,
            notes=f"VT{w:.0%}/BV{1 - w:.0%}; {vt_notes}; sleeve={bv_notes}",
        )
        all_stats.append(st)
        curves[name] = eq_b
        for label, start, end, spy_ref in (
            ("_IS", IS_START, IS_END, spy_is),
            ("_OOS", OOS_START, OOS_END, spy_oos),
        ):
            eq_w, inv_w = window_equity(eq_b, inv_b, start, end)
            all_stats.append(
                perf_stats(
                    f"{name}{label}",
                    eq_w,
                    inv_w,
                    spy_cagr=spy_ref.cagr,
                    spy_sharpe=spy_ref.sharpe,
                    notes=label.strip("_") + " " + st.notes,
                )
            )
        ok, gate = passes_phase5_gates(st, spy_full)
        logger.info(
            "%s CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%% -> %s",
            name,
            st.cagr * 100,
            st.sharpe,
            st.max_drawdown * 100,
            gate,
        )

    timings["total"] = time.perf_counter() - t_all
    save_fragment(args.reports_dir, "voltarget_bigvol", all_stats, timings, [vt_notes, bv_notes], curves)
    window_note = (
        f"SPY IB n={len(spy)}; BIL={'yes' if bil is not None else 'no'}; "
        f"BigVol setups={args.bigvol_setups.name}; VT={TARGET_VOL:.0%}/{LOOKBACK}d/cap{CAP:g}"
    )
    path = _write_scorecard(
        args.outdir, args.reports_dir, spy_full, all_stats, timings, curves, window_note
    )
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
