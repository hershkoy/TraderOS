#!/usr/bin/env python3
"""
Phase 4: Robustness of frozen Blend_SPY70_BV30.

Tests (no weight/param fishing):
  1) 10 bps RT costs on BigVol sleeve
  2) Walk-forward IS / OOS
  3) Stop 20% stress + costs

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_spy_bigvol_blend_robustness.py
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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

from run_spy_bigvol_blend import blend_equities, _normalize_ohlc
from spy_benchmark_screen import (
    DEFAULT_BIGVOL_SETUPS,
    PerfStats,
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
logger = logging.getLogger("spy_bigvol_robustness")

SPY_WEIGHT = 0.70
ALLOC = 0.10
MAX_POS = 15
COST_BPS = 10.0


def _load_spy(start: str, end: str, provider: str) -> pd.DataFrame:
    try:
        spy = load_spy(start, end, provider=provider)
    except RuntimeError:
        logger.warning("SPY %s failed; fallback ALPACA", provider)
        spy = load_spy(start, end, provider="ALPACA")
    return _clip_dates(_normalize_ohlc(spy), start, end)


def _align_sleeve(eq_bv: pd.Series, inv_bv: pd.Series, spy_idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    eq = eq_bv.copy()
    inv = inv_bv.copy()
    eq.index = pd.DatetimeIndex(eq.index).tz_localize(None).normalize()
    inv.index = pd.DatetimeIndex(inv.index).tz_localize(None).normalize()
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    inv = inv[~inv.index.duplicated(keep="last")].sort_index()
    eq_a = eq.reindex(spy_idx)
    first = eq_a.first_valid_index()
    if first is not None:
        eq_a.loc[:first] = eq_a.loc[first]
    eq_a = eq_a.ffill().fillna(1.0)
    if float(eq_a.iloc[0]) != 0:
        eq_a = eq_a / eq_a.iloc[0]
    inv_a = inv.reindex(spy_idx).fillna(0.0)
    return eq_a, inv_a


def _score_window(
    label: str,
    start: str,
    end: str,
    spy_provider: str,
    setups: Path,
    workers: int,
    stop_loss_pct: float,
    cost_bps_rt: float,
) -> Tuple[PerfStats, PerfStats, PerfStats, Dict[str, pd.Series]]:
    """Return spy_stats, sleeve_stats, blend_stats, curves."""
    spy = _load_spy(start, end, spy_provider)
    eq_bh, inv_bh = strategy_buy_hold(spy)
    spy_st = perf_stats(f"SPY_buy_hold_{label}", eq_bh, inv_bh, notes=f"100% SPY [{start}->{end}]")

    try:
        eq_bv, inv_bv, notes = strategy_bigvol_portfolio(
            setups,
            start,
            end,
            alloc_frac=ALLOC,
            max_positions=MAX_POS,
            stop_loss_pct=stop_loss_pct,
            workers=workers,
            cost_bps_rt=cost_bps_rt,
        )
    except RuntimeError as exc:
        # Setups CSV may not cover early IS years (currently starts ~2023-04).
        if "No BigVol setups" not in str(exc):
            raise
        logger.warning("No BigVol setups in %s [%s->%s]; sleeve idle cash", label, start, end)
        eq_bv = pd.Series(1.0, index=eq_bh.index)
        inv_bv = pd.Series(0.0, index=eq_bh.index)
        notes = f"no setups in window [{start}->{end}]; sleeve idle cash (0% yield)"
    eq_bv_a, inv_bv_a = _align_sleeve(eq_bv, inv_bv, eq_bh.index)
    sleeve_st = perf_stats(
        f"BigVol_sleeve_{label}",
        eq_bv_a,
        inv_bv_a,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes=notes,
    )
    eq_c, inv_c = blend_equities(eq_bh, eq_bv_a, SPY_WEIGHT)
    blend_st = perf_stats(
        f"Blend70_30_{label}",
        eq_c,
        inv_c,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes=(
            f"spy=70% bv=30%; stop={stop_loss_pct:.0%}; cost_rt={cost_bps_rt:.0f}bps; "
            f"sleeve_invested={float(inv_bv_a.mean()):.0%}"
        ),
    )
    curves = {
        f"SPY_{label}": eq_bh,
        f"Sleeve_{label}": eq_bv_a,
        f"Blend70_30_{label}": eq_c,
    }
    return spy_st, sleeve_st, blend_st, curves


def _oos_pass(blend: PerfStats, spy: PerfStats) -> Tuple[bool, str]:
    """Phase 4 OOS rule: Sharpe > SPY OR (better MDD and CAGR within ~2pp)."""
    sharpe_ok = blend.sharpe >= spy.sharpe - 1e-9
    mdd_ok = abs(blend.max_drawdown) + 1e-9 < abs(spy.max_drawdown)
    cagr_ok = blend.cagr + 1e-9 >= spy.cagr - 0.02
    if sharpe_ok:
        return True, "OOS_PASS (Sharpe>=SPY)"
    if mdd_ok and cagr_ok:
        return True, "OOS_PASS (MDD+CAGR)"
    return False, "OOS_FAIL"


def _demote_on_costs(blend: PerfStats, spy: PerfStats) -> bool:
    """Kill promote if CAGR >2pp below SPY AND Sharpe <= SPY."""
    cagr_bad = blend.cagr + 1e-9 < spy.cagr - 0.02
    sharpe_bad = blend.sharpe <= spy.sharpe + 1e-9
    return cagr_bad and sharpe_bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 blend robustness")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2025-11-26")
    ap.add_argument("--is-end", default="2022-06-30")
    ap.add_argument("--oos-start", default="2022-07-01")
    ap.add_argument("--spy-provider", default="IB")
    ap.add_argument("--bigvol-setups", type=Path, default=DEFAULT_BIGVOL_SETUPS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase4",
    )
    args = ap.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    timings: Dict[str, float] = {}
    all_stats: List[PerfStats] = []
    all_curves: Dict[str, pd.Series] = {}

    # --- 1) Full window baseline (0 cost) + costs ---
    t0 = time.perf_counter()
    spy0, sleeve0, blend0, c0 = _score_window(
        "full_0bps",
        args.start,
        args.end,
        args.spy_provider,
        args.bigvol_setups,
        args.workers,
        stop_loss_pct=0.15,
        cost_bps_rt=0.0,
    )
    timings["full_0bps"] = time.perf_counter() - t0
    all_stats.extend([spy0, sleeve0, blend0])
    all_curves.update(c0)

    t0 = time.perf_counter()
    spy_c, sleeve_c, blend_c, cc = _score_window(
        "full_10bps",
        args.start,
        args.end,
        args.spy_provider,
        args.bigvol_setups,
        args.workers,
        stop_loss_pct=0.15,
        cost_bps_rt=COST_BPS,
    )
    timings["full_10bps"] = time.perf_counter() - t0
    # Score costed blend vs full-window SPY (0bps SPY path) for apples-to-apples gates
    eq_b10 = cc["Blend70_30_full_10bps"].reindex(c0["SPY_full_0bps"].index).ffill().bfill()
    if float(eq_b10.iloc[0]) != 0:
        eq_b10 = eq_b10 / eq_b10.iloc[0]
    blend_c_vs_spy0 = perf_stats(
        "Blend70_30_full_10bps",
        eq_b10,
        pd.Series(1.0, index=eq_b10.index),
        spy_cagr=spy0.cagr,
        spy_sharpe=spy0.sharpe,
        notes=blend_c.notes,
    )
    all_stats.extend([spy_c, sleeve_c, blend_c_vs_spy0])
    all_curves.update(cc)

    costs_demote = _demote_on_costs(blend_c_vs_spy0, spy0)
    ok_costs, gate_costs = passes_gates(blend_c_vs_spy0, spy0)

    # --- 2) Walk-forward ---
    t0 = time.perf_counter()
    spy_is, sleeve_is, blend_is, cis = _score_window(
        "IS",
        args.start,
        args.is_end,
        args.spy_provider,
        args.bigvol_setups,
        args.workers,
        stop_loss_pct=0.15,
        cost_bps_rt=COST_BPS,
    )
    timings["IS"] = time.perf_counter() - t0
    all_stats.extend([spy_is, sleeve_is, blend_is])
    all_curves.update(cis)

    t0 = time.perf_counter()
    spy_oos, sleeve_oos, blend_oos, coos = _score_window(
        "OOS",
        args.oos_start,
        args.end,
        args.spy_provider,
        args.bigvol_setups,
        args.workers,
        stop_loss_pct=0.15,
        cost_bps_rt=COST_BPS,
    )
    timings["OOS"] = time.perf_counter() - t0
    all_stats.extend([spy_oos, sleeve_oos, blend_oos])
    all_curves.update(coos)

    oos_ok, oos_gate = _oos_pass(blend_oos, spy_oos)
    is_ok, is_gate = passes_gates(blend_is, spy_is)

    # --- 3) Stop 20% stress + costs ---
    t0 = time.perf_counter()
    spy_s, sleeve_s, blend_s, cs = _score_window(
        "stress20_10bps",
        args.start,
        args.end,
        args.spy_provider,
        args.bigvol_setups,
        args.workers,
        stop_loss_pct=0.20,
        cost_bps_rt=COST_BPS,
    )
    timings["stress20"] = time.perf_counter() - t0
    # Score stress blend vs full-window spy0
    eq_st = cs["Blend70_30_stress20_10bps"].reindex(c0["SPY_full_0bps"].index).ffill().bfill()
    if float(eq_st.iloc[0]) != 0:
        eq_st = eq_st / eq_st.iloc[0]
    blend_s_vs = perf_stats(
        "Blend70_30_stress20_10bps",
        eq_st,
        pd.Series(1.0, index=eq_st.index),
        spy_cagr=spy0.cagr,
        spy_sharpe=spy0.sharpe,
        notes=blend_s.notes,
    )
    all_stats.extend([spy_s, sleeve_s, blend_s_vs])
    all_curves.update(cs)
    stress_ok, stress_gate = passes_gates(blend_s_vs, spy0)
    stress_demote = _demote_on_costs(blend_s_vs, spy0)

    timings["total"] = time.perf_counter() - t_all

    # --- Verdict ---
    keep = (not costs_demote) and oos_ok and (not stress_demote)
    # Soft: stress can fail gates but not hard-demote if costs rule OK; plan says confirm not knife-edge
    # Demote if costs kill OR OOS fails. Stress demote only if costs-style kill rule hits.
    if costs_demote or (not oos_ok) or stress_demote:
        keep = False
    else:
        keep = True

    # Persist
    curves_path = args.reports_dir / "equity_curves.csv"
    pd.DataFrame(all_curves).to_csv(curves_path)
    stats_path = args.reports_dir / "scorecard.json"
    stats_path.write_text(json.dumps([asdict(s) for s in all_stats], indent=2), encoding="utf-8")

    args.outdir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y-%m-%d")
    log_path = args.outdir / f"{ts}_edge_hunt_phase4_scorecard.md"

    lines = [
        "# Edge hunt Phase 4 - Blend_SPY70_BV30 robustness",
        "",
        f"Date: {ts}",
        "",
        "## Locked settings",
        "",
        f"- Blend: **70% SPY / 30% BigVol** (frozen)",
        f"- Sleeve: alloc={ALLOC:.0%}, max_pos={MAX_POS}, MA10 exit",
        f"- Costs test: **{COST_BPS:.0f} bps RT** on sleeve trades only",
        f"- Walk-forward IS: `{args.start}` -> `{args.is_end}`",
        f"- Walk-forward OOS: `{args.oos_start}` -> `{args.end}`",
        f"- Stress: stop **20%** + {COST_BPS:.0f}bps RT",
        f"- Setups source: `{args.bigvol_setups.name}`",
        "",
        "## Data note",
        "",
        "Weekly BigVol setups CSV confirms begin ~2023-04. The IS half (through 2022-06) "
        "has an idle sleeve (30% cash at 0% yield + 70% SPY). OOS is the binding robustness "
        "test for the sleeve; IS still validates blend behavior with no satellite signals.",
        "",
        "## Scorecard",
        "",
        format_table(all_stats, spy0),
        "",
        "## Test results",
        "",
        "### 1) Costs (full window, 10bps RT)",
        "",
        f"- Blend 0bps vs SPY: CAGR={blend0.cagr:.2%} Sharpe={blend0.sharpe:.2f} MDD={blend0.max_drawdown:.2%} "
        f"-> {passes_gates(blend0, spy0)[1]}",
        f"- Blend 10bps vs SPY: CAGR={blend_c_vs_spy0.cagr:.2%} Sharpe={blend_c_vs_spy0.sharpe:.2f} "
        f"MDD={blend_c_vs_spy0.max_drawdown:.2%} -> {gate_costs}",
        f"- Costs demote rule (CAGR>2pp below AND Sharpe<=SPY): **{'HIT' if costs_demote else 'clear'}**",
        "",
        "### 2) Walk-forward (10bps RT, stop 15%)",
        "",
        f"- IS blend vs IS SPY: CAGR={blend_is.cagr:.2%} Sharpe={blend_is.sharpe:.2f} "
        f"MDD={blend_is.max_drawdown:.2%} -> {is_gate}",
        f"- OOS blend vs OOS SPY: CAGR={blend_oos.cagr:.2%} Sharpe={blend_oos.sharpe:.2f} "
        f"MDD={blend_oos.max_drawdown:.2%} -> **{oos_gate}**",
        "",
        "### 3) Stop 20% stress (full window, 10bps RT)",
        "",
        f"- Blend stress vs SPY: CAGR={blend_s_vs.cagr:.2%} Sharpe={blend_s_vs.sharpe:.2f} "
        f"MDD={blend_s_vs.max_drawdown:.2%} -> {stress_gate}",
        f"- Stress demote rule: **{'HIT' if stress_demote else 'clear'}**",
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
    lines.append("## Verdict")
    lines.append("")
    if keep:
        lines.append("**KEEP promote** for `Blend_SPY70_BV30`.")
        lines.append("")
        lines.append(
            "Costs, OOS, and stop-20% stress did not trip demote rules. "
            "Treat as capital-allocation on BigVol sleeve — next work may improve sleeve expectancy, "
            "not blend weights."
        )
    else:
        lines.append("**DEMOTE** `Blend_SPY70_BV30`.")
        lines.append("")
        reasons = []
        if costs_demote:
            reasons.append("full-window +10bps demote rule")
        if not oos_ok:
            reasons.append(f"OOS failed ({oos_gate})")
        if stress_demote:
            reasons.append("stop-20% + costs demote rule")
        lines.append("Reasons: " + "; ".join(reasons) + ".")
        lines.append("")
        lines.append(
            "Next: new alpha hunt or declare no robust SPY-beater on this dataset "
            "(do not keep optimizing blend weights)."
        )
    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")

    # README update
    readme = args.outdir / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        link = f"| {ts} | [Phase 4 blend robustness]({ts}_edge_hunt_phase4_scorecard.md) |"
        if "Phase 4" not in text:
            text = text.replace(
                "| 2026-08-22 | [Phase 3 SPY+BigVol blend](2026-08-22_edge_hunt_phase3_scorecard.md) |",
                "| 2026-08-22 | [Phase 3 SPY+BigVol blend](2026-08-22_edge_hunt_phase3_scorecard.md) |\n" + link,
            )
        if "run_spy_bigvol_blend_robustness" not in text:
            text = text.replace(
                "python scripts\\research\\run_spy_bigvol_blend.py",
                "python scripts\\research\\run_spy_bigvol_blend.py\n"
                "python scripts\\research\\run_spy_bigvol_blend_robustness.py",
            )
        if "edge_hunt_phase4" not in text:
            text = text.replace(
                "`reports/edge_hunt_phase3/`.",
                "`reports/edge_hunt_phase3/`, `reports/edge_hunt_phase4/`.",
            )
        verdict = (
            f"## Verdict (2026-08-22)\n\n"
            f"Phase 4: **{'KEEP' if keep else 'DEMOTE'}** `Blend_SPY70_BV30` "
            f"(costs={'fail' if costs_demote else 'ok'}, OOS={oos_gate}, "
            f"stress20={'fail' if stress_demote else 'ok'}).\n"
        )
        text = re.sub(
            r"## Verdict \(2026-08-22\)\n\n.*",
            verdict.rstrip() + "\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
        readme.write_text(text, encoding="utf-8")

    print("\n=== Phase 4 robustness ===")
    print(format_table(all_stats, spy0))
    print(f"\nCosts demote: {costs_demote} | OOS: {oos_gate} | Stress demote: {stress_demote}")
    print(f"VERDICT: {'KEEP' if keep else 'DEMOTE'} Blend_SPY70_BV30")
    print(f"Wrote {log_path}")
    print(f"Total: {_format_elapsed(timings['total'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
