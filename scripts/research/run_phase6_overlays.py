#!/usr/bin/env python3
"""
Phase 6: Portfolio overlays vs SPY (after Phase 5 stock-signal failure).

Candidates:
  1) SPY vol-target 10% / 12% (20d lookback, cap 1.0 and 1.5)
  2) SPY 70% + CTA multi-asset trend 30% (SMA200 long/short on available macros)
  3) Low-vol basket (realized 60d vol among PIT top-500 liquid) — no ROE/FCF
  4) VIX curve — skipped unless VIX data present

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_phase6_overlays.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.cta_trend import blend_spy_cta, load_macro_closes, simulate_cta_sleeve
from utils.research.low_vol import simulate_low_vol_basket
from utils.research.metrics import (
    EVAL_END,
    EVAL_START,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    PerfStats,
    perf_stats,
    window_equity,
)
from utils.research.panel import list_daily_symbols, load_spy_close, load_wide_panels
from utils.research.report import save_fragment, write_phase6_scorecard
from utils.research.vol_target import simulate_vol_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phase6_overlays")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEBUG_LOG = ROOT / "debug-d1a83a.log"
MACRO_CANDIDATES = ["SPY", "TLT", "GLD", "USO", "UUP", "IEF", "DBC", "BIL"]


# region agent log
def _dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "p6") -> None:
    try:
        payload = {
            "sessionId": "d1a83a",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# endregion


def _spy_buy_hold(spy_close, start: str, end: str):
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].dropna()
    eq = s / float(s.iloc[0])
    inv = pd.Series(1.0, index=eq.index)
    return eq, inv


def _align(eq: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    out = eq.reindex(idx)
    first = out.first_valid_index()
    if first is not None:
        out.loc[:first] = out.loc[first]
    return out.ffill().fillna(1.0)


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 portfolio overlays vs SPY")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--spy-start", default="2010-01-04")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--vol-cost-bps", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-lowvol", action="store_true")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase6",
    )
    args = ap.parse_args()

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    spy = load_spy_close(
        datetime.strptime(args.spy_start, "%Y-%m-%d"),
        datetime.strptime(args.end, "%Y-%m-%d"),
    )
    bil_map = load_macro_closes(
        ["BIL"],
        args.spy_start,
        args.end,
        preferred_provider="IB",
        fallback_provider="ALPACA",
    )
    bil = bil_map.get("BIL")
    timings["load_spy_bil"] = time.perf_counter() - t0
    # region agent log
    _dbg(
        "H1",
        "run_phase6_overlays.py:load",
        "spy_bil_loaded",
        {"spy_n": len(spy), "spy_start": str(spy.index[0].date()), "bil_n": 0 if bil is None else len(bil)},
    )
    # endregion

    spy_eq, spy_inv = _spy_buy_hold(spy, args.start, args.end)
    spy_full = perf_stats("SPY_buy_hold", spy_eq, spy_inv, notes="100% IB SPY")
    spy_is_eq, spy_is_inv = window_equity(spy_eq, spy_inv, IS_START, IS_END)
    spy_oos_eq, spy_oos_inv = window_equity(spy_eq, spy_inv, OOS_START, OOS_END)
    spy_is = perf_stats("SPY_buy_hold_IS", spy_is_eq, spy_is_inv, notes="IS")
    spy_oos = perf_stats("SPY_buy_hold_OOS", spy_oos_eq, spy_oos_inv, notes="OOS")

    all_stats: List[PerfStats] = [spy_full, spy_is, spy_oos]
    curves: Dict[str, pd.Series] = {spy_full.name: spy_eq}
    notes_extra: List[str] = []

    # --- Vol target variants ---
    vol_specs = [
        ("VolTarget_12pct_20d_cap1", 0.12, 20, 1.0, "daily"),
        ("VolTarget_10pct_20d_cap1", 0.10, 20, 1.0, "daily"),
        ("VolTarget_12pct_20d_cap1p5", 0.12, 20, 1.5, "daily"),
        ("VolTarget_12pct_60d_cap1", 0.12, 60, 1.0, "weekly"),
    ]
    for name, tv, lb, cap, rebal in vol_specs:
        t0 = time.perf_counter()
        _eg, eq_n, inv, notes = simulate_vol_target(
            spy,
            bil,
            eval_start=args.start,
            eval_end=args.end,
            target_vol=tv,
            lookback=lb,
            leverage_cap=cap,
            cost_bps_rt=args.vol_cost_bps,
            rebalance=rebal,
        )
        timings[f"sim_{name}"] = time.perf_counter() - t0
        eq_a = _align(eq_n, spy_eq.index)
        inv_a = inv.reindex(spy_eq.index).fillna(0.0)
        rows = _add_window_stats(name, eq_a, inv_a, spy_full, spy_is, spy_oos, notes)
        all_stats.extend(rows)
        curves[name] = eq_a
        # region agent log
        _dbg(
            "H2",
            "run_phase6_overlays.py:vol",
            name,
            {
                "cagr": rows[0].cagr,
                "sharpe": rows[0].sharpe,
                "mdd": rows[0].max_drawdown,
                "avg_w": float(inv_a.mean()),
            },
        )
        # endregion
        logger.info(
            "%s CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%%",
            name,
            rows[0].cagr * 100,
            rows[0].sharpe,
            rows[0].max_drawdown * 100,
        )

    # --- CTA sleeve ---
    t0 = time.perf_counter()
    macros = load_macro_closes(
        MACRO_CANDIDATES,
        args.spy_start,
        args.end,
        preferred_provider="IB",
        fallback_provider="ALPACA",
    )
    timings["load_macros"] = time.perf_counter() - t0
    # Prefer trend assets excluding pure cash
    cta_syms = [s for s in ["SPY", "TLT", "GLD", "USO", "UUP", "IEF", "DBC"] if s in macros]
    # region agent log
    _dbg(
        "H3",
        "run_phase6_overlays.py:macros",
        "macro_coverage",
        {"loaded": list(macros.keys()), "cta_syms": cta_syms},
    )
    # endregion
    if len(cta_syms) >= 2:
        close_cta = pd.DataFrame({s: macros[s] for s in cta_syms}).dropna(how="any")
        t0 = time.perf_counter()
        _cg, cta_eq, _exp, cta_notes = simulate_cta_sleeve(
            close_cta,
            eval_start=args.start,
            eval_end=args.end,
            method="sma200",
            cost_bps_rt=args.cost_bps,
            rebalance="monthly",
        )
        timings["sim_cta"] = time.perf_counter() - t0
        cta_a = _align(cta_eq, spy_eq.index)
        blend = blend_spy_cta(spy_eq, cta_a, spy_weight=0.70)
        blend_a = _align(blend, spy_eq.index)
        for name, eq, notes in [
            ("CTA_sma200_sleeve", cta_a, cta_notes),
            ("Blend_SPY70_CTA30", blend_a, f"spy70/cta30; {cta_notes}"),
        ]:
            inv = pd.Series(1.0, index=eq.index)
            rows = _add_window_stats(name, eq, inv, spy_full, spy_is, spy_oos, notes)
            all_stats.extend(rows)
            curves[name] = eq
            logger.info(
                "%s CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%%",
                name,
                rows[0].cagr * 100,
                rows[0].sharpe,
                rows[0].max_drawdown * 100,
            )
        notes_extra.append(f"CTA assets used: {cta_syms}")
    else:
        notes_extra.append(
            f"CTA skipped / incomplete: only {cta_syms} available. "
            "Ingest TLT/GLD/USO/UUP via scripts/research/_ingest_p6_macros.py"
        )
        logger.warning(notes_extra[-1])

    # --- Low-vol basket ---
    if not args.skip_lowvol:
        t0 = time.perf_counter()
        symbols = list_daily_symbols("ALPACA", "1d")
        panels = load_wide_panels(
            symbols,
            start=datetime(2017, 11, 29),
            end=datetime.strptime(args.end, "%Y-%m-%d"),
            provider="ALPACA",
            workers=args.workers,
        )
        timings["load_panel"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _lg, lv_eq, lv_inv, lv_notes = simulate_low_vol_basket(
            panels["close"],
            panels["volume"],
            eval_start=args.start,
            eval_end=args.end,
            liquid_n=500,
            hold_n=50,
            vol_window=60,
            cost_bps_rt=args.cost_bps,
        )
        timings["sim_lowvol"] = time.perf_counter() - t0
        lv_a = _align(lv_eq, spy_eq.index)
        inv_a = lv_inv.reindex(spy_eq.index).fillna(0.0)
        rows = _add_window_stats("LowVol60_top50", lv_a, inv_a, spy_full, spy_is, spy_oos, lv_notes)
        all_stats.extend(rows)
        curves["LowVol60_top50"] = lv_a
        # region agent log
        _dbg(
            "H4",
            "run_phase6_overlays.py:lowvol",
            "LowVol60_top50",
            {"cagr": rows[0].cagr, "sharpe": rows[0].sharpe, "mdd": rows[0].max_drawdown},
        )
        # endregion
        logger.info(
            "LowVol60_top50 CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%%",
            rows[0].cagr * 100,
            rows[0].sharpe,
            rows[0].max_drawdown * 100,
        )

    notes_extra.append("VIX term-structure overlay skipped: no VIX/VXV in market_data.")
    timings["total"] = time.perf_counter() - t_all

    save_fragment(args.reports_dir, "phase6", all_stats, timings, notes_extra, curves)
    window_note = (
        f"SPY IB n={len(spy)}; BIL={'yes' if bil is not None else 'no'}; "
        f"macros={sorted(macros.keys())}; CTA assets={cta_syms if len(cta_syms) >= 2 else 'n/a'}"
    )
    path = write_phase6_scorecard(
        args.outdir,
        args.reports_dir,
        spy_full,
        all_stats,
        timings,
        window_note,
        curves,
        extra_sections=["## Data gaps"] + [f"- {n}" for n in notes_extra],
    )
    # region agent log
    _dbg(
        "H5",
        "run_phase6_overlays.py:done",
        "scorecard_written",
        {
            "path": str(path),
            "n_stats": len(all_stats),
            "best_non_spy_sharpe": max(
                (s.sharpe for s in all_stats if not s.name.startswith("SPY_buy_hold")),
                default=None,
            ),
        },
        run_id="post-fix",
    )
    # endregion
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
