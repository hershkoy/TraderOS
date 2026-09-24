#!/usr/bin/env python3
"""Capital-constrained fixes for ROC A3 vs SPY (holdout 2018-2026).

Addresses the infinite-capital / toxic-FCFS failure:
  - daily deepest-ROC ranking into free slots
  - SPY SMA200 regime + relative-weakness filters
  - time-stop velocity (7 sessions)
  - deeper-dip quality gate (entry_roc <= -5)
  - N-slot fractional book (1 / 4 / 5)
  - idle-cash Rf credit (default 4% annual)

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_roc_rebound_capital_fix.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.run_roc_rebound_etf_grid import (  # noqa: E402
    DEFAULT_OUT,
    ETF_UNIVERSE,
    load_etf_frames,
)
from utils.research.metrics import passes_phase5_gates, perf_stats  # noqa: E402
from utils.research.roc_rebound import (  # noqa: E402
    book_from_trades,
    closes_panel_from_frames,
    drop_top_n_book,
    spy_buy_hold_equity,
)
from utils.research.roc_rebound_capital import (  # noqa: E402
    collect_variant_trades,
    n_slot_equity,
    occupancy_ranked,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("roc_capital_fix")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

A3 = {"require_sma200": True, "vol_scale": True, "vol_k": 1.5, "vol_lookback": 20}

# Fixed overlay stack — not a free parameter fish.
VARIANTS: Sequence[Tuple[str, Dict[str, Any]]] = (
    ("B0_A3_rank1", {**A3, "max_open": 1, "rank_by": "deepest_roc"}),
    (
        "B1_A3_spyRegime_rank1",
        {**A3, "require_spy_sma200": True, "max_open": 1, "rank_by": "deepest_roc"},
    ),
    (
        "B2_A3_relWeak_rank1",
        {**A3, "require_rel_weak": True, "max_open": 1, "rank_by": "deepest_roc"},
    ),
    (
        "B3_A3_tstop7_rank1",
        {**A3, "time_stop_bars": 7, "max_open": 1, "rank_by": "deepest_roc"},
    ),
    (
        "B4_A3_regime_tstop7_rank1",
        {
            **A3,
            "require_spy_sma200": True,
            "time_stop_bars": 7,
            "max_open": 1,
            "rank_by": "deepest_roc",
        },
    ),
    (
        "B5_A3_regime_tstop7_deep5_rank1",
        {
            **A3,
            "require_spy_sma200": True,
            "time_stop_bars": 7,
            "min_entry_roc": -5.0,
            "max_open": 1,
            "rank_by": "deepest_roc",
        },
    ),
    (
        "B6_A3_regime_tstop7_rank4",
        {
            **A3,
            "require_spy_sma200": True,
            "time_stop_bars": 7,
            "max_open": 4,
            "rank_by": "deepest_roc",
        },
    ),
    (
        "B7_A3_regime_tstop7_rank5",
        {
            **A3,
            "require_spy_sma200": True,
            "time_stop_bars": 7,
            "max_open": 5,
            "rank_by": "deepest_roc",
        },
    ),
    (
        "B8_A3_regime_relWeak_tstop7_rank4",
        {
            **A3,
            "require_spy_sma200": True,
            "require_rel_weak": True,
            "time_stop_bars": 7,
            "max_open": 4,
            "rank_by": "deepest_roc",
        },
    ),
    (
        "B9_A3_regime_tstop7_rank4_rf4",
        {
            **A3,
            "require_spy_sma200": True,
            "time_stop_bars": 7,
            "max_open": 4,
            "rank_by": "deepest_roc",
            "rf_annual": 0.04,
        },
    ),
)


def _filter_entry(
    trades: List[Dict[str, Any]],
    *,
    start: Optional[str],
    end: Optional[str],
) -> List[Dict[str, Any]]:
    out = []
    for tr in trades:
        ed = pd.Timestamp(tr["entry_date"])
        if start is not None and ed < pd.Timestamp(start):
            continue
        if end is not None and ed > pd.Timestamp(end):
            continue
        out.append(tr)
    return out


def _stats_dict(st: Any) -> Dict[str, Any]:
    return {
        "sharpe": round(st.sharpe, 3),
        "cagr": round(st.cagr, 4),
        "ann_vol": round(st.ann_vol, 4),
        "max_drawdown": round(st.max_drawdown, 4),
        "pct_time_invested": round(st.pct_time_invested, 4),
        "total_return": round(st.total_return, 4),
        "n_trades": st.n_trades,
        "profit_factor": round(st.profit_factor, 3),
        "expectancy": round(st.expectancy, 4),
        "win_rate": round(st.win_rate, 4),
    }


def evaluate(
    *,
    label: str,
    overlay: Dict[str, Any],
    frames: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    spy_close: pd.Series,
    friction_pct: float,
    hold_start: str,
    hold_end: str,
) -> Dict[str, Any]:
    max_open = int(overlay.get("max_open", 1))
    rank_by = str(overlay.get("rank_by", "deepest_roc"))
    rf = float(overlay.get("rf_annual", 0.0))

    raw = collect_variant_trades(
        frames,
        friction_pct=friction_pct,
        overlay=overlay,
        spy_close=spy_close,
    )
    kept, meta = occupancy_ranked(raw, max_open=max_open, rank_by=rank_by)
    hold_kept = _filter_entry(kept, start=hold_start, end=hold_end)
    hold_raw = _filter_entry(raw, start=hold_start, end=hold_end)
    book = book_from_trades(hold_kept)
    book_u = book_from_trades(hold_raw)

    eq, inv = n_slot_equity(
        kept,
        panel,
        max_open=max_open,
        eval_start=hold_start,
        eval_end=hold_end,
        rf_annual=rf,
    )
    spy_eq, spy_inv = spy_buy_hold_equity(
        spy_close, eval_start=hold_start, eval_end=hold_end
    )
    if not eq.empty and not spy_eq.empty:
        common = eq.index.intersection(spy_eq.index)
        eq = eq.reindex(common).ffill().bfill()
        inv = inv.reindex(common).fillna(0.0)
        spy_eq = spy_eq.reindex(common)
        spy_eq = spy_eq / float(spy_eq.iloc[0])
        spy_inv = spy_inv.reindex(common).fillna(1.0)
        eq = eq / float(eq.iloc[0])

    trade_rets = [float(t["ret_pct"]) / 100.0 for t in hold_kept]
    spy_st = perf_stats("SPY", spy_eq, spy_inv, notes="buy_hold")
    sleeve_st = perf_stats(
        label,
        eq,
        inv,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes="max_open=%s rf=%.2f" % (max_open, rf),
        trade_rets=trade_rets,
    )
    ok, reason = passes_phase5_gates(sleeve_st, spy_st)
    drop3 = drop_top_n_book(hold_kept, 3)

    LOG.info(
        "%-40s raw_n=%4d occ_n=%3d PF=%s E=%s | Sharpe=%.2f MDD=%.1f%% inv=%.0f%% "
        "| SPY Sharpe=%.2f MDD=%.1f%% | %s",
        label,
        len(hold_raw),
        book.get("n"),
        book.get("profit_factor"),
        book.get("expectancy_pct"),
        sleeve_st.sharpe,
        100.0 * sleeve_st.max_drawdown,
        100.0 * sleeve_st.pct_time_invested,
        spy_st.sharpe,
        100.0 * spy_st.max_drawdown,
        "PASS" if ok else "FAIL",
    )
    return {
        "label": label,
        "overlay": {k: v for k, v in overlay.items()},
        "occupancy": meta,
        "holdout_unconstrained_book": book_u,
        "holdout_occ_book": book,
        "drop_top3": drop3,
        "sleeve": _stats_dict(sleeve_st),
        "spy": _stats_dict(spy_st),
        "gate_pass": bool(ok),
        "gate_reason": reason,
        "excess_sharpe": round(sleeve_st.sharpe - spy_st.sharpe, 3),
        "excess_cagr": round(sleeve_st.cagr - spy_st.cagr, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ROC capital-constraint fix grid")
    ap.add_argument("--friction", type=float, default=0.10)
    ap.add_argument("--holdout-start", type=str, default="2018-01-01")
    ap.add_argument("--holdout-end", type=str, default="2026-12-31")
    ap.add_argument("--start", type=str, default="1993-01-01")
    ap.add_argument("--rf-annual", type=float, default=0.04, help="Default Rf for B9")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--symbols", type=str, default=",".join(ETF_UNIVERSE))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if "SPY" not in symbols:
        symbols = ["SPY"] + symbols

    t0 = time.perf_counter()
    frames = load_etf_frames(
        symbols,
        start=datetime.fromisoformat(args.start),
        end=None,
        workers=args.workers,
    )
    panel = closes_panel_from_frames(frames)
    spy_close = panel["SPY"]
    LOG.info("Panel %s .. %s (%d days)", panel.index[0].date(), panel.index[-1].date(), len(panel))

    results: List[Dict[str, Any]] = []
    for label, overlay in VARIANTS:
        ov = dict(overlay)
        if "rf_annual" in ov and ov["rf_annual"] and args.rf_annual != 0.04:
            ov["rf_annual"] = float(args.rf_annual)
        results.append(
            evaluate(
                label=label,
                overlay=ov,
                frames=frames,
                panel=panel,
                spy_close=spy_close,
                friction_pct=float(args.friction),
                hold_start=args.holdout_start,
                hold_end=args.holdout_end,
            )
        )

    passed = [r for r in results if r["gate_pass"]]
    best = max(
        results,
        key=lambda r: (
            1 if r["gate_pass"] else 0,
            r["sleeve"]["sharpe"],
            r["holdout_occ_book"].get("profit_factor") or 0,
        ),
    )
    if passed:
        verdict = "PASS_SOME_FIXES"
    elif best["sleeve"]["sharpe"] >= 0.7 and (best["holdout_occ_book"].get("profit_factor") or 0) >= 1.4:
        verdict = "SOFT_IMPROVE_NO_GATE"
    else:
        verdict = "FAIL_STILL_BEHIND_SPY"
    LOG.info("Best by Sharpe: %s (Sharpe=%.2f gate=%s)", best["label"], best["sleeve"]["sharpe"], best["gate_pass"])
    LOG.info("Verdict: %s", verdict)

    elapsed = time.perf_counter() - t0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "friction_pct": args.friction,
        "holdout_start": args.holdout_start,
        "holdout_end": args.holdout_end,
        "verdict": verdict,
        "best": best["label"],
        "results": results,
        "elapsed_sec": round(elapsed, 2),
    }
    path = outdir / "roc_rebound_capital_fix_summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    rows = []
    for r in results:
        rows.append(
            {
                "label": r["label"],
                "gate": r["gate_pass"],
                "reason": r["gate_reason"],
                "occ_n": r["holdout_occ_book"].get("n"),
                "PF": r["holdout_occ_book"].get("profit_factor"),
                "E": r["holdout_occ_book"].get("expectancy_pct"),
                "sharpe": r["sleeve"]["sharpe"],
                "mdd": r["sleeve"]["max_drawdown"],
                "cagr": r["sleeve"]["cagr"],
                "inv": r["sleeve"]["pct_time_invested"],
                "spy_sharpe": r["spy"]["sharpe"],
                "spy_mdd": r["spy"]["max_drawdown"],
                "excess_sharpe": r["excess_sharpe"],
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "roc_rebound_capital_fix.csv", index=False)
    LOG.info("Wrote %s", path)
    LOG.info("Wall-clock %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
