#!/usr/bin/env python3
"""Frozen A3: pre-declared 2018-2026 holdout + single-slot occupancy vs SPY.

Gates (edge-hunt Phase 5 style):
  Sharpe > 1.0 AND Sharpe >= SPY AND MDD better than SPY.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_roc_rebound_occupancy.py
  python scripts\\research\\run_roc_rebound_occupancy.py --friction 0.10 --holdout-start 2018-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.run_roc_rebound_etf_grid import (  # noqa: E402
    DEFAULT_OUT,
    ETF_UNIVERSE,
    load_etf_frames,
)
from utils.research.metrics import (  # noqa: E402
    passes_phase5_gates,
    perf_stats,
)
from utils.research.roc_rebound import (  # noqa: E402
    YEAR_BUCKETS,
    backtest_roc_rebound,
    book_from_trades,
    closes_panel_from_frames,
    drop_top_n_book,
    geom_mean_year_pf,
    occupancy_fcfs,
    single_slot_equity,
    spy_buy_hold_equity,
    year_bucket_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("roc_rebound_occ")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

A3_OVERLAY = {
    "require_sma200": True,
    "vol_scale": True,
    "vol_k": 1.5,
    "vol_lookback": 20,
}


def _frame_to_bt(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if out.columns[0] != "ts":
            out = out.rename(columns={out.columns[0]: "ts"})
    return out


def collect_a3_trades(
    frames: Dict[str, pd.DataFrame], *, friction_pct: float
) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    for sym, raw in sorted(frames.items()):
        res = backtest_roc_rebound(
            _frame_to_bt(raw),
            symbol=sym,
            friction_pct=friction_pct,
            **A3_OVERLAY,
        )
        trades.extend(res.get("trades") or [])
    return trades


def _filter_entry(
    trades: List[Dict[str, Any]],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tr in trades:
        ed = pd.Timestamp(tr["entry_date"])
        if start is not None and ed < pd.Timestamp(start):
            continue
        if end is not None and ed > pd.Timestamp(end):
            continue
        out.append(tr)
    return out


def _enrich(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    book = book_from_trades(trades)
    years = year_bucket_stats(trades, YEAR_BUCKETS)
    return {
        **book,
        "by_year_bucket": years,
        "geo_pf_year_min10": geom_mean_year_pf(years, min_n=10),
        "drop_top3": drop_top_n_book(trades, 3),
    }


def _stats_dict(st: Any) -> Dict[str, Any]:
    return {
        "name": st.name,
        "start": st.start,
        "end": st.end,
        "total_return": round(st.total_return, 4),
        "cagr": round(st.cagr, 4),
        "ann_vol": round(st.ann_vol, 4),
        "sharpe": round(st.sharpe, 3),
        "sortino": round(st.sortino, 3),
        "max_drawdown": round(st.max_drawdown, 4),
        "calmar": round(st.calmar, 3),
        "pct_time_invested": round(st.pct_time_invested, 4),
        "n_trades": st.n_trades,
        "win_rate": round(st.win_rate, 4),
        "profit_factor": round(st.profit_factor, 3),
        "expectancy": round(st.expectancy, 4),
        "notes": st.notes,
    }


def evaluate_window(
    *,
    label: str,
    kept_trades: List[Dict[str, Any]],
    close_panel: pd.DataFrame,
    spy_close: pd.Series,
    start: str,
    end: Optional[str],
) -> Dict[str, Any]:
    window_trades = _filter_entry(kept_trades, start=start, end=end)
    book = _enrich(window_trades)

    eq, inv = single_slot_equity(
        kept_trades, close_panel, eval_start=start, eval_end=end
    )
    spy_eq, spy_inv = spy_buy_hold_equity(spy_close, eval_start=start, eval_end=end)

    # Align calendars for fair comparison.
    if not eq.empty and not spy_eq.empty:
        common = eq.index.intersection(spy_eq.index)
        eq = eq.reindex(common).ffill().bfill()
        inv = inv.reindex(common).fillna(0.0)
        spy_eq = spy_eq.reindex(common)
        spy_eq = spy_eq / float(spy_eq.iloc[0])
        spy_inv = spy_inv.reindex(common).fillna(1.0)
        eq = eq / float(eq.iloc[0])

    trade_rets = [float(t["ret_pct"]) / 100.0 for t in window_trades]
    spy_st = perf_stats(
        "SPY_buy_hold",
        spy_eq,
        spy_inv,
        notes="100% SPY",
        trade_rets=[],
    )
    sleeve_st = perf_stats(
        "A3_single_slot",
        eq,
        inv,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes="A3 max_open=1 fully invested when in trade",
        trade_rets=trade_rets,
    )
    ok, reason = passes_phase5_gates(sleeve_st, spy_st)
    LOG.info(
        "%s book n=%s PF=%s E=%s | equity Sharpe=%.2f MDD=%.1f%% inv=%.0f%% | "
        "SPY Sharpe=%.2f MDD=%.1f%% | gate=%s (%s)",
        label,
        book.get("n"),
        book.get("profit_factor"),
        book.get("expectancy_pct"),
        sleeve_st.sharpe,
        100.0 * sleeve_st.max_drawdown,
        100.0 * sleeve_st.pct_time_invested,
        spy_st.sharpe,
        100.0 * spy_st.max_drawdown,
        "PASS" if ok else "FAIL",
        reason,
    )
    return {
        "label": label,
        "start": start,
        "end": end,
        "trade_book": book,
        "sleeve": _stats_dict(sleeve_st),
        "spy": _stats_dict(spy_st),
        "gate_pass": bool(ok),
        "gate_reason": reason,
        "excess_sharpe": round(sleeve_st.sharpe - spy_st.sharpe, 3),
        "excess_cagr": round(sleeve_st.cagr - spy_st.cagr, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="A3 single-slot occupancy vs SPY")
    ap.add_argument("--friction", type=float, default=0.10)
    ap.add_argument("--holdout-start", type=str, default="2018-01-01")
    ap.add_argument("--holdout-end", type=str, default="2026-12-31")
    ap.add_argument("--start", type=str, default="1993-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--max-open", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--symbols", type=str, default=",".join(ETF_UNIVERSE))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if "SPY" not in symbols:
        symbols = ["SPY"] + symbols

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else None
    t0 = time.perf_counter()

    frames = load_etf_frames(symbols, start=start, end=end, workers=args.workers)
    LOG.info("Loaded %d frames", sum(1 for d in frames.values() if d is not None and not d.empty))

    raw_trades = collect_a3_trades(frames, friction_pct=float(args.friction))
    LOG.info("A3 unconstrained trades: %d", len(raw_trades))
    kept, occ_meta = occupancy_fcfs(
        raw_trades, max_open=int(args.max_open), unique_symbol=True
    )
    LOG.info("Occupancy max_open=%s kept=%s skipped=%s", args.max_open, occ_meta["n_kept"], occ_meta["n_skipped"])

    panel = closes_panel_from_frames(frames)
    if "SPY" not in panel.columns:
        LOG.error("SPY missing from close panel")
        return 1
    spy_close = panel["SPY"]

    # Pre-declared holdout trade book (unconstrained + occupancy).
    hold_start = args.holdout_start
    hold_end = args.holdout_end
    raw_hold = _filter_entry(raw_trades, start=hold_start, end=hold_end)
    kept_hold = _filter_entry(kept, start=hold_start, end=hold_end)
    LOG.info(
        "Holdout %s..%s unconstrained n=%s PF=%s | occ n=%s PF=%s",
        hold_start,
        hold_end,
        len(raw_hold),
        book_from_trades(raw_hold).get("profit_factor"),
        len(kept_hold),
        book_from_trades(kept_hold).get("profit_factor"),
    )

    windows = [
        ("full_sample", args.start, args.end),
        ("is_pre_holdout", args.start, _day_before(hold_start)),
        ("holdout", hold_start, hold_end),
    ]
    results: List[Dict[str, Any]] = []
    for label, w_start, w_end in windows:
        results.append(
            evaluate_window(
                label=label,
                kept_trades=kept,
                close_panel=panel,
                spy_close=spy_close,
                start=w_start,
                end=w_end,
            )
        )

    hold_row = next(r for r in results if r["label"] == "holdout")
    full_row = next(r for r in results if r["label"] == "full_sample")
    if hold_row["gate_pass"]:
        verdict = "PASS_HOLDOUT_GATES"
    elif (
        (hold_row["sleeve"]["sharpe"] or 0) > 0.7
        and (hold_row["trade_book"].get("profit_factor") or 0) >= 1.5
    ):
        verdict = "SOFT_FAIL_SHARPE_GATES"
    else:
        verdict = "FAIL_HOLDOUT"
    LOG.info("Verdict: %s", verdict)

    elapsed = time.perf_counter() - t0
    payload = {
        "friction_pct": args.friction,
        "max_open": args.max_open,
        "holdout_start": hold_start,
        "holdout_end": hold_end,
        "occupancy": occ_meta,
        "unconstrained_holdout_book": _enrich(raw_hold),
        "occupancy_holdout_book": _enrich(kept_hold),
        "windows": results,
        "verdict": verdict,
        "elapsed_sec": round(elapsed, 2),
        "note": (
            "Single-slot fully invested when in a trade; cash otherwise. "
            "Phase-5 gate: Sharpe>1 and Sharpe>=SPY and MDD better than SPY."
        ),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "roc_rebound_occupancy_summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(kept).to_csv(outdir / "roc_rebound_a3_occ_trades.csv", index=False)
    rows = []
    for r in results:
        rows.append(
            {
                "window": r["label"],
                "gate": r["gate_pass"],
                "reason": r["gate_reason"],
                "n": r["trade_book"].get("n"),
                "PF": r["trade_book"].get("profit_factor"),
                "E": r["trade_book"].get("expectancy_pct"),
                "sharpe": r["sleeve"]["sharpe"],
                "mdd": r["sleeve"]["max_drawdown"],
                "cagr": r["sleeve"]["cagr"],
                "inv_pct": r["sleeve"]["pct_time_invested"],
                "spy_sharpe": r["spy"]["sharpe"],
                "spy_mdd": r["spy"]["max_drawdown"],
                "spy_cagr": r["spy"]["cagr"],
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "roc_rebound_occupancy_vs_spy.csv", index=False)
    LOG.info("Wrote %s", path)
    LOG.info("Full gate=%s Holdout gate=%s", full_row["gate_pass"], hold_row["gate_pass"])
    LOG.info("Wall-clock %.1fs", elapsed)
    return 0


def _day_before(iso_date: str) -> str:
    return str((pd.Timestamp(iso_date) - pd.Timedelta(days=1)).date())


if __name__ == "__main__":
    raise SystemExit(main())
