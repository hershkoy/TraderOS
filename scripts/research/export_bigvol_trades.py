#!/usr/bin/env python3
"""
Export Phase-6b BigVol sleeve trades and print a TradingView plot plan.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\export_bigvol_trades.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\export_bigvol_trades.py --symbol AMD
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from spy_benchmark_screen import DEFAULT_BIGVOL_SETUPS, strategy_bigvol_portfolio
from utils.research.metrics import EVAL_END, EVAL_START

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("export_bigvol_trades")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

# Frozen Phase 6b sleeve
ALLOC = 0.10
MAX_POS = 15
STOP = 0.15
COST_BPS = 10.0


def _unix_day(date_s: str) -> int:
    """UTC midnight unix seconds for a calendar date (TV daily bars)."""
    return int(pd.Timestamp(date_s, tz="UTC").timestamp())


def main() -> int:
    ap = argparse.ArgumentParser(description="Export BigVol sleeve trades for TV markers")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--setups", type=Path, default=DEFAULT_BIGVOL_SETUPS)
    ap.add_argument("--symbol", default="", help="Focus symbol (default: busiest)")
    ap.add_argument("--max-markers", type=int, default=8, help="Max trades to mark on TV")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase6b" / "trades",
    )
    args = ap.parse_args()

    if not args.setups.exists():
        raise FileNotFoundError(args.setups)

    eq, inv, notes, trades = strategy_bigvol_portfolio(
        setups_csv=args.setups,
        start=args.start,
        end=args.end,
        alloc_frac=ALLOC,
        max_positions=MAX_POS,
        stop_loss_pct=STOP,
        cost_bps_rt=COST_BPS,
        return_trades=True,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "bigvol_sleeve_trades.csv"
    trades.to_csv(out_csv, index=False)
    logger.info("Wrote %d trades -> %s", len(trades), out_csv)
    logger.info("Sleeve notes: %s", notes)
    logger.info("Equity end=%.3f avg_invested=%.2f", float(eq.iloc[-1]), float(inv.mean()))

    if trades.empty:
        logger.error("No trades to plot")
        return 1

    counts = trades["symbol"].value_counts()
    focus = args.symbol.upper() if args.symbol else str(counts.index[0])
    focus_trades = trades[trades["symbol"] == focus].copy()
    if focus_trades.empty:
        logger.error("No trades for %s; top symbols: %s", focus, counts.head(10).to_dict())
        return 1

    # Prefer a handful of recent completed trades for readable TV markers
    focus_trades = focus_trades.sort_values("entry_date").tail(int(args.max_markers))
    markers = []
    for _, row in focus_trades.iterrows():
        entry_ts = _unix_day(row["entry_date"])
        exit_ts = _unix_day(row["exit_date"])
        markers.append(
            {
                "side": "BUY",
                "date": row["entry_date"],
                "time": entry_ts,
                "price": float(row["entry_px"]),
                "label": f"BUY {row['entry_date']}",
            }
        )
        markers.append(
            {
                "side": "SELL",
                "date": row["exit_date"],
                "time": exit_ts,
                "price": float(row["exit_px"]),
                "label": f"SELL {row['exit_date']} ({row['reason']})",
                "reason": row["reason"],
                "pnl_pct": float(row["pnl_pct"]),
            }
        )

    plan = {
        "symbol": focus,
        "tv_symbol": f"NASDAQ:{focus}" if focus.isalpha() else focus,
        "timeframe": "D",
        "n_trades_symbol": int((trades["symbol"] == focus).sum()),
        "n_trades_total": int(len(trades)),
        "top_symbols": counts.head(15).to_dict(),
        "trades": focus_trades.to_dict(orient="records"),
        "markers": markers,
        "scroll_to": str(focus_trades["entry_date"].iloc[0]),
        "csv": str(out_csv),
    }
    plan_path = args.outdir / f"tv_plot_plan_{focus}.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    logger.info("Focus symbol=%s trades=%d plan=%s", focus, len(focus_trades), plan_path)
    print(json.dumps({"focus": focus, "plan": str(plan_path), "top": counts.head(8).to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
