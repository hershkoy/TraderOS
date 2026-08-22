#!/usr/bin/env python3
"""
Phase 5a: 12-1 cross-sectional momentum with unused deltas vs Phase 1.

Deltas (do not re-run vanilla SPX 12-1):
  - Point-in-time top 500 by 30-day dollar volume
  - Invest only when SPY close > SMA200
  - 10 bps round-trip

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_xs_mom_delta.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.metrics import (
    EVAL_END,
    EVAL_START,
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    perf_stats,
    window_equity,
)
from utils.research.panel import list_daily_symbols, load_spy_close, load_wide_panels
from utils.research.report import save_fragment, write_phase5_scorecard
from utils.research.xs_momentum import simulate_xs_momentum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xs_mom_delta")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _spy_buy_hold(spy_close, start: str, end: str):
    import pandas as pd

    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].dropna()
    eq = s / float(s.iloc[0])
    inv = eq.copy() * 0.0 + 1.0
    return eq, inv


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5a 12-1 PIT+SMA200 vs SPY")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--panel-start", default="2017-11-29")
    ap.add_argument("--spy-start", default="2017-01-03")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--liquid-n", type=int, default=500)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt_phase5",
    )
    args = ap.parse_args()

    timings = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    symbols = list_daily_symbols("ALPACA", "1d")
    timings["list_symbols"] = time.perf_counter() - t0
    logger.info("ALPACA 1d symbols: %d", len(symbols))
    if len(symbols) < args.top_n + 10:
        logger.error("Universe too small (%d)", len(symbols))
        return 1

    t0 = time.perf_counter()
    panels = load_wide_panels(
        symbols,
        start=datetime.strptime(args.panel_start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        provider="ALPACA",
        timeframe="1d",
        workers=args.workers,
    )
    timings["load_panel"] = time.perf_counter() - t0
    close, volume = panels["close"], panels["volume"]
    logger.info("Panel close shape=%s", close.shape)

    t0 = time.perf_counter()
    spy = load_spy_close(
        datetime.strptime(args.spy_start, "%Y-%m-%d"),
        datetime.strptime(args.end, "%Y-%m-%d"),
    )
    timings["load_spy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    eq_g, eq_n, inv, notes = simulate_xs_momentum(
        close,
        volume,
        spy,
        eval_start=args.start,
        eval_end=args.end,
        top_n=args.top_n,
        liquid_n=args.liquid_n,
        cost_bps_rt=args.cost_bps,
        min_price=args.min_price,
        min_adv=args.min_adv,
    )
    timings["sim"] = time.perf_counter() - t0

    spy_eq, spy_inv = _spy_buy_hold(spy, args.start, args.end)
    spy_full = perf_stats("SPY_buy_hold", spy_eq, spy_inv, notes="100% IB/ALPACA SPY")
    xs_full = perf_stats(
        "XS_mom_12_1_PIT500_SMA200_10bps",
        eq_n.reindex(spy_eq.index).ffill().fillna(1.0),
        inv.reindex(spy_eq.index).fillna(0.0),
        spy_cagr=spy_full.cagr,
        spy_sharpe=spy_full.sharpe,
        notes=notes,
    )

    spy_is_eq, spy_is_inv = window_equity(spy_eq, spy_inv, IS_START, IS_END)
    xs_is_eq, xs_is_inv = window_equity(eq_n, inv, IS_START, IS_END)
    spy_is = perf_stats("SPY_buy_hold_IS", spy_is_eq, spy_is_inv, notes="IS window")
    xs_is = perf_stats(
        "XS_mom_IS",
        xs_is_eq,
        xs_is_inv,
        spy_cagr=spy_is.cagr,
        spy_sharpe=spy_is.sharpe,
        notes="IS " + notes,
    )

    spy_oos_eq, spy_oos_inv = window_equity(spy_eq, spy_inv, OOS_START, OOS_END)
    xs_oos_eq, xs_oos_inv = window_equity(eq_n, inv, OOS_START, OOS_END)
    spy_oos = perf_stats("SPY_buy_hold_OOS", spy_oos_eq, spy_oos_inv, notes="OOS window")
    xs_oos = perf_stats(
        "XS_mom_OOS",
        xs_oos_eq,
        xs_oos_inv,
        spy_cagr=spy_oos.cagr,
        spy_sharpe=spy_oos.sharpe,
        notes="OOS " + notes,
    )

    timings["total"] = time.perf_counter() - t_all
    all_stats = [spy_full, xs_full, spy_is, xs_is, spy_oos, xs_oos]
    curves = {
        spy_full.name: spy_eq,
        xs_full.name: eq_n.reindex(spy_eq.index).ffill().fillna(1.0),
    }
    save_fragment(
        args.reports_dir,
        "xs_mom",
        all_stats,
        timings,
        [notes],
        curves,
    )
    window_note = (
        f"ALPACA 1d panel {close.shape[1]} names x {close.shape[0]} dates; "
        f"SPY n={len(spy)} {spy.index[0].date()} -> {spy.index[-1].date()}"
    )
    path = write_phase5_scorecard(
        args.outdir,
        args.reports_dir,
        spy_full,
        all_stats,
        timings,
        window_note,
        curves,
        extra_sections=[
            "## Phase 5a note",
            "This fragment is 12-1 only. Run `run_swing_mr_screen.py` to merge swing MR into the same scorecard.",
        ],
    )
    logger.info("Wrote %s", path)
    logger.info(
        "XS net CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%% vs SPY CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%%",
        xs_full.cagr * 100,
        xs_full.sharpe,
        xs_full.max_drawdown * 100,
        spy_full.cagr * 100,
        spy_full.sharpe,
        spy_full.max_drawdown * 100,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
