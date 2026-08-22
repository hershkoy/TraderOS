#!/usr/bin/env python3
"""
Phase 5b: swing / multi-day mean reversion screen vs SPY.

Four named variants (not a lookback grid):
  dump3_stock, dump3_spy200, rsi2_stock, rsi2_spy200

IS (2018-2022) selects the best Sharpe variant (min trade count); OOS uses that freeze.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\run_swing_mr_screen.py
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
from utils.research.report import save_fragment, write_phase5_scorecard
from utils.research.swing_mr import ALL_VARIANTS, simulate_swing_mr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("swing_mr_screen")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

MIN_IS_TRADES = 20


def _spy_buy_hold(spy_close, start: str, end: str):
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].dropna()
    eq = s / float(s.iloc[0])
    inv = pd.Series(1.0, index=eq.index)
    return eq, inv


def _align_to(eq: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    out = eq.reindex(idx)
    first = out.first_valid_index()
    if first is not None:
        out.loc[:first] = out.loc[first]
    return out.ffill().fillna(1.0)


def _load_xs_fragment(reports_dir: Path) -> List[PerfStats]:
    path = reports_dir / "fragments" / "xs_mom.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for d in raw.get("stats", []):
        d = dict(d)
        d.pop("extra", None)
        try:
            out.append(PerfStats(**d))
        except TypeError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5b swing mean reversion vs SPY")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--panel-start", default="2017-11-29")
    ap.add_argument("--spy-start", default="2017-01-03")
    ap.add_argument("--max-pos", type=int, default=15)
    ap.add_argument("--hold-days", type=int, default=8)
    ap.add_argument("--stop", type=float, default=0.10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--liquid-n", type=int, default=500)
    ap.add_argument("--min-price", type=float, default=10.0)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--dump-thresh", type=float, default=-0.05)
    ap.add_argument("--rsi-thresh", type=float, default=10.0)
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

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    symbols = list_daily_symbols("ALPACA", "1d")
    timings["list_symbols"] = time.perf_counter() - t0
    logger.info("ALPACA 1d symbols: %d", len(symbols))
    if len(symbols) < 50:
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
    open_px, close, low, volume = panels["open"], panels["close"], panels["low"], panels["volume"]
    logger.info("Panel close shape=%s", close.shape)

    t0 = time.perf_counter()
    spy = load_spy_close(
        datetime.strptime(args.spy_start, "%Y-%m-%d"),
        datetime.strptime(args.end, "%Y-%m-%d"),
    )
    timings["load_spy"] = time.perf_counter() - t0

    spy_eq, spy_inv = _spy_buy_hold(spy, args.start, args.end)
    spy_full = perf_stats("SPY_buy_hold", spy_eq, spy_inv, notes="100% IB/ALPACA SPY")
    spy_is_eq, spy_is_inv = window_equity(spy_eq, spy_inv, IS_START, IS_END)
    spy_oos_eq, spy_oos_inv = window_equity(spy_eq, spy_inv, OOS_START, OOS_END)
    spy_is = perf_stats("SPY_buy_hold_IS", spy_is_eq, spy_is_inv, notes="IS window")
    spy_oos = perf_stats("SPY_buy_hold_OOS", spy_oos_eq, spy_oos_inv, notes="OOS window")

    common_kwargs = dict(
        eval_start=args.start,
        eval_end=args.end,
        max_positions=args.max_pos,
        hold_days=args.hold_days,
        stop_loss_pct=args.stop,
        cost_bps_rt=args.cost_bps,
        liquid_n=args.liquid_n,
        min_price=args.min_price,
        min_adv=args.min_adv,
        dump_thresh=args.dump_thresh,
        rsi_thresh=args.rsi_thresh,
    )

    variant_eq: Dict[str, pd.Series] = {}
    variant_inv: Dict[str, pd.Series] = {}
    variant_trades: Dict[str, List[float]] = {}
    variant_notes: Dict[str, str] = {}
    all_stats: List[PerfStats] = [spy_full]
    curves: Dict[str, pd.Series] = {spy_full.name: spy_eq}

    for variant in ALL_VARIANTS:
        t0 = time.perf_counter()
        logger.info("Simulating %s ...", variant)
        eq, inv, trades, exits, notes = simulate_swing_mr(
            open_px, close, low, volume, spy, variant=variant, **common_kwargs
        )
        timings[f"sim_{variant}"] = time.perf_counter() - t0
        eq_a = _align_to(eq, spy_eq.index)
        inv_a = inv.reindex(spy_eq.index).fillna(0.0)
        variant_eq[variant] = eq_a
        variant_inv[variant] = inv_a
        variant_trades[variant] = trades
        variant_notes[variant] = notes
        st = perf_stats(
            f"MR_{variant}",
            eq_a,
            inv_a,
            spy_cagr=spy_full.cagr,
            spy_sharpe=spy_full.sharpe,
            notes=notes,
            trade_rets=trades,
        )
        all_stats.append(st)
        curves[st.name] = eq_a
        logger.info(
            "%s trades=%d CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%% (%s)",
            variant,
            st.n_trades,
            st.cagr * 100,
            st.sharpe,
            st.max_drawdown * 100,
            timings[f"sim_{variant}"],
        )

    # IS freeze: best Sharpe among variants with enough trades
    is_rows = []
    for variant in ALL_VARIANTS:
        eq_is, inv_is = window_equity(variant_eq[variant], variant_inv[variant], IS_START, IS_END)
        # Approximate IS trades by full-sample count is wrong; re-sim IS window for trade stats
        t0 = time.perf_counter()
        eq_i, inv_i, trades_i, _ex, notes_i = simulate_swing_mr(
            open_px,
            close,
            low,
            volume,
            spy,
            variant=variant,
            eval_start=IS_START,
            eval_end=IS_END,
            max_positions=args.max_pos,
            hold_days=args.hold_days,
            stop_loss_pct=args.stop,
            cost_bps_rt=args.cost_bps,
            liquid_n=args.liquid_n,
            min_price=args.min_price,
            min_adv=args.min_adv,
            dump_thresh=args.dump_thresh,
            rsi_thresh=args.rsi_thresh,
        )
        timings[f"sim_{variant}_IS"] = time.perf_counter() - t0
        st_is = perf_stats(
            f"MR_{variant}_IS",
            eq_is if not eq_is.empty else eq_i,
            inv_is if inv_is is not None else inv_i,
            spy_cagr=spy_is.cagr,
            spy_sharpe=spy_is.sharpe,
            notes="IS " + notes_i,
            trade_rets=trades_i,
        )
        is_rows.append((variant, st_is))
        all_stats.append(st_is)

    eligible = [(v, s) for v, s in is_rows if s.n_trades >= MIN_IS_TRADES]
    pool = eligible or is_rows
    winner_variant, winner_is = max(pool, key=lambda x: x[1].sharpe)
    logger.info("IS freeze winner=%s Sharpe=%.2f trades=%d", winner_variant, winner_is.sharpe, winner_is.n_trades)

    eq_oos, inv_oos = window_equity(
        variant_eq[winner_variant], variant_inv[winner_variant], OOS_START, OOS_END
    )
    t0 = time.perf_counter()
    _eq_o, _inv_o, trades_o, _ex_o, notes_o = simulate_swing_mr(
        open_px,
        close,
        low,
        volume,
        spy,
        variant=winner_variant,
        eval_start=OOS_START,
        eval_end=OOS_END,
        max_positions=args.max_pos,
        hold_days=args.hold_days,
        stop_loss_pct=args.stop,
        cost_bps_rt=args.cost_bps,
        liquid_n=args.liquid_n,
        min_price=args.min_price,
        min_adv=args.min_adv,
        dump_thresh=args.dump_thresh,
        rsi_thresh=args.rsi_thresh,
    )
    timings["sim_winner_OOS"] = time.perf_counter() - t0
    st_oos = perf_stats(
        f"MR_{winner_variant}_OOS",
        eq_oos,
        inv_oos,
        spy_cagr=spy_oos.cagr,
        spy_sharpe=spy_oos.sharpe,
        notes="OOS freeze " + notes_o,
        trade_rets=trades_o,
    )
    all_stats.extend([spy_is, spy_oos, st_oos])

    timings["total"] = time.perf_counter() - t_all
    save_fragment(
        args.reports_dir,
        "swing_mr",
        all_stats,
        timings,
        [variant_notes[v] for v in ALL_VARIANTS] + [f"IS freeze winner={winner_variant}"],
        curves,
    )

    xs_stats = _load_xs_fragment(args.reports_dir)
    merged: List[PerfStats] = []
    seen = set()
    # Prefer swing SPY as the benchmark row; keep unique names, XS first then MR
    for s in xs_stats + all_stats:
        if s.name in seen:
            continue
        seen.add(s.name)
        merged.append(s)

    spy_bench = spy_full
    for s in merged:
        if s.name == "SPY_buy_hold":
            spy_bench = s
            break

    xs_curves = {}
    xs_csv = args.reports_dir / "equity_xs_mom.csv"
    if xs_csv.exists():
        xdf = pd.read_csv(xs_csv, index_col=0, parse_dates=True)
        for col in xdf.columns:
            xs_curves[col] = xdf[col]
    merged_curves = {**xs_curves, **curves}

    window_note = (
        f"ALPACA 1d panel {close.shape[1]} names x {close.shape[0]} dates; "
        f"SPY n={len(spy)} {spy.index[0].date()} -> {spy.index[-1].date()}; "
        f"IS freeze winner=`{winner_variant}` (min_IS_trades={MIN_IS_TRADES})"
    )
    extra = [
        "## Phase 5b IS freeze",
        f"- Winner: `{winner_variant}` IS Sharpe={winner_is.sharpe:.2f} trades={winner_is.n_trades} "
        f"OOS Sharpe={st_oos.sharpe:.2f} trades={st_oos.n_trades} MDD={st_oos.max_drawdown:.2%}",
        "- Variants are named (dump vs RSI2, stock SMA20 vs SPY SMA200), not a parameter grid.",
        "- Phase 5c (15m entry timing) only if a daily variant is close to the gate.",
    ]
    path = write_phase5_scorecard(
        args.outdir,
        args.reports_dir,
        spy_bench,
        merged,
        timings,
        window_note,
        merged_curves,
        extra_sections=extra,
    )
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
