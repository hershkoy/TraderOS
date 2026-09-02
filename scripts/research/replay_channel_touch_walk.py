#!/usr/bin/env python3
"""Walk-replay channel-touch on one symbol vs the original full-series backtest.

The detector is fed completed bars in chronological order (prefix 0..t). Fills
that only appear when later bars are visible are the look-ahead gap vs
`trades_for_symbol`.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\replay_channel_touch_walk.py --symbol AAPL
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\replay_channel_touch_walk.py --symbol AAPL --preset 15m
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\replay_channel_touch_walk.py --symbol AAPL --freeze-h2
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_touch_scale import (  # noqa: E402
    BARS_PER_RTH_SESSION,
    DAILY_WINDOW_BARS,
    DAILY_WINDOW_STEP_BARS,
    PRESET_15M,
)
from utils.research.channel_touch_walk_replay import (  # noqa: E402
    apply_quality_filters,
    batch_trades_for_symbol,
    compare_trade_lists,
    comparison_frame,
    summarize_side,
    walk_replay_trades,
)
from utils.scanning.channel_touch import LIVE_DEFAULTS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("replay_channel_touch_walk")


def _scan_kwargs(is_15m: bool, args: argparse.Namespace) -> dict:
    live = LIVE_DEFAULTS
    if is_15m:
        p = PRESET_15M
        kw = {
            "entry_mode": "l3_touch",
            "entry_touch": 3,
            "pivot_len": int(p["pivot_len"]),
            "window_bars": int(p["window_bars"]),
            "window_step_bars": int(p["window_step_bars"]),
            "stop_pct": float(p["stop_pct"]),
            "trail_pct": float(p["trail_pct"]),
            "trail_pct_wide": float(p["trail_pct_wide"]),
            "squeeze_adaptive": True,
            "squeeze_lookback": int(p["squeeze_lookback"]),
            "atr_stop_mult": float(p["atr_stop_mult"]),
            "stop_pct_floor": float(p["stop_pct_floor"]),
            "stop_pct_ceil": float(p["stop_pct_ceil"]),
            "adv_lookback": int(p["adv_lookback"]),
            "include_time": True,
            "entry_slip_pct": 0.001,
            "max_l3_wait_bars": 252 * BARS_PER_RTH_SESSION,
            "min_l3_wait_bars": 12,
            "h2_resist_break": True,
            "h2_resist_break_only": bool(args.h2_resist_break_only),
            "error_pct": float(p["error_pct"]),
            "flat_pct": float(p["flat_pct"]),
            "min_bars_apart": int(p["min_bars_apart"]),
            "min_rally_pct": float(p["min_rally_pct"]),
            "min_pullback_pct": float(p["min_pullback_pct"]),
            "min_total_rise_pct": float(p["min_total_rise_pct"]),
            "max_low_pivots": int(p["max_low_pivots"]),
        }
    else:
        kw = {
            "entry_mode": "l3_touch",
            "entry_touch": 3,
            "pivot_len": int(live["pivot_len"]),
            "window_bars": int(live["window_bars"]),
            "window_step_bars": int(live["window_step_bars"]),
            "stop_pct": float(live["stop_pct"]),
            "trail_pct": 0.10,
            "trail_pct_wide": 0.18,
            "squeeze_adaptive": True,
            "squeeze_lookback": 100,
            "atr_stop_mult": live["atr_stop_mult"],
            "stop_pct_floor": float(live["stop_pct_floor"]),
            "stop_pct_ceil": float(live["stop_pct_ceil"]),
            "adv_lookback": 20,
            "include_time": False,
            "entry_slip_pct": float(live["entry_slip_pct"]),
            "max_l3_wait_bars": int(live["max_l3_wait_bars"]),
            "min_l3_wait_bars": int(live["min_l3_wait_bars"]),
            "h2_resist_break": True,
            "h2_resist_break_only": bool(args.h2_resist_break_only),
            "error_pct": float(live["error_pct"]),
            "flat_pct": float(live["flat_pct"]),
            "min_bars_apart": int(live["min_bars_apart"]),
            "min_rally_pct": float(live["min_rally_pct"]),
            "min_pullback_pct": float(live["min_pullback_pct"]),
            "min_total_rise_pct": float(live["min_total_rise_pct"]),
            "max_low_pivots": int(live["max_low_pivots"]),
        }
        if not args.no_window_scan:
            kw["window_bars"] = int(DAILY_WINDOW_BARS)
            kw["window_step_bars"] = int(DAILY_WINDOW_STEP_BARS)
        else:
            kw["window_bars"] = 0
            kw["window_step_bars"] = 0
    return kw


def _quality_kwargs(is_15m: bool, args: argparse.Namespace) -> dict:
    if is_15m:
        return {
            "max_channel_span_days": 10.0,
            "require_in_channel": False,
            "max_beyond_width": None,
            "max_rsi": None,
        }
    return {
        "max_channel_span_days": float(LIVE_DEFAULTS["max_channel_span_days"]),
        "require_in_channel": False,
        "max_beyond_width": None,
        "max_rsi": None,
    }


def _fmt(s: dict) -> str:
    return "n=%s E=%s PF=%s WR=%s med=%s" % (
        s.get("n_trades"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        s.get("median_gain_pct"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Causal walk-replay vs original full-series channel-touch (one symbol)"
    )
    ap.add_argument("--symbol", default="AAPL", help="Single stock to compare")
    ap.add_argument("--preset", default="", choices=("", "15m"))
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--provider", default="")
    ap.add_argument("--timeframe", default="")
    ap.add_argument(
        "--freeze-h2",
        action="store_true",
        help="Keep first-seen L1-L2-H2 geometry (live watchlist). Default re-scans each prefix.",
    )
    ap.add_argument(
        "--no-h2-resist-break-only",
        dest="h2_resist_break_only",
        action="store_false",
        help="Keep L3 support-tag fills as well as resistance-break fills",
    )
    ap.set_defaults(h2_resist_break_only=True)
    ap.add_argument("--no-window-scan", action="store_true")
    ap.add_argument("--friction-pct", type=float, default=None)
    ap.add_argument("--progress-every", type=int, default=250)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    args = ap.parse_args()

    is_15m = (args.preset or "").strip() == "15m" or (args.timeframe or "").strip() == "15m"
    symbol = str(args.symbol).strip().upper()
    timeframe = (args.timeframe or ("15m" if is_15m else "1d")).strip()
    provider = (args.provider or ("IB" if is_15m else "ALPACA")).strip().upper()
    end_s = (args.end or "").strip() or ("2025-12-02" if is_15m else "2026-08-27")
    friction = float(
        args.friction_pct if args.friction_pct is not None else (0.10 if is_15m else 0.25)
    )
    scan = _scan_kwargs(is_15m, args)
    quality = _quality_kwargs(is_15m, args)

    logger.info(
        "Walk-replay %s %s/%s %s -> %s freeze_h2=%s resist_break_only=%s window=%s/%s",
        symbol,
        provider,
        timeframe,
        args.start,
        end_s,
        bool(args.freeze_h2),
        bool(args.h2_resist_break_only),
        scan.get("window_bars"),
        scan.get("window_step_bars"),
    )

    t_load = time.perf_counter()
    load_kw = dict(
        timeframe=timeframe,
        provider=provider,
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        end=datetime.strptime(end_s, "%Y-%m-%d"),
        use_cache=True,
        workers=1,
    )
    if not is_15m:
        load_kw["fallback_provider"] = "IB"
        load_kw["merge_mode"] = "prefix"
    panels = load_ohlcv_many([symbol], **load_kw)
    df = panels.get(symbol)
    if df is None or df.empty:
        logger.error("No OHLCV for %s", symbol)
        return 1
    logger.info("Loaded %s bars=%d in %.1fs", symbol, len(df), time.perf_counter() - t_load)

    t_batch = time.perf_counter()
    batch_raw = batch_trades_for_symbol(symbol, df, **scan)
    batch = apply_quality_filters(batch_raw, **quality)
    logger.info("Batch trades n=%d in %.1fs", len(batch), time.perf_counter() - t_batch)

    t_walk = time.perf_counter()
    walk_raw = walk_replay_trades(
        symbol,
        df,
        freeze_h2=bool(args.freeze_h2),
        progress_every=int(args.progress_every),
        **scan,
    )
    walk = apply_quality_filters(walk_raw, **quality)
    logger.info("Walk trades n=%d in %.1fs", len(walk), time.perf_counter() - t_walk)

    include_time = bool(scan.get("include_time"))
    cmp = compare_trade_lists(batch, walk, include_time=include_time)
    logger.info(
        "Compare matched=%d only_batch=%d only_walk=%d (buy_date overlap=%d)",
        cmp["n_matched"],
        cmp["n_only_batch"],
        cmp["n_only_walk"],
        cmp["n_matched_buy_date"],
    )
    print("")
    print("==== %s %s walk-replay vs original batch ====" % (symbol, timeframe))
    print("batch  %s" % _fmt(summarize_side(batch, friction)))
    print("walk   %s" % _fmt(summarize_side(walk, friction)))
    print(
        "match  %d / batch %d / walk %d | only_batch=%d only_walk=%d"
        % (
            cmp["n_matched"],
            cmp["n_batch"],
            cmp["n_walk"],
            cmp["n_only_batch"],
            cmp["n_only_walk"],
        )
    )
    frame = comparison_frame(cmp)
    if not frame.empty:
        show = frame.copy()
        if len(show) > 40:
            show = pd.concat(
                [
                    show[show["bucket"] != "matched"].head(20),
                    show[show["bucket"] == "matched"].head(20),
                ],
                ignore_index=True,
            )
        print("")
        print(show.to_string(index=False))

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "walk_replay_%s_%s_%s" % (symbol.lower(), timeframe, stamp)
    batch_path = args.outdir / (stem + "_batch.csv")
    walk_path = args.outdir / (stem + "_walk.csv")
    cmp_path = args.outdir / (stem + "_compare.csv")
    pd.DataFrame(batch).to_csv(batch_path, index=False)
    pd.DataFrame(walk).to_csv(walk_path, index=False)
    frame.to_csv(cmp_path, index=False)
    logger.info("Wrote %s", cmp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
