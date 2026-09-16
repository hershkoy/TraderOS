#!/usr/bin/env python3
"""Backtest: short H3+ resistance tags on ascending channels after H2.

Opposite of the long L3 / H2-resist-break books. Causal H2 freeze; fade later
from-below resistance tags; cover at support and/or ATR+trail stop above.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_resist_short.py --n-symbols 300 --workers 4 --load-workers 8
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_resist_short.py --preset 15m --n-symbols 300 --workers 4
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    _summarize,
    apply_friction,
    enrich_rs,
    filter_trades,
    keep_one_per_symbol_day,
    select_same_day_rs,
    summarize_by_year,
)
from find_ascending_channels import _pick_symbols, list_symbols_fast  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_touch_resist_short import (  # noqa: E402
    geo_year_pf,
    scan_resist_short,
)
from utils.research.channel_touch_scale import (  # noqa: E402
    BARS_PER_RTH_SESSION,
    DAILY_WINDOW_BARS,
    DAILY_WINDOW_STEP_BARS,
    PRESET_15M,
    apply_daily_long_history_defaults,
    overlay_preset,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("channel_touch_resist_short")


def _fmt(s: dict) -> str:
    return (
        "n=%(n_trades)s E=%(expectancy_pct)s PF=%(profit_factor)s WR=%(win_rate_pct)s%% "
        "med=%(median_gain_pct)s avgW=%(avg_win_pct)s avgL=%(avg_loss_pct)s hold=%(avg_hold_days)s"
        % {k: s.get(k) for k in (
            "n_trades",
            "expectancy_pct",
            "profit_factor",
            "win_rate_pct",
            "median_gain_pct",
            "avg_win_pct",
            "avg_loss_pct",
            "avg_hold_days",
        )}
    )


def _net(df: pd.DataFrame, friction: float) -> pd.DataFrame:
    if df is None or df.empty or not friction:
        return df
    return apply_friction(df, float(friction))


def _print_book(label: str, df: pd.DataFrame, *, gain_col: str) -> Optional[float]:
    print("=== %s ===" % label)
    if df is None or df.empty:
        print("n=0")
        return None
    print(_fmt(_summarize(df, gain_col=gain_col)))
    ydf = summarize_by_year(df, gain_col=gain_col, buckets=YEAR_BUCKETS)
    print(ydf.to_string(index=False))
    geo = geo_year_pf(ydf)
    if geo is not None:
        print("geo_year_PF=%.3f (equal-weight YEAR_BUCKETS, skip n<30)" % geo)
    if "exit_reason" in df.columns:
        print(df["exit_reason"].value_counts().to_string())
    if "touch_num" in df.columns:
        print("touch_num counts:")
        print(df["touch_num"].value_counts().sort_index().to_string())
    return geo


def _daily_base(args: argparse.Namespace) -> dict:
    return {
        "stop_pct": float(args.stop_pct),
        "trail_pct": float(args.trail_pct),
        "atr_stop_mult": float(args.atr_stop_mult) if args.atr_stop_mult is not None else None,
        "stop_pct_floor": float(args.stop_pct_floor),
        "stop_pct_ceil": float(args.stop_pct_ceil),
        "entry_slip_pct": float(args.entry_slip_pct),
        "max_l3_wait_bars": int(args.max_wait_bars),
        "min_h3_wait_bars": int(args.min_h3_wait_bars),
        "entry_touch": int(args.entry_touch),
        "leave_width_frac": float(args.leave_width_frac),
        "touch_error_pct": float(args.touch_error_pct) if args.touch_error_pct is not None else None,
        "support_exit": bool(args.support_exit),
        "max_hold_days": int(args.max_hold_days) if args.max_hold_days else None,
        "window_bars": int(args.window_bars or 0),
        "window_step_bars": int(args.window_step_bars or 0),
        "include_time": False,
        "channel_kwargs": {
            "pivot_len": int(args.pivot_len),
            "error_pct": float(args.error_pct),
            "flat_pct": float(args.flat_pct),
            "min_bars_apart": int(args.min_bars_apart),
            "min_intervening_rally_pct": float(args.min_rally_pct),
            "min_intervening_pullback_pct": float(args.min_pullback_pct),
            "min_total_rise_pct": float(args.min_total_rise_pct),
            "max_low_pivots": int(args.max_low_pivots),
            "causal_h2": True,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Short H3+ channel resistance tags")
    ap.add_argument("--preset", choices=("", "15m"), default="")
    ap.add_argument("--n-symbols", type=int, default=300)
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--start", type=str, default="2018-11-01")
    ap.add_argument("--end", type=str, default="2026-08-27")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--min-bars", type=int, default=400)
    ap.add_argument("--friction-pct", type=float, default=None)
    ap.add_argument("--stop-pct", type=float, default=0.03)
    ap.add_argument("--trail-pct", type=float, default=0.10)
    ap.add_argument("--atr-stop-mult", type=float, default=2.0)
    ap.add_argument("--stop-pct-floor", type=float, default=0.015)
    ap.add_argument("--stop-pct-ceil", type=float, default=0.06)
    ap.add_argument("--entry-slip-pct", type=float, default=0.0)
    ap.add_argument("--max-wait-bars", type=int, default=120)
    ap.add_argument("--min-h3-wait-bars", type=int, default=6)
    ap.add_argument("--entry-touch", type=int, default=3, help="3=H3+, 4=skip H3 take H4+")
    ap.add_argument("--leave-width-frac", type=float, default=0.20)
    ap.add_argument("--touch-error-pct", type=float, default=None)
    ap.add_argument("--support-exit", action="store_true", default=True)
    ap.add_argument("--no-support-exit", action="store_false", dest="support_exit")
    ap.add_argument("--max-hold-days", type=int, default=0)
    ap.add_argument("--pivot-len", type=int, default=15)
    ap.add_argument("--error-pct", type=float, default=1.2)
    ap.add_argument("--flat-pct", type=float, default=0.04)
    ap.add_argument("--min-bars-apart", type=int, default=15)
    ap.add_argument("--min-rally-pct", type=float, default=4.0)
    ap.add_argument("--min-pullback-pct", type=float, default=3.0)
    ap.add_argument("--min-total-rise-pct", type=float, default=3.0)
    ap.add_argument("--max-low-pivots", type=int, default=16)
    ap.add_argument("--window-bars", type=int, default=0)
    ap.add_argument("--window-step-bars", type=int, default=0)
    ap.add_argument("--no-window-scan", action="store_true")
    ap.add_argument("--max-channel-span-days", type=float, default=365.0)
    ap.add_argument("--require-in-channel", action="store_true", default=False)
    ap.add_argument("--max-beyond-width", type=float, default=None)
    ap.add_argument("--max-entries-per-day", type=int, default=0, help="0=unique symbol/day")
    args = ap.parse_args()

    if (args.preset or "").strip() == "15m":
        overlay_preset(args, PRESET_15M, sys.argv[1:])
        args.min_h3_wait_bars = int(getattr(args, "min_h3_wait_bars", 12) or 12)
        if "--min-h3-wait-bars" not in sys.argv[1:]:
            args.min_h3_wait_bars = 12
        if args.friction_pct is None:
            args.friction_pct = 0.10
        if args.end == "2026-08-27":
            args.end = str(PRESET_15M.get("end", "2025-12-02"))
    else:
        apply_daily_long_history_defaults(args)
        if args.friction_pct is None:
            args.friction_pct = 0.25

    friction = float(args.friction_pct or 0.0)
    gain_col = "gain_pct_net" if friction else "gain_pct"
    is_15m = (args.preset or "").strip() == "15m"
    rs_bars = BARS_PER_RTH_SESSION if is_15m else 1
    span_cap = float(args.max_channel_span_days) if args.max_channel_span_days else (10.0 if is_15m else 365.0)

    t0 = time.perf_counter()
    names_only = [s.strip().upper() for s in str(args.symbols or "").split(",") if s.strip()]
    if names_only:
        symbols = sorted(set(names_only) | {"SPY"})
    elif args.all_symbols:
        provider = "IB" if is_15m else "ALPACA"
        tf = "15m" if is_15m else "1d"
        symbols = sorted({s.upper() for s in list_symbols_fast(provider, tf)} | {"SPY"})
    else:
        provider = "IB" if is_15m else "ALPACA"
        tf = "15m" if is_15m else "1d"
        symbols = _pick_symbols(int(args.n_symbols), provider, tf, int(args.min_bars))
        symbols = sorted(set(s.upper() for s in symbols) | {"SPY"})
    logger.info("Symbols=%d (ex SPY)", len([s for s in symbols if s != "SPY"]))

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if is_15m:
        panels = load_ohlcv_many(
            symbols,
            timeframe="15m",
            provider="IB",
            start=start,
            end=end,
            workers=int(args.load_workers),
            chunk_size=int(args.chunk_size),
            use_cache=True,
        )
    else:
        panels = load_ohlcv_many(
            symbols,
            timeframe="1d",
            provider="ALPACA",
            start=start,
            end=end,
            fallback_provider="IB",
            merge_mode="prefix",
            workers=int(args.load_workers),
            use_cache=True,
            chunk_size=int(args.chunk_size),
        )
    spy_df = panels.get("SPY")
    if spy_df is None or spy_df.empty:
        logger.error("No SPY panel")
        return 1
    logger.info("OHLCV loaded in %.1fs", time.perf_counter() - t0)

    base = _daily_base(args)
    if is_15m:
        base["include_time"] = True
        base["window_bars"] = int(getattr(args, "window_bars", 0) or PRESET_15M["window_bars"])
        base["window_step_bars"] = int(
            getattr(args, "window_step_bars", 0) or PRESET_15M["window_step_bars"]
        )
        ck = base["channel_kwargs"]
        ck["pivot_len"] = int(PRESET_15M["pivot_len"]) if "--pivot-len" not in sys.argv[1:] else int(args.pivot_len)
        ck["error_pct"] = float(PRESET_15M["error_pct"]) if "--error-pct" not in sys.argv[1:] else float(args.error_pct)
        ck["min_bars_apart"] = int(PRESET_15M["min_bars_apart"])
        ck["min_intervening_rally_pct"] = float(PRESET_15M["min_rally_pct"])
        ck["min_intervening_pullback_pct"] = float(PRESET_15M["min_pullback_pct"])
        ck["min_total_rise_pct"] = float(PRESET_15M["min_total_rise_pct"])
        ck["flat_pct"] = float(PRESET_15M["flat_pct"])
        ck["max_low_pivots"] = int(PRESET_15M["max_low_pivots"])
    elif int(args.window_bars or 0) <= 0 and not args.no_window_scan:
        base["window_bars"] = DAILY_WINDOW_BARS
        base["window_step_bars"] = DAILY_WINDOW_STEP_BARS

    t_scan = time.perf_counter()
    raw = scan_resist_short(panels, symbols=symbols, workers=int(args.workers), base=base)
    wall_scan = time.perf_counter() - t_scan
    logger.info("Scan n=%d in %.1fs", len(raw), wall_scan)
    if raw.empty:
        logger.error("No trades")
        return 1

    raw = enrich_rs(raw, panels, spy_df, lookbacks=(63, 126), bars_per_session=rs_bars)
    outdir = dated_outdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = outdir / ("channel_touch_resist_short_raw_%s.csv" % stamp)
    raw.to_csv(raw_path, index=False)
    logger.info("Wrote %s", raw_path)

    filt = filter_trades(
        raw,
        max_channel_span_days=span_cap,
        require_in_channel=bool(args.require_in_channel),
        max_beyond_width=args.max_beyond_width,
    )
    filt_n = _net(filt, friction)
    uniq = keep_one_per_symbol_day(filt_n)
    print("wall_clock_load_scan=%.1fs scan=%.1fs friction=%.2f%%" % (
        time.perf_counter() - t0, wall_scan, friction
    ))
    print("side=short entry_touch>=%d min_wait=%d support_exit=%s atr_k=%s" % (
        int(args.entry_touch),
        int(args.min_h3_wait_bars),
        bool(args.support_exit),
        args.atr_stop_mult,
    ))
    _print_book("raw (no filters)", _net(raw, friction), gain_col=gain_col)
    _print_book("span<=%s" % int(span_cap), filt_n, gain_col=gain_col)
    _print_book("span unique-symbol/day", uniq, gain_col=gain_col)

    h3 = uniq.loc[uniq["touch_num"] == 3] if "touch_num" in uniq.columns else uniq.iloc[0:0]
    h4p = uniq.loc[uniq["touch_num"] >= 4] if "touch_num" in uniq.columns else uniq.iloc[0:0]
    _print_book("unique H3 only", h3, gain_col=gain_col)
    _print_book("unique H4+", h4p, gain_col=gain_col)

    if int(args.max_entries_per_day or 0) > 0:
        rs1 = select_same_day_rs(filt_n, rs_col="rs_spy_126d", max_per_day=int(args.max_entries_per_day))
        _print_book("span RS top%d/day" % int(args.max_entries_per_day), rs1, gain_col=gain_col)

    # A/B: no support cover (trail/ATR only) on same fills — re-sim would need rescan;
    # report occupancy already used support_exit from args. Second scan if support was on.
    if bool(args.support_exit):
        base2 = dict(base)
        base2["support_exit"] = False
        t2 = time.perf_counter()
        raw2 = scan_resist_short(panels, symbols=symbols, workers=int(args.workers), base=base2)
        logger.info("No-support-exit rescan n=%d in %.1fs", len(raw2), time.perf_counter() - t2)
        if not raw2.empty:
            raw2 = enrich_rs(raw2, panels, spy_df, lookbacks=(63, 126), bars_per_session=rs_bars)
            f2 = filter_trades(raw2, max_channel_span_days=span_cap)
            u2 = keep_one_per_symbol_day(_net(f2, friction))
            _print_book("unique ATR+trail only (no support cover)", u2, gain_col=gain_col)

    uniq_path = outdir / ("channel_touch_resist_short_unique_%s.csv" % stamp)
    uniq.to_csv(uniq_path, index=False)
    print("Wrote %s" % uniq_path)
    print("Wrote %s" % raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
