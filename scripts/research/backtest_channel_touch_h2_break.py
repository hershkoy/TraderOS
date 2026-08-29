#!/usr/bin/env python3
"""A/B: after H2, fill a close above resistance instead of cancelling.

MGNI 2020-10: H2 in Aug, first close>resist ~Oct 13 (currently aborts the L3 wait).
Nightly stays off until this beats keeper E and PF.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_h2_break.py
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

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    _scan_trades,
    _summarize,
    apply_friction,
    enrich_rs,
    filter_trades,
    select_same_day_rs,
    summarize_by_year,
)
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_touch_scale import (  # noqa: E402
    DAILY_WINDOW_BARS,
    DAILY_WINDOW_STEP_BARS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("channel_touch_h2_break")

RAW_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_trades_raw_20260828_194314.csv"
)
KEEPER_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_trades_20260828_194314.csv"
)
L3_FILTERS = dict(
    require_in_channel=True,
    max_channel_span_days=365.0,
    max_beyond_width=0.25,
    max_rsi=50.0,
)


def _fmt(s: dict) -> str:
    return "n=%s E=%s PF=%s WR=%s med=%s hard_stop=%s" % (
        s.get("n_trades"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        s.get("median_gain_pct"),
        s.get("hard_stop_exits"),
    )


def _net(df: pd.DataFrame, friction: float) -> pd.DataFrame:
    if df.empty:
        return df
    if "gain_pct_net" in df.columns:
        return df
    return apply_friction(df, friction) if friction else df


def main() -> int:
    ap = argparse.ArgumentParser(description="H2 resistance-break A/B")
    ap.add_argument("--raw", type=Path, default=RAW_TRADES)
    ap.add_argument("--keeper", type=Path, default=KEEPER_TRADES)
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-27")
    ap.add_argument("--friction-pct", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--load-workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.perf_counter()
    friction = float(args.friction_pct)
    gain_col = "gain_pct_net" if friction else "gain_pct"

    raw = pd.read_csv(args.raw)
    keeper = pd.read_csv(args.keeper)
    symbols = sorted(set(raw["stock"].astype(str).str.upper()) | {"SPY"})
    logger.info("Raw n=%d unique=%d keeper n=%d", len(raw), len(symbols) - 1, len(keeper))

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        fallback_provider="IB",
        merge_mode="prefix",
        workers=int(args.load_workers),
    )
    spy_df = panels.get("SPY")
    if spy_df is None or spy_df.empty:
        logger.error("No SPY panel")
        return 1
    logger.info("OHLCV loaded in %.1fs", time.perf_counter() - t0)

    base = {
        "entry_touch": 3,
        "stop_pct": 0.03,
        "trail_pct": 0.10,
        "trail_pct_wide": 0.18,
        "squeeze_adaptive": True,
        "squeeze_pctile": 75.0,
        "squeeze_lookback": 100,
        "pivot_len": 15,
        "entry_mode": "l3_touch",
        "atr_stop_mult": 2.0,
        "stop_pct_floor": 0.015,
        "stop_pct_ceil": 0.06,
        "window_bars": DAILY_WINDOW_BARS,
        "window_step_bars": DAILY_WINDOW_STEP_BARS,
        "entry_features": True,
        "entry_slip_pct": 0.001,
        "max_l3_wait_bars": 252,
        "min_l3_wait_bars": 6,
        "h2_resist_break": True,
        "channel_kwargs": {
            "error_pct": 1.2,
            "flat_pct": 0.04,
            "min_bars_apart": 15,
            "min_intervening_rally_pct": 4.0,
            "min_intervening_pullback_pct": 3.0,
            "min_total_rise_pct": 3.0,
            "max_low_pivots": 16,
        },
    }
    t_scan = time.perf_counter()
    scanned = _scan_trades(panels, symbols=symbols, workers=int(args.workers), base=base)
    logger.info("Scan n=%d in %.1fs", len(scanned), time.perf_counter() - t_scan)
    if scanned.empty:
        logger.error("No trades")
        return 1
    scanned = enrich_rs(scanned, panels, spy_df, lookbacks=(63, 126))

    is_brk = scanned["resist_break"].fillna(False).astype(bool)
    l3 = scanned.loc[~is_brk].copy()
    brk = scanned.loc[is_brk].copy()
    l3_q = filter_trades(l3, **L3_FILTERS)
    brk_span = filter_trades(brk, max_channel_span_days=365.0) if not brk.empty else brk
    l3_q = _net(l3_q, friction)
    brk_n = _net(brk, friction)
    brk_span_n = _net(brk_span, friction)
    keeper_n = _net(keeper, friction)

    print("=== keeper CSV (RS top1 L3) ===")
    print(_fmt(_summarize(keeper_n, gain_col=gain_col)))
    print("=== rescanned L3 quality (no RS) ===")
    print(_fmt(_summarize(l3_q, gain_col=gain_col)))
    print("=== resist-break only (no span cap) ===")
    print(_fmt(_summarize(brk_n, gain_col=gain_col)))
    if not brk_n.empty:
        print(summarize_by_year(brk_n, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    print("=== resist-break + span<=365 ===")
    print(_fmt(_summarize(brk_span_n, gain_col=gain_col)))
    if not brk_span_n.empty:
        print(
            summarize_by_year(brk_span_n, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(
                index=False
            )
        )

    l3_rs = select_same_day_rs(l3_q, rs_col="rs_spy_126d", max_per_day=1)
    combo = pd.concat([l3_q, brk_span_n], ignore_index=True, sort=False)
    combo_rs = select_same_day_rs(combo, rs_col="rs_spy_126d", max_per_day=1)
    print("=== L3 quality + RS top1 (rescanned) ===")
    print(_fmt(_summarize(l3_rs, gain_col=gain_col)))
    print(summarize_by_year(l3_rs, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    print("=== L3 quality + resist-break span365 + RS top1 ===")
    print(_fmt(_summarize(combo_rs, gain_col=gain_col)))
    print(summarize_by_year(combo_rs, gain_col=gain_col, buckets=YEAR_BUCKETS).to_string(index=False))
    n_brk_kept = (
        int(combo_rs["resist_break"].fillna(False).astype(bool).sum()) if not combo_rs.empty else 0
    )
    print("resist-break kept after RS=%d / span365=%d / raw_break=%d" % (n_brk_kept, len(brk_span_n), len(brk_n)))

    mgni = scanned[scanned["stock"] == "MGNI"].copy()
    if not mgni.empty:
        cols = [
            c
            for c in (
                "buy_date",
                "sell_date",
                "buy_price",
                "gain_pct",
                "exit_reason",
                "resist_break",
                "channel_start",
                "channel_end",
                "channel_span_days",
                "channel_pos",
                "rsi_14",
            )
            if c in mgni.columns
        ]
        print("=== MGNI fills (rescan) ===")
        print(mgni[cols].sort_values("buy_date").to_string(index=False))

    outdir = ROOT / "reports" / "ascending_channels"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not brk.empty:
        path = outdir / ("channel_touch_h2_resist_break_%s.csv" % stamp)
        brk.to_csv(path, index=False)
        logger.info("Wrote %s", path)
    print("elapsed_sec=%.1f" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
