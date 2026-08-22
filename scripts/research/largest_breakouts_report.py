#!/usr/bin/env python3
"""
Largest breakouts report from TimescaleDB daily bars.

Definition (documented defaults):
  - Breakout: close > prior lookback-day high AND volume >= rvol_min * SMA(volume, lookback)
  - Buy: breakout-day close
  - Sell: day of max close between min_hold and max_hold trading days after entry
  - Keep only trades with gain_pct >= min_gain
  - Non-overlapping per symbol (skip new entries while a prior trade window is open)

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\largest_breakouts_report.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\largest_breakouts_report.py --n-symbols 100 --top 50
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many
from utils.db.timescaledb_client import get_timescaledb_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("largest_breakouts_report")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _pick_symbols(
    n: int,
    provider: str,
    timeframe: str,
    min_bars: int,
) -> List[str]:
    """Pick n liquid symbols with enough daily history (by recent avg dollar volume)."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")

    # Prefer names with enough bars; rank by recent notional volume.
    sql = """
        WITH recent AS (
            SELECT
                symbol,
                COUNT(*) AS n_bars,
                AVG(close * volume) AS avg_dollar_vol
            FROM market_data
            WHERE provider = %s
              AND timeframe = %s
              AND ts >= NOW() - INTERVAL '400 days'
            GROUP BY symbol
            HAVING COUNT(*) >= %s
        )
        SELECT symbol
        FROM recent
        ORDER BY avg_dollar_vol DESC NULLS LAST
        LIMIT %s
    """
    cur = client.connection.cursor()
    try:
        cur.execute(sql, (provider.upper(), timeframe, int(min_bars), int(n)))
        symbols = [r[0] for r in cur.fetchall()]
    finally:
        cur.close()

    if len(symbols) < n:
        logger.warning(
            "Only found %d symbols with >=%d recent bars; requested %d",
            len(symbols),
            min_bars,
            n,
        )
    return symbols


def _find_breakouts(
    symbol: str,
    df: pd.DataFrame,
    lookback: int,
    rvol_min: float,
    min_hold: int,
    max_hold: int,
    min_price: float,
    min_gain: float,
) -> List[dict]:
    if min_hold < 1:
        raise ValueError("min_hold must be >= 1")
    if max_hold < min_hold:
        raise ValueError("max_hold must be >= min_hold")
    if df is None or df.empty or len(df) < lookback + max_hold + 5:
        return []

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out = out.sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            return []

    prior_high = out["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    vol_sma = out["volume"].shift(1).rolling(lookback, min_periods=lookback).mean()
    rvol = out["volume"] / vol_sma.replace(0, np.nan)
    is_breakout = (
        (out["close"] > prior_high)
        & (rvol >= rvol_min)
        & (out["close"] >= min_price)
        & prior_high.notna()
        & vol_sma.notna()
    )

    closes = out["close"].to_numpy(dtype=float)
    dates = out.index
    flags = is_breakout.to_numpy(dtype=bool)
    n = len(out)
    trades: List[dict] = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        # Peak only after min_hold trading days (inclusive through max_hold)
        start = i + min_hold
        end = min(i + max_hold, n - 1)
        if start > end:
            i += 1
            continue
        window = closes[start : end + 1]
        if window.size == 0:
            i += 1
            continue
        rel = int(np.nanargmax(window))
        sell_i = start + rel
        buy_px = float(closes[i])
        sell_px = float(closes[sell_i])
        if not np.isfinite(buy_px) or buy_px <= 0 or not np.isfinite(sell_px):
            i += 1
            continue
        hold = sell_i - i
        if hold < min_hold:
            i += 1
            continue
        gain_pct = (sell_px / buy_px - 1.0) * 100.0
        if gain_pct < min_gain:
            # Failed filter: still advance past entry to avoid re-counting same breakout
            i += 1
            continue
        trades.append(
            {
                "stock": symbol.upper(),
                "buy_date": dates[i].strftime("%Y-%m-%d"),
                "sell_date": dates[sell_i].strftime("%Y-%m-%d"),
                "buy_close": round(buy_px, 4),
                "sell_close": round(sell_px, 4),
                "gain_pct": round(gain_pct, 2),
                "gain_pct_per_day": round(gain_pct / hold, 3),
                "hold_duration_days": int(hold),
                "rvol": round(float(rvol.iloc[i]), 2) if np.isfinite(rvol.iloc[i]) else None,
                "prior_high": round(float(prior_high.iloc[i]), 4),
            }
        )
        # Non-overlapping: jump past the measured hold window
        i = sell_i + 1
    return trades


def main() -> int:
    ap = argparse.ArgumentParser(description="Largest breakouts report from DB")
    ap.add_argument("--n-symbols", type=int, default=100)
    ap.add_argument("--top", type=int, default=100, help="Rows to keep in ranked report")
    ap.add_argument("--lookback", type=int, default=20, help="Prior high lookback (days)")
    ap.add_argument("--rvol-min", type=float, default=1.5)
    ap.add_argument("--min-hold", type=int, default=10, help="Min trading days before sell")
    ap.add_argument("--max-hold", type=int, default=40, help="Max trading days to peak")
    ap.add_argument("--min-gain", type=float, default=10.0, help="Min total gain %% to keep")
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-bars", type=int, default=180, help="Min recent bars to qualify")
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2025-11-26")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "largest_breakouts",
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol list (skips DB liquidity pick)",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _pick_symbols(
            n=args.n_symbols,
            provider=args.provider,
            timeframe=args.timeframe,
            min_bars=args.min_bars,
        )
    if not symbols:
        logger.error("No symbols selected")
        return 1

    logger.info(
        "Loading %d symbols (%s %s) %s -> %s",
        len(symbols),
        args.provider,
        args.timeframe,
        args.start,
        args.end,
    )
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    panels: Dict[str, pd.DataFrame] = load_ohlcv_many(
        symbols,
        timeframe=args.timeframe,
        provider=args.provider,
        start=start_dt,
        end=end_dt,
        use_cache=True,
        chunk_size=50,
        workers=1,
    )
    logger.info("Loaded %d / %d symbols with data", len(panels), len(symbols))

    all_trades: List[dict] = []
    for sym in symbols:
        df = panels.get(sym)
        if df is None:
            continue
        all_trades.extend(
            _find_breakouts(
                symbol=sym,
                df=df,
                lookback=args.lookback,
                rvol_min=args.rvol_min,
                min_hold=args.min_hold,
                max_hold=args.max_hold,
                min_price=args.min_price,
                min_gain=args.min_gain,
            )
        )

    if not all_trades:
        logger.error("No breakouts found")
        return 1

    trades = pd.DataFrame(all_trades)
    trades = trades.sort_values(
        ["gain_pct", "gain_pct_per_day"], ascending=False
    ).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_csv = args.outdir / f"all_breakouts_{stamp}.csv"
    top_csv = args.outdir / f"largest_breakouts_top{args.top}_{stamp}.csv"
    meta_txt = args.outdir / f"largest_breakouts_meta_{stamp}.txt"

    trades.to_csv(all_csv, index=False)
    top = trades.head(int(args.top)).copy()
    report_cols = [
        "stock",
        "buy_date",
        "sell_date",
        "gain_pct",
        "gain_pct_per_day",
        "hold_duration_days",
    ]
    top[report_cols].to_csv(top_csv, index=False)

    elapsed = time.perf_counter() - t0
    meta_lines = [
        "Largest breakouts report",
        f"symbols_requested={args.n_symbols}",
        f"symbols_loaded={len(panels)}",
        f"provider={args.provider}",
        f"timeframe={args.timeframe}",
        f"start={args.start}",
        f"end={args.end}",
        f"lookback={args.lookback}",
        f"rvol_min={args.rvol_min}",
        f"min_hold={args.min_hold}",
        f"max_hold={args.max_hold}",
        f"min_gain={args.min_gain}",
        f"min_price={args.min_price}",
        f"total_breakouts={len(trades)}",
        f"top_rows={len(top)}",
        f"elapsed_sec={elapsed:.1f}",
        "",
        "Breakout: close > prior lookback high AND volume >= rvol_min * SMA(volume, lookback)",
        "Buy: breakout-day close; Sell: day of max close in [min_hold, max_hold] trading days",
        "Filters: hold_duration_days >= min_hold AND gain_pct >= min_gain",
        "hold_duration_days = trading days from buy to sell",
        "gain_pct_per_day = gain_pct / hold_duration_days",
        "",
        "Symbols:",
        ",".join(symbols),
    ]
    meta_txt.write_text("\n".join(meta_lines), encoding="utf-8")

    logger.info("Wrote %d breakouts -> %s", len(trades), all_csv)
    logger.info("Wrote top %d -> %s", len(top), top_csv)
    logger.info("Meta -> %s", meta_txt)
    logger.info("Elapsed %.1fs", elapsed)

    print("\nTop 20 largest breakouts:")
    print(top[report_cols].head(20).to_string(index=False))
    print(f"\nFull top report: {top_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
