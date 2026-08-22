#!/usr/bin/env python3
"""
Backtest: long ascending-channel bottom touches (>= Nth touch).

Rules:
  - Detect classical ascending channels (same as find_ascending_channels)
  - Enter long on each bottom touch number >= entry_touch (default 3)
  - Entry at close of pivot-confirmation bar (touch_index + pivot_len)
  - Exit: 3% hard stop OR 10% trailing stop from peak (whichever is higher)
  - One open position per symbol (skip new entries while in a trade)

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py --n-symbols 100 --symbols GLD
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from find_ascending_channels import _pick_symbols, find_channels
from utils.data.ohlcv_loader import load_ohlcv_many

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_channel_touch_trades")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _simulate_trade(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    entry_i: int,
    *,
    stop_pct: float,
    trail_pct: float,
) -> Optional[dict]:
    """Long from entry_i close; exit on hard stop or trailing stop (intrabar low)."""
    n = len(close)
    if entry_i < 0 or entry_i >= n - 1:
        return None
    entry_px = float(close[entry_i])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None

    hard_stop = entry_px * (1.0 - stop_pct)
    peak = entry_px
    exit_i = n - 1
    exit_px = float(close[exit_i])
    exit_reason = "eod"

    for i in range(entry_i + 1, n):
        hi = float(high[i])
        lo = float(low[i])
        if np.isfinite(hi):
            peak = max(peak, hi)
        trail_stop = peak * (1.0 - trail_pct)
        stop_level = max(hard_stop, trail_stop)
        if np.isfinite(lo) and lo <= stop_level:
            # Gap through stop: fill at open if open < stop, else stop
            # We don't have separate open check beyond low; use stop_level
            exit_i = i
            exit_px = float(stop_level)
            exit_reason = "hard_stop" if stop_level <= hard_stop + 1e-12 else "trail_stop"
            # If trail and hard equal at start, prefer hard_stop label when near hard
            if abs(stop_level - hard_stop) < 1e-9:
                exit_reason = "hard_stop"
            elif stop_level > hard_stop + 1e-9:
                exit_reason = "trail_stop"
            break
        exit_i = i
        exit_px = float(close[i])
        exit_reason = "eod"

    hold = int(exit_i - entry_i)
    gain_pct = (exit_px / entry_px - 1.0) * 100.0
    return {
        "buy_date": dates[entry_i].strftime("%Y-%m-%d"),
        "sell_date": dates[exit_i].strftime("%Y-%m-%d"),
        "buy_price": round(entry_px, 4),
        "sell_price": round(exit_px, 4),
        "gain_pct": round(gain_pct, 2),
        "hold_days": hold,
        "exit_reason": exit_reason,
        "peak_price": round(float(peak), 4),
        "entry_i": int(entry_i),
        "exit_i": int(exit_i),
    }


def trades_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    entry_touch: int = 3,
    stop_pct: float = 0.03,
    trail_pct: float = 0.10,
    pivot_len: int = 15,
    **channel_kwargs,
) -> List[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out = out.sort_index()

    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    dates = out.index
    n = len(out)

    channels = find_channels(out, pivot_len=pivot_len, **channel_kwargs)
    trades: List[dict] = []
    busy_until = -1

    for ch in channels:
        touch_idxs: List[int] = list(ch.get("touch_indices") or [])
        if len(touch_idxs) < entry_touch:
            continue
        for touch_num, t_idx in enumerate(touch_idxs, start=1):
            if touch_num < entry_touch:
                continue
            # Pivot confirmation delay
            entry_i = int(t_idx) + int(ch.get("pivot_len", pivot_len))
            if entry_i <= busy_until or entry_i >= n:
                continue
            sim = _simulate_trade(
                high,
                low,
                close,
                dates,
                entry_i,
                stop_pct=stop_pct,
                trail_pct=trail_pct,
            )
            if sim is None:
                continue
            trades.append(
                {
                    "stock": symbol.upper(),
                    "channel_start": ch["start_date"],
                    "channel_end": ch["end_date"],
                    "touch_num": touch_num,
                    "touch_date": dates[t_idx].strftime("%Y-%m-%d"),
                    "touch_price": round(float(low[t_idx]), 4),
                    **{k: v for k, v in sim.items() if k not in ("entry_i", "exit_i")},
                    "entry_i": sim["entry_i"],
                    "exit_i": sim["exit_i"],
                }
            )
            busy_until = sim["exit_i"]
    return trades


def _summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    wins = trades[trades["gain_pct"] > 0]
    losses = trades[trades["gain_pct"] <= 0]
    avg_win = float(wins["gain_pct"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["gain_pct"].mean()) if len(losses) else 0.0
    # Expectancy per trade in %
    wr = len(wins) / len(trades) if len(trades) else 0.0
    expectancy = wr * avg_win + (1.0 - wr) * avg_loss
    # Profit factor on % gains (approx)
    gp = float(wins["gain_pct"].sum()) if len(wins) else 0.0
    gl = float((-losses["gain_pct"]).sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 1e-12 else float("inf") if gp > 0 else 0.0
    return {
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["stock"].nunique()),
        "win_rate_pct": round(wr * 100.0, 2),
        "avg_gain_pct": round(float(trades["gain_pct"].mean()), 2),
        "median_gain_pct": round(float(trades["gain_pct"].median()), 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expectancy_pct": round(expectancy, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "avg_hold_days": round(float(trades["hold_days"].mean()), 1),
        "hard_stop_exits": int((trades["exit_reason"] == "hard_stop").sum()),
        "trail_stop_exits": int((trades["exit_reason"] == "trail_stop").sum()),
        "eod_exits": int((trades["exit_reason"] == "eod").sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Channel bottom-touch long backtest")
    ap.add_argument("--n-symbols", type=int, default=100)
    ap.add_argument("--symbols", default="", help="Comma list override (e.g. GLD,SPY)")
    ap.add_argument("--entry-touch", type=int, default=3, help="First touch number to buy")
    ap.add_argument("--stop-pct", type=float, default=0.03)
    ap.add_argument("--trail-pct", type=float, default=0.10)
    ap.add_argument("--pivot-len", type=int, default=15)
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-11-26")
    ap.add_argument("--min-bars", type=int, default=180)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _pick_symbols(args.n_symbols, args.provider, args.timeframe, args.min_bars)
        # Ensure GLD is included when scanning liquid universe
        if "GLD" not in symbols:
            symbols = ["GLD"] + symbols

    logger.info(
        "Backtest %d symbols | entry touch>=%d | stop=%.1f%% trail=%.1f%%",
        len(symbols),
        args.entry_touch,
        args.stop_pct * 100,
        args.trail_pct * 100,
    )
    panels = load_ohlcv_many(
        symbols,
        timeframe=args.timeframe,
        provider=args.provider,
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        use_cache=True,
        chunk_size=50,
    )

    all_trades: List[dict] = []
    for sym in symbols:
        df = panels.get(sym)
        if df is None:
            continue
        all_trades.extend(
            trades_for_symbol(
                sym,
                df,
                entry_touch=args.entry_touch,
                stop_pct=args.stop_pct,
                trail_pct=args.trail_pct,
                pivot_len=args.pivot_len,
            )
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_csv = args.outdir / f"channel_touch_trades_{stamp}.csv"
    summary_txt = args.outdir / f"channel_touch_trades_summary_{stamp}.txt"

    if not all_trades:
        logger.warning("No trades generated")
        pd.DataFrame().to_csv(trades_csv, index=False)
        summary_txt.write_text("No trades\n", encoding="utf-8")
        print("No trades")
        return 0

    trades = pd.DataFrame(all_trades).sort_values(
        ["gain_pct", "buy_date"], ascending=[False, True]
    ).reset_index(drop=True)
    # Drop helper index cols from report CSV
    report_cols = [
        "stock",
        "channel_start",
        "channel_end",
        "touch_num",
        "touch_date",
        "touch_price",
        "buy_date",
        "sell_date",
        "buy_price",
        "sell_price",
        "gain_pct",
        "hold_days",
        "exit_reason",
        "peak_price",
    ]
    trades[report_cols].to_csv(trades_csv, index=False)

    summary = _summarize(trades)
    elapsed = time.perf_counter() - t0
    lines = [
        "Ascending channel bottom-touch long backtest",
        f"entry_touch>={args.entry_touch}",
        f"stop_pct={args.stop_pct}",
        f"trail_pct={args.trail_pct}",
        f"pivot_len={args.pivot_len} (entry at touch+pivot_len close)",
        f"provider={args.provider} timeframe={args.timeframe}",
        f"start={args.start} end={args.end}",
        f"symbols={len(panels)}",
        f"elapsed_sec={elapsed:.1f}",
        "",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "Exit: hard stop = entry*(1-stop); trail = peak*(1-trail); fill at max(hard,trail) when low hits",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Wrote %d trades -> %s", len(trades), trades_csv)
    logger.info("Summary -> %s", summary_txt)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nTop 20 trades by gain %:")
    print(trades[report_cols].head(20).to_string(index=False))
    gld = trades[trades["stock"] == "GLD"]
    if not gld.empty:
        print("\nGLD trades:")
        print(gld[report_cols].to_string(index=False))
    print(f"\nFull trades: {trades_csv}")
    print(f"Summary: {summary_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
