#!/usr/bin/env python3
"""
Backtest: long ascending-channel bottom touches (>= Nth touch).

Rules:
  - Detect classical ascending channels (same as find_ascending_channels)
  - Enter long on each bottom touch number >= entry_touch (default 3)
  - Entry at close of pivot-confirmation bar (touch_index + pivot_len)
  - Exit: 3% hard stop OR 10% trailing stop from peak (whichever is higher)
  - One open position per symbol (skip new entries while in a trade)

Edge filters (optional):
  - 20d average dollar volume (ADV) floor
  - ATR%% of price floor (volatility)
  - Same-day relative-strength ranking vs SPY (keep top-N entries per buy_date)
  - Round-trip friction scenarios (slippage + commission) on expectancy

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py --symbols GLD
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py --all-symbols --workers 4 --load-workers 8 --edge-improve
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from find_ascending_channels import _pick_symbols, find_channels, list_symbols_fast
from utils.data.ohlcv_loader import load_ohlcv_many

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_channel_touch_trades")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr.astype(float)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    tr = _true_range(high, low, close)
    atr = np.full(len(close), np.nan, dtype=float)
    if len(close) < length:
        return atr
    atr[length - 1] = float(np.nanmean(tr[:length]))
    alpha = 1.0 / float(length)
    for i in range(length, len(close)):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def _adv_20(close: np.ndarray, volume: np.ndarray, i: int, lookback: int = 20) -> float:
    if i < 0 or lookback <= 0:
        return float("nan")
    start = max(0, i - lookback + 1)
    c = close[start : i + 1]
    v = volume[start : i + 1]
    dv = c * v
    if len(dv) == 0 or not np.isfinite(dv).any():
        return float("nan")
    return float(np.nanmean(dv))


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
            exit_i = i
            exit_px = float(stop_level)
            if abs(stop_level - hard_stop) < 1e-9:
                exit_reason = "hard_stop"
            elif stop_level > hard_stop + 1e-9:
                exit_reason = "trail_stop"
            else:
                exit_reason = "hard_stop"
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
    adv_lookback: int = 20,
    atr_len: int = 14,
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
    volume = (
        out["volume"].to_numpy(dtype=float)
        if "volume" in out.columns
        else np.full(len(out), np.nan, dtype=float)
    )
    atr = _atr(high, low, close, length=atr_len)
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
            entry_px = float(close[entry_i])
            atr_i = float(atr[entry_i]) if entry_i < len(atr) else float("nan")
            atr_pct = (atr_i / entry_px * 100.0) if entry_px > 0 and np.isfinite(atr_i) else float("nan")
            adv = _adv_20(close, volume, entry_i, lookback=adv_lookback)
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
                    "adv_20": round(adv, 2) if np.isfinite(adv) else None,
                    "atr_pct": round(atr_pct, 3) if np.isfinite(atr_pct) else None,
                }
            )
            busy_until = sim["exit_i"]
    return trades


def _worker_symbol_trades(payload: dict) -> List[dict]:
    """ProcessPool worker: run trades for one symbol from a pickled OHLCV frame."""
    return trades_for_symbol(
        payload["symbol"],
        payload["df"],
        entry_touch=payload["entry_touch"],
        stop_pct=payload["stop_pct"],
        trail_pct=payload["trail_pct"],
        pivot_len=payload["pivot_len"],
    )


def _summarize(trades: pd.DataFrame, gain_col: str = "gain_pct") -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "n_symbols": 0,
            "win_rate_pct": None,
            "avg_gain_pct": None,
            "median_gain_pct": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "avg_hold_days": None,
        }
    g = trades[gain_col].astype(float)
    wins = trades[g > 0]
    losses = trades[g <= 0]
    avg_win = float(wins[gain_col].mean()) if len(wins) else 0.0
    avg_loss = float(losses[gain_col].mean()) if len(losses) else 0.0
    wr = len(wins) / len(trades) if len(trades) else 0.0
    expectancy = wr * avg_win + (1.0 - wr) * avg_loss
    gp = float(wins[gain_col].sum()) if len(wins) else 0.0
    gl = float((-losses[gain_col]).sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 1e-12 else float("inf") if gp > 0 else 0.0
    out = {
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["stock"].nunique()) if "stock" in trades.columns else 0,
        "win_rate_pct": round(wr * 100.0, 2),
        "avg_gain_pct": round(float(g.mean()), 2),
        "median_gain_pct": round(float(g.median()), 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expectancy_pct": round(expectancy, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "avg_hold_days": round(float(trades["hold_days"].mean()), 1) if "hold_days" in trades.columns else None,
    }
    if "exit_reason" in trades.columns:
        out["hard_stop_exits"] = int((trades["exit_reason"] == "hard_stop").sum())
        out["trail_stop_exits"] = int((trades["exit_reason"] == "trail_stop").sum())
        out["eod_exits"] = int((trades["exit_reason"] == "eod").sum())
    return out


def _return_lookback(close: pd.Series, asof: pd.Timestamp, lookback: int) -> float:
    """Simple return from lookback bars before asof to asof (inclusive end)."""
    if close is None or close.empty or lookback <= 0:
        return float("nan")
    s = close.sort_index()
    if s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_convert(None)
    # last available bar on/before asof
    hist = s.loc[:asof]
    if len(hist) < lookback + 1:
        return float("nan")
    end_px = float(hist.iloc[-1])
    start_px = float(hist.iloc[-(lookback + 1)])
    if not np.isfinite(end_px) or not np.isfinite(start_px) or start_px <= 0:
        return float("nan")
    return end_px / start_px - 1.0


def enrich_rs(
    trades: pd.DataFrame,
    panels: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = (63, 126),
) -> pd.DataFrame:
    """Attach relative strength vs SPY at each buy_date."""
    if trades.empty:
        return trades
    out = trades.copy()
    spy = spy_df["close"].astype(float).copy()
    if not isinstance(spy.index, pd.DatetimeIndex):
        spy.index = pd.DatetimeIndex(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)

    close_cache: Dict[str, pd.Series] = {}
    for lb in lookbacks:
        col = f"rs_spy_{lb}d"
        vals: List[float] = []
        for _, row in out.iterrows():
            sym = str(row["stock"]).upper()
            asof = pd.Timestamp(row["buy_date"])
            if sym not in close_cache:
                df = panels.get(sym)
                if df is None or df.empty:
                    close_cache[sym] = pd.Series(dtype=float)
                else:
                    c = df["close"].astype(float).copy()
                    if not isinstance(c.index, pd.DatetimeIndex):
                        c.index = pd.DatetimeIndex(c.index)
                    if c.index.tz is not None:
                        c.index = c.index.tz_convert(None)
                    close_cache[sym] = c
            stock_ret = _return_lookback(close_cache[sym], asof, lb)
            spy_ret = _return_lookback(spy, asof, lb)
            if np.isfinite(stock_ret) and np.isfinite(spy_ret):
                vals.append(round((stock_ret - spy_ret) * 100.0, 3))
            else:
                vals.append(float("nan"))
        out[col] = vals
    return out


def filter_trades(
    trades: pd.DataFrame,
    *,
    min_adv: Optional[float] = None,
    min_atr_pct: Optional[float] = None,
) -> pd.DataFrame:
    if trades.empty:
        return trades
    m = pd.Series(True, index=trades.index)
    if min_adv is not None:
        m &= trades["adv_20"].fillna(0) >= float(min_adv)
    if min_atr_pct is not None:
        m &= trades["atr_pct"].fillna(0) >= float(min_atr_pct)
    return trades.loc[m].copy()


def select_same_day_rs(
    trades: pd.DataFrame,
    *,
    rs_col: str = "rs_spy_126d",
    max_per_day: int = 1,
) -> pd.DataFrame:
    """Keep top-N trades per buy_date ranked by relative strength vs SPY."""
    if trades.empty or max_per_day <= 0:
        return trades
    if rs_col not in trades.columns:
        return trades
    ranked = trades.sort_values(
        ["buy_date", rs_col],
        ascending=[True, False],
        na_position="last",
    )
    kept = ranked.groupby("buy_date", sort=False).head(int(max_per_day))
    return kept.reset_index(drop=True)


def apply_friction(trades: pd.DataFrame, round_trip_pct: float) -> pd.DataFrame:
    """Deduct round-trip friction (slippage + commission) from gross gain_pct."""
    out = trades.copy()
    out["gain_pct_net"] = out["gain_pct"].astype(float) - float(round_trip_pct)
    return out


def edge_scenarios(
    trades: pd.DataFrame,
    *,
    friction_pcts: Sequence[float] = (0.10, 0.25),
) -> pd.DataFrame:
    """Compare liquidity/vol/RS filters and net expectancy after friction."""
    scenarios: List[Tuple[str, pd.DataFrame]] = [
        ("baseline", trades),
        ("adv>=5M", filter_trades(trades, min_adv=5_000_000)),
        ("adv>=20M", filter_trades(trades, min_adv=20_000_000)),
        ("atr>=1.5%", filter_trades(trades, min_atr_pct=1.5)),
        ("atr>=2.0%", filter_trades(trades, min_atr_pct=2.0)),
        ("adv20M+atr1.5", filter_trades(trades, min_adv=20_000_000, min_atr_pct=1.5)),
        (
            "adv20M+atr1.5+rs_top1",
            select_same_day_rs(
                filter_trades(trades, min_adv=20_000_000, min_atr_pct=1.5),
                rs_col="rs_spy_126d",
                max_per_day=1,
            ),
        ),
        (
            "adv20M+atr1.5+rs_top3",
            select_same_day_rs(
                filter_trades(trades, min_adv=20_000_000, min_atr_pct=1.5),
                rs_col="rs_spy_126d",
                max_per_day=3,
            ),
        ),
        (
            "rs_top1_only",
            select_same_day_rs(trades, rs_col="rs_spy_126d", max_per_day=1),
        ),
    ]

    rows: List[dict] = []
    for name, subset in scenarios:
        gross = _summarize(subset, gain_col="gain_pct")
        row = {
            "scenario": name,
            "n_trades": gross["n_trades"],
            "n_symbols": gross["n_symbols"],
            "win_rate_pct": gross["win_rate_pct"],
            "avg_win_pct": gross["avg_win_pct"],
            "avg_loss_pct": gross["avg_loss_pct"],
            "expectancy_gross_pct": gross["expectancy_pct"],
            "profit_factor": gross["profit_factor"],
            "avg_hold_days": gross["avg_hold_days"],
        }
        for fr in friction_pcts:
            net_df = apply_friction(subset, fr)
            net = _summarize(net_df, gain_col="gain_pct_net")
            key = f"expectancy_net_{fr:.2f}pct"
            row[key] = net["expectancy_pct"]
            row[f"net_above_1_{fr:.2f}"] = (
                None if net["expectancy_pct"] is None else bool(net["expectancy_pct"] >= 1.0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


REPORT_COLS = [
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
    "adv_20",
    "atr_pct",
    "rs_spy_63d",
    "rs_spy_126d",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Channel bottom-touch long backtest")
    ap.add_argument("--n-symbols", type=int, default=100)
    ap.add_argument(
        "--all-symbols",
        action="store_true",
        help="Scan full ALPACA 1d universe (fast DISTINCT list; skips liquidity HAVING)",
    )
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
    ap.add_argument("--workers", type=int, default=1, help="Process workers for scan/backtest")
    ap.add_argument("--load-workers", type=int, default=4, help="Thread workers for OHLCV load")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--min-adv", type=float, default=None, help="Filter: min 20d ADV dollars")
    ap.add_argument("--min-atr-pct", type=float, default=None, help="Filter: min ATR%% of price")
    ap.add_argument(
        "--max-entries-per-day",
        type=int,
        default=0,
        help="If >0, keep top-N same-day entries by RS vs SPY (126d)",
    )
    ap.add_argument(
        "--friction-pct",
        type=float,
        default=0.0,
        help="Round-trip friction %% deducted from each trade gain (e.g. 0.25)",
    )
    ap.add_argument(
        "--edge-improve",
        action="store_true",
        help="Write liquidity/vol/RS/friction scenario comparison table",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    t_sym = time.perf_counter()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.all_symbols:
        symbols = list_symbols_fast(args.provider, args.timeframe)
        logger.info("Full universe list: %d symbols (%.1fs)", len(symbols), time.perf_counter() - t_sym)
    else:
        symbols = _pick_symbols(args.n_symbols, args.provider, args.timeframe, args.min_bars)
        if "GLD" not in symbols:
            symbols = ["GLD"] + symbols

    # Always include SPY for relative-strength ranking
    if "SPY" not in symbols:
        symbols = list(symbols) + ["SPY"]

    logger.info(
        "Backtest %d symbols | entry touch>=%d | stop=%.1f%% trail=%.1f%% | workers=%d load_workers=%d",
        len(symbols),
        args.entry_touch,
        args.stop_pct * 100,
        args.trail_pct * 100,
        args.workers,
        args.load_workers,
    )
    t_load = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe=args.timeframe,
        provider=args.provider,
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        use_cache=True,
        chunk_size=args.chunk_size,
        workers=max(1, int(args.load_workers)),
    )
    logger.info(
        "Loaded %d/%d panels in %.1fs",
        len(panels),
        len(symbols),
        time.perf_counter() - t_load,
    )

    spy_df = panels.get("SPY")
    if spy_df is None or spy_df.empty:
        logger.error("SPY panel missing; cannot compute relative strength")
        return 1

    t_scan = time.perf_counter()
    all_trades: List[dict] = []
    payloads = [
        {
            "symbol": sym,
            "df": panels[sym],
            "entry_touch": args.entry_touch,
            "stop_pct": args.stop_pct,
            "trail_pct": args.trail_pct,
            "pivot_len": args.pivot_len,
        }
        for sym in symbols
        if sym in panels and sym != "SPY"
    ]
    if args.workers and args.workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futs = [pool.submit(_worker_symbol_trades, p) for p in payloads]
            for fut in as_completed(futs):
                all_trades.extend(fut.result())
    else:
        for p in payloads:
            all_trades.extend(_worker_symbol_trades(p))
    logger.info("Scan+trade done in %.1fs (%d trades)", time.perf_counter() - t_scan, len(all_trades))

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_csv = args.outdir / f"channel_touch_trades_{stamp}.csv"
    summary_txt = args.outdir / f"channel_touch_trades_summary_{stamp}.txt"
    scenarios_csv = args.outdir / f"channel_touch_edge_scenarios_{stamp}.csv"

    if not all_trades:
        logger.warning("No trades generated")
        pd.DataFrame().to_csv(trades_csv, index=False)
        summary_txt.write_text("No trades\n", encoding="utf-8")
        print("No trades")
        return 0

    trades = pd.DataFrame(all_trades)
    t_rs = time.perf_counter()
    trades = enrich_rs(trades, panels, spy_df, lookbacks=(63, 126))
    logger.info("RS enrichment done in %.1fs", time.perf_counter() - t_rs)

    # Optional live filters for the primary report
    filtered = filter_trades(trades, min_adv=args.min_adv, min_atr_pct=args.min_atr_pct)
    if args.max_entries_per_day and args.max_entries_per_day > 0:
        filtered = select_same_day_rs(
            filtered, rs_col="rs_spy_126d", max_per_day=int(args.max_entries_per_day)
        )
    if args.friction_pct and args.friction_pct > 0:
        filtered = apply_friction(filtered, args.friction_pct)
        gain_col = "gain_pct_net"
    else:
        gain_col = "gain_pct"

    filtered = filtered.sort_values(
        [gain_col if gain_col in filtered.columns else "gain_pct", "buy_date"],
        ascending=[False, True],
    ).reset_index(drop=True)

    export_cols = [c for c in REPORT_COLS if c in filtered.columns]
    if "gain_pct_net" in filtered.columns:
        export_cols = export_cols + ["gain_pct_net"]
    filtered[export_cols].to_csv(trades_csv, index=False)

    summary = _summarize(filtered, gain_col=gain_col if gain_col in filtered.columns else "gain_pct")
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
        f"min_adv={args.min_adv}",
        f"min_atr_pct={args.min_atr_pct}",
        f"max_entries_per_day={args.max_entries_per_day}",
        f"friction_pct={args.friction_pct}",
        f"elapsed_sec={elapsed:.1f}",
        "",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "Exit: hard stop = entry*(1-stop); trail = peak*(1-trail); fill at max(hard,trail) when low hits",
        "Features: adv_20 = 20d mean(close*volume); atr_pct = ATR14/close*100; rs_spy_Nd = stock_ret - SPY_ret",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    scenario_df = None
    if args.edge_improve:
        scenario_df = edge_scenarios(trades, friction_pcts=(0.10, 0.25))
        scenario_df.to_csv(scenarios_csv, index=False)
        logger.info("Edge scenarios -> %s", scenarios_csv)

    logger.info("Wrote %d trades -> %s", len(filtered), trades_csv)
    logger.info("Summary -> %s", summary_txt)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if scenario_df is not None and not scenario_df.empty:
        print("\nEdge improvement scenarios:")
        print(scenario_df.to_string(index=False))
        print(f"\nScenarios CSV: {scenarios_csv}")
    print("\nTop 20 trades by gain %:")
    show_cols = [c for c in export_cols if c in filtered.columns]
    print(filtered[show_cols].head(20).to_string(index=False))
    gld = filtered[filtered["stock"] == "GLD"]
    if not gld.empty:
        print("\nGLD trades:")
        print(gld[show_cols].to_string(index=False))
    print(f"\nFull trades: {trades_csv}")
    print(f"Summary: {summary_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
