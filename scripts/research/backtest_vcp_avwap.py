#!/usr/bin/env python3
"""Backtest causal VCP AVWAP breakout with realistic 15m fills.

Daily VCP / 52w / AVWAP / ATR from completed bars through the prior session.
Buy first RTH 15m close above pivot and AVWAP with RVOL (signal-close).
Sell: ATR*1 hard stop or weekly close below 10-week SMA, fill next 15m mid.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\backtest_vcp_avwap.py --n-symbols 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    apply_friction,
    summarize_by_year,
)
from find_ascending_channels import _pick_15m_by_daily_adv  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_volume_geometry import book_stats, geom_mean_year_pf  # noqa: E402
from utils.research.panel import list_daily_symbols, to_naive_index  # noqa: E402
from utils.research.realistic_purchaser import as_et  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.vcp_avwap import (  # noqa: E402
    VcpParams,
    arm_sessions,
    avwap_series,
    find_vcp_breakout_fill,
    index_rth_15m_fast,
    occupancy_take,
    rvol_window_bars,
    simulate_vcp_exit,
    weekly_close_below_ma,
    weekly_sma_by_date,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("vcp_avwap")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

FRICTION_PCT = 0.25


def _naive_daily(df: pd.DataFrame) -> pd.DataFrame:
    return to_naive_index(df)


def _stats_block(label: str, trades: pd.DataFrame, gain_col: str = "gain_pct") -> None:
    g = pd.to_numeric(trades[gain_col], errors="coerce") if not trades.empty else pd.Series(dtype=float)
    s = book_stats(g)
    years = summarize_by_year(trades, gain_col=gain_col, buckets=YEAR_BUCKETS)
    geo = geom_mean_year_pf(years)
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s WR=%s median=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s.get("win_rate_pct"),
        s.get("median_pct"),
    )
    if not years.empty:
        print(years.to_string(index=False))


def _utc_clock(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d %H:%M")


def _utc_date(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d")


def _occupancy(trades: List[dict], max_open: int) -> List[dict]:
    cap = 10**9 if int(max_open) <= 0 else int(max_open)
    return occupancy_take(trades, max_open=cap)


def _bar_vol(bar: dict) -> float | None:
    try:
        val = float(bar.get("volume"))
    except (TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def _prior_volumes_cached(
    cache: Dict[str, tuple],
    sym: str,
    by_day: dict,
    watch: date,
) -> List[float]:
    last_d, hist = cache.get(sym, (None, []))
    if last_d is not None and last_d >= watch:
        hist = []
        last_d = None
    days = [d for d in by_day.keys() if d < watch and (last_d is None or d > last_d)]
    days.sort()
    extra: List[float] = []
    for day in days:
        for bar in by_day.get(day) or []:
            vol = _bar_vol(bar)
            if vol is not None:
                extra.append(vol)
    hist = list(hist) + extra
    new_last = days[-1] if days else last_d
    cache[sym] = (new_last if new_last is not None else watch, hist)
    return hist


def main() -> int:
    ap = argparse.ArgumentParser(description="Causal VCP AVWAP + realistic 15m fills")
    ap.add_argument("--n-symbols", type=int, default=300)
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--max-open", type=int, default=0, help="0 = unique-symbol uncapped concurrent")
    ap.add_argument("--friction-pct", type=float, default=FRICTION_PCT)
    ap.add_argument("--fill-mode", default="signal-close")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    params = VcpParams(fill_mode=str(args.fill_mode))
    outdir = args.outdir or dated_outdir(ROOT / "reports" / "vcp_avwap")
    outdir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    t0 = time.perf_counter()
    if args.all_symbols:
        symbols = list_daily_symbols("IB", "15m")
    else:
        symbols = _pick_15m_by_daily_adv(int(args.n_symbols))
    timings["list_symbols"] = time.perf_counter() - t0
    LOG.info("Universe %d symbols (%.1fs)", len(symbols), timings["list_symbols"])
    if len(symbols) < 20:
        LOG.error("Universe too small (%d)", len(symbols))
        return 1

    t0 = time.perf_counter()
    daily_frames = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        use_cache=True,
        workers=max(1, int(args.load_workers)),
        chunk_size=int(args.chunk_size),
        fallback_provider="IB",
        merge_mode="prefix",
    )
    timings["load_daily"] = time.perf_counter() - t0
    LOG.info("Daily panels %d/%d (%.1fs)", len(daily_frames), len(symbols), timings["load_daily"])

    t0 = time.perf_counter()
    arms_by_sym: Dict[str, List[dict]] = {}
    daily_by_sym: Dict[str, pd.DataFrame] = {}
    n_arm = 0
    n_daily = len(daily_frames)
    for i, (raw_sym, raw_df) in enumerate(daily_frames.items(), 1):
        if raw_df is None or raw_df.empty:
            continue
        sym = str(raw_sym).upper()
        df = _naive_daily(raw_df)
        daily_by_sym[sym] = df
        rows = arm_sessions(df, params)
        if rows:
            arms_by_sym[sym] = rows
            n_arm += len(rows)
        if i % 50 == 0:
            LOG.info("VCP scan %d/%d arms=%d", i, n_daily, n_arm)
    timings["signals"] = time.perf_counter() - t0
    LOG.info(
        "Armed watch rows %d on %d names (%.1fs)",
        n_arm,
        len(arms_by_sym),
        timings["signals"],
    )
    if n_arm == 0:
        LOG.error("No VCP arms")
        return 1

    arm_rows = []
    for sym, rows in arms_by_sym.items():
        for rec in rows:
            item = dict(rec)
            item["stock"] = sym
            arm_rows.append(item)
    arm_df = pd.DataFrame(arm_rows)
    arm_path = outdir / "vcp_avwap_signals.csv"
    arm_df.to_csv(arm_path, index=False)
    LOG.info("Wrote %s", arm_path)

    need = sorted(arms_by_sym.keys())
    t0 = time.perf_counter()
    panels_15m = load_ohlcv_many(
        need,
        timeframe="15m",
        provider="IB",
        start=start,
        end=end,
        use_cache=True,
        workers=max(1, int(args.load_workers)),
        chunk_size=int(args.chunk_size),
    )
    timings["load_15m"] = time.perf_counter() - t0
    LOG.info("15m panels %d/%d (%.1fs)", len(panels_15m), len(need), timings["load_15m"])

    LOG.info("Indexing RTH 15m sessions")
    indexed: Dict[str, dict] = {}
    for i_sym, (sym, df) in enumerate(panels_15m.items(), 1):
        if df is None or df.empty:
            continue
        indexed[str(sym).upper()] = index_rth_15m_fast(df, naive_tz="UTC")
        if i_sym % 50 == 0:
            LOG.info("Indexed 15m %d/%d", i_sym, len(panels_15m))
    LOG.info("Indexed 15m %d names", len(indexed))

    weekly_below_by_sym: Dict[str, dict] = {}
    vol_cache: Dict[str, tuple] = {}
    rvol_bars = rvol_window_bars(params)
    t0 = time.perf_counter()
    raw: List[dict] = []
    n_skip_15m = 0
    n_skip_fill = 0
    n_skip_exit = 0
    n_skip_open = 0
    open_until: Dict[str, date] = {}
    n_arms_total = len(arm_rows)
    LOG.info("Starting fills on %d arm rows", n_arms_total)

    for i_arm, rec in enumerate(sorted(arm_rows, key=lambda r: (r["watch_date"], r["stock"])), 1):
        if i_arm % 10000 == 0:
            LOG.info("Fill walk %d/%d kept=%d skip_open=%d", i_arm, n_arms_total, len(raw), n_skip_open)
        sym = str(rec["stock"]).upper()
        watch = rec["watch_date"]
        held_until = open_until.get(sym)
        if held_until is not None and watch <= held_until:
            n_skip_open += 1
            continue
        by_day = indexed.get(sym)
        if not by_day:
            n_skip_15m += 1
            continue
        sess_bars = list(by_day.get(watch) or [])
        if not sess_bars:
            n_skip_fill += 1
            continue
        prior_vols = _prior_volumes_cached(vol_cache, sym, by_day, watch)
        bought = find_vcp_breakout_fill(
            sess_bars,
            pivot=float(rec["pivot_price"]),
            avwap=float(rec["avwap"]),
            rvol_min=float(params.breakout_rvol_min),
            prior_volumes=prior_vols,
            rvol_window_bars=rvol_bars,
            fill_mode=params.fill_mode,
        )
        if not bought.filled or bought.fill_px is None or bought.exec_bar_ts is None:
            n_skip_fill += 1
            continue
        daily = daily_by_sym[sym]
        below = weekly_below_by_sym.get(sym)
        if below is None:
            sma = weekly_sma_by_date(daily["close"], period=int(params.weekly_ma_period))
            below = weekly_close_below_ma(daily["close"], sma)
            weekly_below_by_sym[sym] = below
        avwap_asof = avwap_series(daily, int(rec["base_start_i"]))
        atr = rec.get("atr")
        try:
            atr_f = float(atr) if atr is not None else float("nan")
        except (TypeError, ValueError):
            atr_f = float("nan")
        if atr_f != atr_f or atr_f <= 0:
            atr_f = None
        exited = simulate_vcp_exit(
            by_day,
            entry_ts=bought.exec_bar_ts,
            entry_px=float(bought.fill_px),
            atr_at_entry=atr_f,
            atr_mult=float(params.atr_mult_stop),
            weekly_below=below,
            avwap_by_prior_session=avwap_asof,
        )
        if not exited.filled or exited.sell_px is None or exited.exec_bar_ts is None:
            n_skip_exit += 1
            continue
        buy_px = float(bought.fill_px)
        sell_px = float(exited.sell_px)
        gain = (sell_px / buy_px - 1.0) * 100.0
        raw.append(
            {
                "stock": sym,
                "signal_date": str(rec["signal_date"]),
                "watch_date": str(watch),
                "buy_date": _utc_date(bought.exec_bar_ts),
                "buy_ts": bought.exec_bar_ts,
                "buy_time": _utc_clock(bought.exec_bar_ts),
                "buy_price": round(buy_px, 4),
                "sell_date": _utc_date(exited.exec_bar_ts),
                "sell_ts": exited.exec_bar_ts,
                "sell_time": _utc_clock(exited.exec_bar_ts),
                "sell_price": round(sell_px, 4),
                "gain_pct": round(gain, 4),
                "exit_reason": exited.exit_reason,
                "pivot_price": round(float(rec["pivot_price"]), 4),
                "avwap": round(float(rec["avwap"]), 4),
                "depth_pct": round(float(rec["depth_pct"]), 4),
                "atr": None if atr_f is None else round(float(atr_f), 4),
                "near_52w": round(float(rec["near_52w"]), 4),
            }
        )
        try:
            open_until[sym] = as_et(exited.exec_bar_ts).date()
        except (TypeError, ValueError):
            open_until[sym] = watch
    timings["fills"] = time.perf_counter() - t0
    LOG.info(
        "Filled %d skip_15m=%d skip_buy=%d skip_exit=%d skip_open=%d (%.1fs)",
        len(raw),
        n_skip_15m,
        n_skip_fill,
        n_skip_exit,
        n_skip_open,
        timings["fills"],
    )

    occupied = _occupancy(raw, int(args.max_open))
    trades = pd.DataFrame(occupied)
    if not trades.empty:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"]).dt.strftime("%Y-%m-%d")
        trades["sell_date"] = pd.to_datetime(trades["sell_date"]).dt.strftime("%Y-%m-%d")
    csv_path = outdir / "vcp_avwap_trades.csv"
    trades.to_csv(csv_path, index=False)
    LOG.info("Occupancy kept %d / %d -> %s", len(trades), len(raw), csv_path)

    _stats_block("gross", trades)
    if not trades.empty:
        net = apply_friction(trades, float(args.friction_pct))
        _stats_block("fric%.2f" % float(args.friction_pct), net, gain_col="gain_pct_net")
        if "exit_reason" in trades.columns:
            print(trades["exit_reason"].value_counts().to_string())

    timings["wall"] = time.perf_counter() - t_all
    LOG.info("Wall %.1fs %s", timings["wall"], timings)
    (outdir / "vcp_avwap_timings.txt").write_text(
        "\n".join("%s=%.3f" % (k, v) for k, v in timings.items()) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
