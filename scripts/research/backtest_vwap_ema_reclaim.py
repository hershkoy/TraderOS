#!/usr/bin/env python3
"""Backtest SMB LOD reclaim (9 EMA x session VWAP) with realistic 15m fills.

Buy next-mid (default) or signal-close after a completed 15m 9 EMA cross up
through session VWAP, LOD already in. Sell measured-move target / 1/3 stop
at the next same-session 15m mid (EOD close if no next bar). Friction 0.10%.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python -u scripts\\research\\backtest_vwap_ema_reclaim.py --n-symbols 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
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
from utils.research.panel import list_daily_symbols  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.vwap_ema_reclaim import (  # noqa: E402
    ReclaimParams,
    collect_symbol_trades,
    drop_top_n_expectancy,
    index_rth_15m_fast,
    max_concurrent,
    occupancy_take,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("vwap_ema_reclaim")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

FRICTION_PCT = 0.10


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


def _stats_block(label: str, trades: pd.DataFrame, gain_col: str = "gain_pct") -> None:
    g = pd.to_numeric(trades[gain_col], errors="coerce") if not trades.empty else pd.Series(dtype=float)
    s = book_stats(g)
    years = summarize_by_year(trades, gain_col=gain_col, buckets=YEAR_BUCKETS)
    geo = geom_mean_year_pf(years)
    drop3 = drop_top_n_expectancy(g.tolist(), 3)
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s WR=%s median=%s drop-top-3 E=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s.get("win_rate_pct"),
        s.get("median_pct"),
        None if drop3 is None else round(drop3, 4),
    )
    if not years.empty:
        print(years.to_string(index=False))
    if not trades.empty and "buy_date" in trades.columns:
        bd = pd.to_datetime(trades["buy_date"], errors="coerce")
        y22 = trades.loc[bd.dt.year == 2022]
        if not y22.empty:
            s22 = book_stats(pd.to_numeric(y22[gain_col], errors="coerce"))
            LOG.info(
                "%s calendar 2022 n=%s E=%s PF=%s WR=%s",
                label,
                s22.get("n"),
                s22.get("expectancy_pct"),
                s22.get("profit_factor"),
                s22.get("win_rate_pct"),
            )


def _rows_from_raw(raw: List[dict], stock: str) -> List[dict]:
    out: List[dict] = []
    for rec in raw:
        buy_ts = rec["buy_ts"]
        sell_ts = rec["sell_ts"]
        out.append(
            {
                "stock": stock,
                "buy_date": _utc_date(buy_ts),
                "buy_ts": buy_ts,
                "buy_time": _utc_clock(buy_ts),
                "buy_price": round(float(rec["buy_price"]), 4),
                "sell_date": _utc_date(sell_ts),
                "sell_ts": sell_ts,
                "sell_time": _utc_clock(sell_ts),
                "sell_price": round(float(rec["sell_price"]), 4),
                "gain_pct": round(float(rec["gain_pct"]), 4),
                "exit_reason": rec.get("exit_reason"),
                "lod": round(float(rec["lod"]), 4),
                "cross_px": round(float(rec["cross_px"]), 4),
                "unit": round(float(rec["unit"]), 4),
                "stop_px": round(float(rec["stop_px"]), 4),
                "target_px": round(float(rec["target_px"]), 4),
                "signal_ts": rec.get("signal_ts"),
                "fill_mode": rec.get("fill_mode"),
            }
        )
    return out


def _run_book(
    label: str,
    indexed: Dict[str, dict],
    params: ReclaimParams,
    *,
    max_open: int,
    friction_pct: float,
    outdir: Path,
) -> pd.DataFrame:
    raw: List[dict] = []
    n_sym = len(indexed)
    for i, (sym, by_day) in enumerate(indexed.items(), 1):
        rows = collect_symbol_trades(by_day, params=params, naive_tz="UTC")
        raw.extend(_rows_from_raw(rows, sym))
        if i % 50 == 0:
            LOG.info("%s scan %d/%d trades=%d", label, i, n_sym, len(raw))
    cap = 10**9 if int(max_open) <= 0 else int(max_open)
    occupied = occupancy_take(raw, max_open=cap) if raw else []
    trades = pd.DataFrame(occupied)
    if not trades.empty:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"]).dt.strftime("%Y-%m-%d")
        trades["sell_date"] = pd.to_datetime(trades["sell_date"]).dt.strftime("%Y-%m-%d")
    stem = label.replace(" ", "_")
    csv_path = outdir / ("vwap_ema_reclaim_%s_trades.csv" % stem)
    trades.to_csv(csv_path, index=False)
    peak, med = max_concurrent(occupied)
    LOG.info(
        "%s occupancy kept %d / %d max_conc=%d median_conc=%.1f -> %s",
        label,
        len(trades),
        len(raw),
        peak,
        med,
        csv_path,
    )
    _stats_block("%s gross" % label, trades)
    if not trades.empty:
        net = apply_friction(trades, float(friction_pct))
        _stats_block("%s fric%.2f" % (label, float(friction_pct)), net, gain_col="gain_pct_net")
        if "exit_reason" in trades.columns:
            print(trades["exit_reason"].value_counts().to_string())
    return trades


def main() -> int:
    ap = argparse.ArgumentParser(description="SMB LOD reclaim + realistic 15m fills")
    ap.add_argument("--n-symbols", type=int, default=300)
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--max-open", type=int, default=0, help="0 = unique-symbol uncapped")
    ap.add_argument("--friction-pct", type=float, default=FRICTION_PCT)
    ap.add_argument(
        "--fill-mode",
        default="both",
        help="next-mid (default book), signal-close, or both",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir or dated_outdir(ROOT / "reports" / "vwap_ema_reclaim")
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
    panels_15m = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=start,
        end=end,
        use_cache=True,
        workers=max(1, int(args.load_workers)),
        chunk_size=int(args.chunk_size),
    )
    timings["load_15m"] = time.perf_counter() - t0
    LOG.info("15m panels %d/%d (%.1fs)", len(panels_15m), len(symbols), timings["load_15m"])

    t0 = time.perf_counter()
    indexed: Dict[str, dict] = {}
    for i_sym, (sym, df) in enumerate(panels_15m.items(), 1):
        if df is None or df.empty:
            continue
        indexed[str(sym).upper()] = index_rth_15m_fast(df, naive_tz="UTC")
        if i_sym % 50 == 0:
            LOG.info("Indexed 15m %d/%d", i_sym, len(panels_15m))
    timings["index_15m"] = time.perf_counter() - t0
    LOG.info("Indexed 15m %d names (%.1fs)", len(indexed), timings["index_15m"])

    mode_arg = str(args.fill_mode).strip().lower().replace("_", "-")
    modes: List[str]
    if mode_arg in ("both", "all"):
        modes = ["next-mid", "signal-close"]
    else:
        modes = [mode_arg]

    t0 = time.perf_counter()
    for mode in modes:
        params = ReclaimParams(fill_mode=mode)
        _run_book(
            mode,
            indexed,
            params,
            max_open=int(args.max_open),
            friction_pct=float(args.friction_pct),
            outdir=outdir,
        )
    timings["fills"] = time.perf_counter() - t0
    timings["wall"] = time.perf_counter() - t_all
    LOG.info("Wall %.1fs %s", timings["wall"], timings)
    (outdir / "vwap_ema_reclaim_timings.txt").write_text(
        "\n".join("%s=%.3f" % (k, v) for k, v in timings.items()) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
