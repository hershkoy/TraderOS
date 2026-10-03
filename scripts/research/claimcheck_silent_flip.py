#!/usr/bin/env python3
"""Claim-check ProRealAlgos Silent Flip on IB 15m RTH bars.

Levels are prior-session high/low plus the most recent beyond-range session
extreme. Entry is a 3-candle open sequence (touch, silent opposite candle,
stop at the silent extreme). Stop sits beyond the tested line; target is the
opposite inner range rail. Day trade; 15:45 ET flatten.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python -u scripts\\research\\claimcheck_silent_flip.py --symbols UBER,NVDA,BAC,PFE,XOM
  python -u scripts\\research\\claimcheck_silent_flip.py --n-symbols 300
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.silent_flip import (  # noqa: E402
    CLAIMED,
    SilentFlipParams,
    TRAIL_BREAKEVEN,
    TRAIL_NONE,
    VIDEO_SYMBOLS,
    WINDOW_FIRST_60,
    WINDOW_OPEN_ONLY,
    collect_symbol_trades,
    drop_top_n_expectancy,
    index_rth_15m_fast,
    max_concurrent,
    occupancy_take,
    rows_from_raw,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_silent_flip")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

FRICTION_PCT = 0.10
DEFAULT_OUT = ROOT / "reports" / "research" / "silent_flip"


def _stats_block(label: str, trades: pd.DataFrame, gain_col: str = "gain_pct") -> Dict[str, Any]:
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
        if "side" in trades.columns:
            print(trades["side"].value_counts().to_string())
        if "level_name" in trades.columns:
            print(trades["level_name"].value_counts().to_string())
        if "exit_reason" in trades.columns:
            print(trades["exit_reason"].value_counts().to_string())
    return {
        "label": label,
        "stats": s,
        "geo_pf": None if geo is None else round(float(geo), 3),
        "drop_top3_e": None if drop3 is None else round(float(drop3), 4),
        "years": years.to_dict(orient="records") if years is not None and not years.empty else [],
    }


def _collect_raw(indexed: Dict[str, dict], params: SilentFlipParams) -> List[dict]:
    raw: List[dict] = []
    n_sym = len(indexed)
    for i, (sym, by_day) in enumerate(indexed.items(), 1):
        rows = collect_symbol_trades(by_day, params=params)
        raw.extend(rows_from_raw(rows, sym))
        if i % 50 == 0:
            LOG.info("scan %d/%d trades=%d", i, n_sym, len(raw))
    return raw


def _run_book(
    label: str,
    raw: List[dict],
    params: SilentFlipParams,
    *,
    max_open: int,
    friction_pct: float,
    outdir: Path,
) -> pd.DataFrame:
    cap = 10**9 if int(max_open) <= 0 else int(max_open)
    occupied = occupancy_take(raw, max_open=cap) if raw else []
    trades = pd.DataFrame(occupied)
    if not trades.empty:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"]).dt.strftime("%Y-%m-%d")
        trades["sell_date"] = pd.to_datetime(trades["sell_date"]).dt.strftime("%Y-%m-%d")
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    csv_path = outdir / ("silent_flip_%s_trades.csv" % stem)
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
    summary = _stats_block("%s gross" % label, trades)
    net_summary: Optional[Dict[str, Any]] = None
    if not trades.empty:
        net = apply_friction(trades, float(friction_pct))
        net_summary = _stats_block("%s fric%.2f" % (label, float(friction_pct)), net, gain_col="gain_pct_net")
    payload = {
        "claimed": CLAIMED,
        "params": {
            "window": params.window,
            "min_body_frac": params.min_body_frac,
            "touch_tol_pct": params.touch_tol_pct,
            "stop_buffer_pct": params.stop_buffer_pct,
            "trail": params.trail,
            "max_open": int(max_open),
            "friction_pct": float(friction_pct),
        },
        "n_raw": len(raw),
        "n_kept": len(trades),
        "max_concurrent": peak,
        "median_concurrent": med,
        "gross": summary,
        "net": net_summary,
    }
    (outdir / ("silent_flip_%s_summary.json" % stem)).write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return trades


def _parse_symbols(text: str) -> List[str]:
    return [p.strip().upper() for p in str(text).split(",") if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Silent Flip 15m claim-check")
    ap.add_argument("--n-symbols", type=int, default=300)
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--symbols", default="", help="Comma list; default is ADV universe")
    ap.add_argument("--video-names", action="store_true", help="NVDA,BAC,PFE,UBER,XOM")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--max-open", type=int, default=0, help="0 = unique-symbol uncapped")
    ap.add_argument("--occ", default="0,8,15", help="Occupancy caps besides the primary book")
    ap.add_argument("--friction-pct", type=float, default=FRICTION_PCT)
    ap.add_argument("--window", default=WINDOW_FIRST_60, choices=(WINDOW_FIRST_60, WINDOW_OPEN_ONLY))
    ap.add_argument("--min-body-frac", type=float, default=0.30)
    ap.add_argument("--trail", default=TRAIL_NONE, choices=(TRAIL_NONE, TRAIL_BREAKEVEN))
    ap.add_argument("--variants", default="baseline", help="baseline|open_only|no_strong|breakeven|all")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir or dated_outdir(DEFAULT_OUT)
    outdir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    t0 = time.perf_counter()
    if args.video_names:
        symbols = list(VIDEO_SYMBOLS)
    elif str(args.symbols).strip():
        symbols = _parse_symbols(args.symbols)
    elif args.all_symbols:
        from utils.research.panel import list_daily_symbols  # noqa: WPS433

        symbols = list_daily_symbols("IB", "15m")
    else:
        symbols = _pick_15m_by_daily_adv(int(args.n_symbols))
    timings["list_symbols"] = time.perf_counter() - t0
    LOG.info("Universe %d symbols (%.1fs)", len(symbols), timings["list_symbols"])
    if not symbols:
        LOG.error("Universe empty")
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

    variant_arg = str(args.variants).strip().lower()
    variants: List[tuple[str, SilentFlipParams]]
    if variant_arg in ("all", "suite"):
        variants = [
            ("baseline", SilentFlipParams(window=WINDOW_FIRST_60, min_body_frac=0.30, trail=TRAIL_NONE)),
            ("open_only", SilentFlipParams(window=WINDOW_OPEN_ONLY, min_body_frac=0.30, trail=TRAIL_NONE)),
            ("no_strong", SilentFlipParams(window=WINDOW_FIRST_60, min_body_frac=0.0, trail=TRAIL_NONE)),
            ("breakeven", SilentFlipParams(window=WINDOW_FIRST_60, min_body_frac=0.30, trail=TRAIL_BREAKEVEN)),
        ]
    elif variant_arg == "open_only":
        variants = [("open_only", SilentFlipParams(window=WINDOW_OPEN_ONLY, min_body_frac=float(args.min_body_frac)))]
    elif variant_arg == "no_strong":
        variants = [("no_strong", SilentFlipParams(window=WINDOW_FIRST_60, min_body_frac=0.0))]
    elif variant_arg in ("breakeven", "trail"):
        variants = [("breakeven", SilentFlipParams(window=args.window, min_body_frac=float(args.min_body_frac), trail=TRAIL_BREAKEVEN))]
    else:
        variants = [
            (
                "baseline",
                SilentFlipParams(
                    window=str(args.window),
                    min_body_frac=float(args.min_body_frac),
                    trail=str(args.trail),
                ),
            )
        ]

    occ_caps = [int(x) for x in str(args.occ).split(",") if str(x).strip() != ""]
    t0 = time.perf_counter()
    for name, params in variants:
        LOG.info("Collect %s", name)
        raw = _collect_raw(indexed, params)
        _run_book(
            name,
            raw,
            params,
            max_open=int(args.max_open),
            friction_pct=float(args.friction_pct),
            outdir=outdir,
        )
        for cap in occ_caps:
            if cap <= 0 or cap == int(args.max_open):
                continue
            if not raw:
                continue
            _run_book(
                "%s_occ%d" % (name, cap),
                raw,
                params,
                max_open=cap,
                friction_pct=float(args.friction_pct),
                outdir=outdir,
            )
    timings["fills"] = time.perf_counter() - t0
    timings["wall"] = time.perf_counter() - t_all
    LOG.info("Wall %.1fs %s", timings["wall"], timings)
    (outdir / "silent_flip_timings.txt").write_text(
        "\n".join("%s=%.3f" % (k, v) for k, v in timings.items()) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
