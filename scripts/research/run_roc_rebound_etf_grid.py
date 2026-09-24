#!/usr/bin/env python3
"""ETF-only ROC rebound robustness grid.

Variants:
  A0 baseline (fixed +/-3%)
  A1 + SMA200 trend filter
  A2 vol-scaled entry (ROC < -k * daily_vol% * sqrt(7))
  A3 A1 + A2
  T20 / T40 time-stop overlays on A0 and A3

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_roc_rebound_etf_grid.py
  python scripts\\research\\run_roc_rebound_etf_grid.py --friction 0.10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.roc_rebound import (  # noqa: E402
    ERA_BUCKETS,
    YEAR_BUCKETS,
    backtest_roc_rebound,
    book_from_trades,
    drop_top_n_book,
    geom_mean_year_pf,
    year_bucket_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("roc_rebound_etf_grid")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "roc_rebound"
YAHOO_CACHE_DIR = DEFAULT_OUT / "yahoo_cache"

ETF_UNIVERSE: Sequence[str] = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "MDY",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLU",
    "XLB",
    "XLC",
    "GLD",
    "EFA",
    "EEM",
)

# Fixed overlays — not a parameter fish.
GRID: Sequence[Tuple[str, Dict[str, Any]]] = (
    ("A0_baseline", {}),
    ("A1_sma200", {"require_sma200": True}),
    ("A2_vol_scale", {"vol_scale": True, "vol_k": 1.5, "vol_lookback": 20}),
    (
        "A3_sma200_vol",
        {"require_sma200": True, "vol_scale": True, "vol_k": 1.5, "vol_lookback": 20},
    ),
    ("T20_on_A0", {"time_stop_bars": 20}),
    ("T40_on_A0", {"time_stop_bars": 40}),
    (
        "T20_on_A3",
        {
            "require_sma200": True,
            "vol_scale": True,
            "vol_k": 1.5,
            "vol_lookback": 20,
            "time_stop_bars": 20,
        },
    ),
    (
        "T40_on_A3",
        {
            "require_sma200": True,
            "vol_scale": True,
            "vol_k": 1.5,
            "vol_lookback": 20,
            "time_stop_bars": 40,
        },
    ),
)


def _frame_to_bt(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if "datetime" in out.columns:
            out = out.rename(columns={"datetime": "ts"})
        elif out.columns[0] != "ts":
            out = out.rename(columns={out.columns[0]: "ts"})
    return out


def load_yahoo_daily(symbol: str, *, refresh: bool = False) -> pd.DataFrame:
    YAHOO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = YAHOO_CACHE_DIR / ("%s.json" % symbol.upper())
    if refresh or not path.exists() or path.stat().st_size < 5_000:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/%s"
            "?period1=727876800&period2=9999999999&interval=1d&events=history"
            % symbol.upper()
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
        LOG.info("Downloaded Yahoo %s", symbol)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(result["timestamp"], unit="s", utc=True),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    )
    return (
        df.dropna(subset=["close", "open"])
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )


def load_etf_frames(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: Optional[datetime],
    workers: int,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    syms = [s.upper() for s in symbols]
    try:
        frames = load_ohlcv_many(
            syms,
            timeframe="1d",
            provider="ALPACA",
            start=start,
            end=end,
            use_cache=True,
            workers=workers,
            chunk_size=20,
            fallback_provider="IB",
            merge_mode="prefix",
        )
        for s, df in frames.items():
            if df is not None and not df.empty:
                out[s.upper()] = _frame_to_bt(df)
    except Exception as exc:
        LOG.warning("TimescaleDB load failed (%s); Yahoo-only", exc)

    for s in syms:
        df = out.get(s)
        if df is not None and len(df) >= 250:
            continue
        try:
            ydf = load_yahoo_daily(s)
            if start is not None:
                ydf = ydf[ydf["ts"] >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                ydf = ydf[ydf["ts"] <= pd.Timestamp(end, tz="UTC")]
            # Prefer longer Yahoo history for ETFs when DB is short.
            if df is None or len(ydf) > len(df):
                out[s] = ydf.reset_index(drop=True)
                LOG.info("Using Yahoo %s (%d bars)", s, len(out[s]))
        except Exception as exc:
            LOG.warning("Yahoo failed %s: %s", s, exc)
            if s not in out:
                out[s] = pd.DataFrame()
    return out


def _variant_book(
    frames: Dict[str, pd.DataFrame],
    *,
    name: str,
    overlay: Dict[str, Any],
    friction_pct: float,
) -> Dict[str, Any]:
    all_trades: List[Dict[str, Any]] = []
    per_symbol: List[Dict[str, Any]] = []
    for sym, raw in sorted(frames.items()):
        df = _frame_to_bt(raw)
        res = backtest_roc_rebound(
            df,
            symbol=sym,
            friction_pct=friction_pct,
            **overlay,
        )
        per_symbol.append(
            {
                "symbol": sym,
                "n": res.get("n", 0),
                "E": res.get("expectancy_pct"),
                "PF": res.get("profit_factor"),
                "WR": res.get("win_rate_pct"),
                "mdd": res.get("mdd_trade_eq_pct"),
                "median": res.get("median_pct"),
                "avg_hold": res.get("avg_hold_days"),
                "bars": res.get("bars"),
            }
        )
        for tr in res.get("trades") or []:
            tr = dict(tr)
            tr["variant"] = name
            all_trades.append(tr)

    book = book_from_trades(all_trades)
    eras = year_bucket_stats(all_trades, ERA_BUCKETS)
    years = year_bucket_stats(all_trades, YEAR_BUCKETS)
    drop3 = drop_top_n_book(all_trades, 3)
    drop5 = drop_top_n_book(all_trades, 5)
    # Winner-$ gate: did PF lift vs A0 by clipping fat winners?
    return {
        "variant": name,
        "overlay": overlay,
        "friction_pct": friction_pct,
        "book": book,
        "by_era": eras,
        "by_year_bucket": years,
        "geo_pf_year_min10": geom_mean_year_pf(years, min_n=10),
        "geo_pf_year_min30": geom_mean_year_pf(years, min_n=30),
        "drop_top3": drop3,
        "drop_top5": drop5,
        "per_symbol": per_symbol,
        "n_symbols_with_trades": int(sum(1 for r in per_symbol if int(r["n"] or 0) > 0)),
        "trades": all_trades,
    }


def run_grid(
    frames: Dict[str, pd.DataFrame],
    *,
    friction_pct: float,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for name, overlay in GRID:
        t0 = time.perf_counter()
        row = _variant_book(frames, name=name, overlay=overlay, friction_pct=friction_pct)
        row["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        results.append(row)
        b = row["book"]
        LOG.info(
            "%-16s n=%4s WR=%s PF=%s E=%s med=%s MDD=%s geo10=%s drop3_PF=%s drop3_E=%s",
            name,
            b.get("n"),
            b.get("win_rate_pct"),
            b.get("profit_factor"),
            b.get("expectancy_pct"),
            b.get("median_pct"),
            b.get("mdd_trade_eq_pct"),
            None
            if row.get("geo_pf_year_min10") is None
            else round(float(row["geo_pf_year_min10"]), 3),
            (row.get("drop_top3") or {}).get("profit_factor"),
            (row.get("drop_top3") or {}).get("expectancy_pct"),
        )
        for era in row.get("by_era") or []:
            if int(era.get("n") or 0) <= 0:
                continue
            LOG.info(
                "                 era %-10s n=%s WR=%s PF=%s E=%s",
                era.get("bucket"),
                era.get("n"),
                era.get("win_rate_pct"),
                era.get("profit_factor"),
                era.get("expectancy_pct"),
            )
    return {"variants": results, "friction_pct": friction_pct}


def _verdict(variants: List[Dict[str, Any]]) -> str:
    by_name = {v["variant"]: v for v in variants}
    a0 = by_name.get("A0_baseline")
    if not a0:
        return "no_baseline"
    base_pf = (a0.get("book") or {}).get("profit_factor") or 0.0
    base_e = (a0.get("book") or {}).get("expectancy_pct") or 0.0
    base_sum_win = (a0.get("book") or {}).get("sum_win_pct") or 0.0
    notes: List[str] = []
    for v in variants:
        if v["variant"] == "A0_baseline":
            continue
        b = v.get("book") or {}
        pf = b.get("profit_factor")
        e = b.get("expectancy_pct")
        if pf is None or e is None:
            notes.append("%s: empty" % v["variant"])
            continue
        sum_win = b.get("sum_win_pct") or 0.0
        # Winner-$: PF up but total winner dollars down a lot vs baseline.
        winner_clip = (
            pf > base_pf
            and base_sum_win > 0
            and sum_win < 0.85 * base_sum_win
            and e < base_e
        )
        geo = v.get("geo_pf_year_min10")
        era_ok = all(
            (er.get("profit_factor") or 0) >= 1.0
            for er in (v.get("by_era") or [])
            if int(er.get("n") or 0) >= 20
        )
        lift = pf >= base_pf * 1.05 and e >= base_e * 0.95
        if winner_clip:
            notes.append("%s: REJECT winner-$ clip (PF %.2f E %.2f)" % (v["variant"], pf, e))
        elif lift and era_ok and (geo is None or geo >= 1.1):
            notes.append("%s: INTERESTING PF %.2f E %.2f geo10=%s" % (v["variant"], pf, e, geo))
        else:
            notes.append(
                "%s: no promote (PF %.2f E %.2f vs A0 %.2f/%.2f)"
                % (v["variant"], pf, e, base_pf, base_e)
            )
    return " | ".join(notes)


def main() -> int:
    ap = argparse.ArgumentParser(description="ROC rebound ETF robustness grid")
    ap.add_argument("--friction", type=float, default=0.10, help="Round-trip friction %%")
    ap.add_argument("--also-zero", action="store_true", help="Also run friction 0 grid")
    ap.add_argument("--start", type=str, default="1993-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument(
        "--symbols",
        type=str,
        default=",".join(ETF_UNIVERSE),
        help="Comma-separated ETF list",
    )
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else None

    t0 = time.perf_counter()
    frames = load_etf_frames(symbols, start=start, end=end, workers=args.workers)
    loaded = {s: len(df) for s, df in frames.items() if df is not None and not df.empty}
    LOG.info("Loaded %d ETFs: %s", len(loaded), loaded)

    frictions = [float(args.friction)]
    if args.also_zero and float(args.friction) != 0.0:
        frictions = [0.0, float(args.friction)]

    all_runs: Dict[str, Any] = {"symbols": symbols, "start": args.start, "grids": {}}
    for fric in frictions:
        LOG.info("=== Grid friction=%.2f%% ===", fric)
        grid = run_grid(frames, friction_pct=fric)
        verdict = _verdict(grid["variants"])
        LOG.info("Verdict: %s", verdict)
        # Strip heavy trade lists from summary copy; keep separately.
        slim_variants = []
        for v in grid["variants"]:
            slim = {k: val for k, val in v.items() if k != "trades"}
            slim_variants.append(slim)
        all_runs["grids"][str(fric)] = {
            "variants": slim_variants,
            "verdict": verdict,
        }

    elapsed = time.perf_counter() - t0
    all_runs["elapsed_sec"] = round(elapsed, 2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "roc_rebound_etf_grid_summary.json"
    path.write_text(json.dumps(all_runs, indent=2, default=str), encoding="utf-8")

    # Flat CSV of variant books for the primary friction.
    primary = all_runs["grids"][str(float(args.friction))]["variants"]
    rows = []
    for v in primary:
        b = v.get("book") or {}
        rows.append(
            {
                "variant": v["variant"],
                "friction": args.friction,
                "n": b.get("n"),
                "WR": b.get("win_rate_pct"),
                "PF": b.get("profit_factor"),
                "E": b.get("expectancy_pct"),
                "median": b.get("median_pct"),
                "MDD": b.get("mdd_trade_eq_pct"),
                "avg_hold": b.get("avg_hold_days"),
                "geo10": v.get("geo_pf_year_min10"),
                "drop3_PF": (v.get("drop_top3") or {}).get("profit_factor"),
                "drop3_E": (v.get("drop_top3") or {}).get("expectancy_pct"),
                "sum_win": b.get("sum_win_pct"),
                "sum_loss": b.get("sum_loss_pct"),
            }
        )
    csv_path = outdir / "roc_rebound_etf_grid.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    LOG.info("Wrote %s", path)
    LOG.info("Wrote %s", csv_path)
    LOG.info("Wall-clock %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
