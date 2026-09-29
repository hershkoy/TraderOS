#!/usr/bin/env python3
"""Claim-check QS RSI+IBS mean-reversion on Nasdaq-100 stocks.

Rules (from video tjL4wcI403E / public summary):
  close > SMA200, RSI(3) < 20, IBS < 0.3; exit close > prior high;
  next-open fills; max 5 open positions.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_qs_rsi_ndx.py
  python scripts\\research\\claimcheck_qs_rsi_ndx.py --friction 0.25 --start 2010-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402
from utils.research.choppy_regime import occupancy_ranked  # noqa: E402
from utils.research.panel import frames_to_wide, list_daily_symbols  # noqa: E402
from utils.research.qs_rsi_ndx import (  # noqa: E402
    CLAIMED,
    book_stats,
    collect_panel_trades,
    portfolio_equity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_qs_rsi_ndx")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "qs_rsi_ndx"


def nasdaq100_symbols() -> List[str]:
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB connection failed")
    q = """
    SELECT DISTINCT UPPER(symbol)
    FROM ticker_universe
    WHERE is_active = TRUE AND index_name = 'nasdaq100'
    ORDER BY 1
    """
    rows = client.execute_query(q)
    return [str(r[0]).upper() for r in (rows or [])]


def _frame_norm(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if "ts" not in out.columns:
                out = out.rename(columns={out.columns[0]: "ts"})
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t_wall = time.perf_counter()
    ndx = nasdaq100_symbols()
    daily = {s.upper() for s in list_daily_symbols(provider=args.provider, timeframe="1d")}
    symbols = sorted(set(ndx) & daily)
    LOG.info("NDX members=%d with %s 1d bars=%d", len(ndx), args.provider, len(symbols))
    if args.limit and args.limit > 0:
        symbols = symbols[: int(args.limit)]
        LOG.info("Limited to first %d symbols", len(symbols))

    start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else datetime(2006, 1, 1)
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    t0 = time.perf_counter()
    frames_raw = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider=args.provider,
        start=start,
        end=end,
        use_cache=not args.no_cache,
        workers=int(args.workers),
        fallback_provider=("IB" if args.ib_prefix else None),
        merge_mode=("prefix" if args.ib_prefix else "replace"),
    )
    frames = {s: _frame_norm(df) for s, df in frames_raw.items() if df is not None and not df.empty}
    load_s = time.perf_counter() - t0
    LOG.info("Loaded %d/%d frames in %.1fs", len(frames), len(symbols), load_s)

    # Wide close panel needs DatetimeIndex (ts column -> index).
    frames_idx: Dict[str, pd.DataFrame] = {}
    for s, df in frames.items():
        x = df.copy()
        if "ts" in x.columns:
            x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
            x = x.dropna(subset=["ts"]).set_index("ts")
        frames_idx[s] = x

    t0 = time.perf_counter()
    raw = collect_panel_trades(
        frames,
        rsi_period=int(args.rsi_period),
        rsi_entry=float(args.rsi_entry),
        ibs_max=float(args.ibs_max),
        sma_trend=int(args.sma),
        friction_pct=float(args.friction),
        start=args.start,
        end=args.end,
    )
    collect_s = time.perf_counter() - t0
    LOG.info("Unconstrained trades=%d (%.1fs)", len(raw), collect_s)

    occ = occupancy_ranked(raw, max_open=int(args.max_open), score_ascending=True)
    raw_stats = book_stats(raw)
    occ_stats = book_stats(occ)

    close_panel = frames_to_wide(frames_idx, "close")
    if not close_panel.empty:
        cal = close_panel.index
    else:
        cal = pd.DatetimeIndex([])
    LOG.info(
        "Close panel %s x %s  %s -> %s",
        close_panel.shape[0],
        close_panel.shape[1],
        cal.min() if len(cal) else None,
        cal.max() if len(cal) else None,
    )
    port = portfolio_equity(occ, cal, close_panel, max_open=int(args.max_open))

    wall = time.perf_counter() - t_wall
    result: Dict[str, Any] = {
        "claimed": CLAIMED,
        "params": {
            "rsi_period": int(args.rsi_period),
            "rsi_entry": float(args.rsi_entry),
            "ibs_max": float(args.ibs_max),
            "sma": int(args.sma),
            "max_open": int(args.max_open),
            "friction_pct": float(args.friction),
            "provider": args.provider,
            "ib_prefix": bool(args.ib_prefix),
            "start": args.start or "2006-01-01",
            "end": args.end,
            "n_symbols_ndx": len(ndx),
            "n_symbols_used": len(frames),
            "survivorship_note": (
                "Current NDX members only (ticker_universe); not Norgate "
                "point-in-time constituents / delistings."
            ),
        },
        "unconstrained": raw_stats,
        "occupancy": occ_stats,
        "portfolio": {
            "cagr_pct": port.get("cagr_pct"),
            "mdd_pct": port.get("mdd_pct"),
            "exposure_pct": port.get("exposure_pct"),
            "sharpe": port.get("sharpe"),
            "final_equity": port.get("final_equity"),
        },
        "timings_s": {
            "load": round(load_s, 2),
            "collect": round(collect_s, 2),
            "wall": round(wall, 2),
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "qs_rsi_ndx_r%dlt%s_ibs%s_occ%d_fric%s" % (
        int(args.rsi_period),
        str(args.rsi_entry).replace(".", "p"),
        str(args.ibs_max).replace(".", "p"),
        int(args.max_open),
        str(args.friction).replace(".", "p"),
    )
    if not occ.empty:
        occ.to_csv(outdir / ("%s_trades.csv" % stem), index=False)
    (outdir / ("%s_summary.json" % stem)).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    eq = port.get("equity")
    if eq is not None and len(eq):
        eq.to_csv(outdir / ("%s_equity.csv" % stem), header=["equity"])

    _print_compare(result)
    LOG.info("Wrote %s (wall %.1fs)", outdir / ("%s_summary.json" % stem), wall)
    return result


def _print_compare(result: Dict[str, Any]) -> None:
    c = result["claimed"]
    o = result["occupancy"]
    p = result["portfolio"]
    LOG.info("=== Claim vs ours (occ-%d) ===", result["params"]["max_open"])
    rows = [
        ("n", c["n"], o.get("n")),
        ("WR%", c["win_rate_pct"], o.get("win_rate_pct")),
        ("E%", c["expectancy_pct"], o.get("expectancy_pct")),
        ("avg_win%", c["avg_win_pct"], o.get("avg_win_pct")),
        ("avg_loss%", c["avg_loss_pct"], o.get("avg_loss_pct")),
        ("PF", c["profit_factor"], o.get("profit_factor")),
        ("CAGR%", c["cagr_pct"], p.get("cagr_pct")),
        ("exposure%", c["exposure_pct"], p.get("exposure_pct")),
        ("MDD%", c["mdd_pct"], p.get("mdd_pct")),
    ]
    for name, claimed, got in rows:
        LOG.info("  %-12s claimed=%-8s ours=%s", name, claimed, got)
    LOG.info("  geo_year_PF=%s  Sharpe=%s", o.get("geo_mean_year_pf"), p.get("sharpe"))
    for y in o.get("by_year") or []:
        LOG.info(
            "  year %-9s n=%-4s E=%-7s WR=%-6s PF=%s",
            y.get("bucket"),
            y.get("n"),
            y.get("expectancy_pct"),
            y.get("win_rate_pct"),
            y.get("profit_factor"),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Claim-check QS RSI+IBS on NDX stocks")
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--ib-prefix", action="store_true", default=True)
    ap.add_argument("--no-ib-prefix", action="store_false", dest="ib_prefix")
    ap.add_argument("--rsi-period", type=int, default=3)
    ap.add_argument("--rsi-entry", type=float, default=20.0)
    ap.add_argument("--ibs-max", type=float, default=0.3)
    ap.add_argument("--sma", type=int, default=200)
    ap.add_argument("--max-open", type=int, default=5)
    ap.add_argument("--friction", type=float, default=0.0, help="Round-turn friction percent")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Debug: first N symbols")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
