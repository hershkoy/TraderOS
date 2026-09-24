#!/usr/bin/env python3
"""Backtest Quantified Strategies 7-day ROC rebound on SPY.

Rules:
  Buy next open when 7d ROC closes below -3% (flat).
  Sell next open when 7d ROC closes above +3% (long).
  No stop / no target.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_roc_rebound.py
  python scripts\\research\\backtest_roc_rebound.py --refresh-yahoo --friction 0.25
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.roc_rebound import (  # noqa: E402
    CLAIMED,
    backtest_roc_rebound,
    sweep_exit_thresholds,
    sweep_roc_lengths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_roc_rebound")

DEFAULT_OUTDIR = ROOT / "reports" / "research" / "roc_rebound"
YAHOO_CACHE = ROOT / "reports" / "research" / "spy_yahoo.json"


def download_yahoo(path: Path = YAHOO_CACHE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
        "?period1=727876800&period2=9999999999&interval=1d&events=history"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    logger.info("Downloaded Yahoo SPY -> %s (%d bytes)", path, len(data))
    return path


def load_yahoo(path: Path = YAHOO_CACHE, refresh: bool = False) -> pd.DataFrame:
    if refresh or not path.exists() or path.stat().st_size < 10_000:
        download_yahoo(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, unit="s", utc=True),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    )
    df = (
        df.dropna(subset=["close", "open"])
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )
    logger.info(
        "Yahoo SPY: %d bars  %s -> %s",
        len(df),
        df["ts"].iloc[0].date(),
        df["ts"].iloc[-1].date(),
    )
    return df


def load_timescale_spy() -> Optional[pd.DataFrame]:
    try:
        from utils.db.timescaledb_client import get_timescaledb_client
        from utils.scanning.deep_pullback_spy import load_spy_daily
    except Exception as exc:  # pragma: no cover
        logger.warning("TimescaleDB import failed: %s", exc)
        return None
    try:
        client = get_timescaledb_client()
        # Full history (no lookback clip) for research claim-check.
        df = load_spy_daily(client, lookback_bars=0)
    except Exception as exc:
        logger.warning("TimescaleDB SPY load failed: %s", exc)
        return None
    if df is None or df.empty:
        return None
    logger.info(
        "TimescaleDB SPY: %d bars  %s -> %s",
        len(df),
        pd.Timestamp(df["ts"].iloc[0]).date(),
        pd.Timestamp(df["ts"].iloc[-1]).date(),
    )
    return df


def _summary_row(label: str, r: Dict[str, Any]) -> str:
    return (
        f"{label:18s}  n={r.get('n', 0):4d}  WR={r.get('win_rate_pct')}%  "
        f"PF={r.get('profit_factor')}  E={r.get('expectancy_pct')}%  "
        f"eq={r.get('final_equity')}  MDD={r.get('mdd_trade_eq_pct')}%"
    )


def _print_claim_delta(ours: Dict[str, Any]) -> None:
    logger.info("--- Claim check (video SF1OtXR5crQ) ---")
    logger.info(
        "Claimed: n=%s WR=%s%% PF=%s eq~%s MDD~%s%%",
        CLAIMED["n"],
        CLAIMED["win_rate_pct"],
        CLAIMED["profit_factor"],
        CLAIMED["final_equity"],
        CLAIMED["mdd_pct"],
    )
    logger.info(
        "Ours:    n=%s WR=%s%% PF=%s eq=%s MDD=%s%%",
        ours.get("n"),
        ours.get("win_rate_pct"),
        ours.get("profit_factor"),
        ours.get("final_equity"),
        ours.get("mdd_trade_eq_pct"),
    )
    for era in ours.get("by_era") or []:
        c = CLAIMED["eras"].get(era["bucket"], {})
        logger.info(
            "  Era %-10s  ours n=%s WR=%s PF=%s | claimed n=%s WR=%s PF=%s",
            era["bucket"],
            era.get("n"),
            era.get("win_rate_pct"),
            era.get("profit_factor"),
            c.get("n"),
            c.get("win_rate_pct"),
            c.get("profit_factor"),
        )


def run(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    yahoo = load_yahoo(refresh=args.refresh_yahoo)
    if args.start:
        yahoo = yahoo[yahoo["ts"] >= pd.Timestamp(args.start, tz="UTC")].reset_index(drop=True)

    primary = backtest_roc_rebound(
        yahoo,
        symbol="SPY",
        roc_length=args.roc_length,
        entry_thresh_pct=args.entry_thresh,
        exit_thresh_pct=args.exit_thresh,
        friction_pct=args.friction,
    )
    logger.info(_summary_row("Yahoo primary", primary))
    _print_claim_delta(primary)

    length_sweep = sweep_roc_lengths(
        yahoo,
        entry_thresh_pct=args.entry_thresh,
        exit_thresh_pct=args.exit_thresh,
        friction_pct=args.friction,
    )
    logger.info("--- ROC length sweep (4-11) ---")
    for row in length_sweep:
        logger.info(
            "  len=%2d  n=%4d  WR=%s  PF=%s  E=%s  eq=%s",
            row["roc_length"],
            row["n"],
            row["win_rate_pct"],
            row["profit_factor"],
            row["expectancy_pct"],
            row["final_equity"],
        )

    exit_sweep = sweep_exit_thresholds(
        yahoo,
        roc_length=args.roc_length,
        entry_thresh_pct=args.entry_thresh,
        friction_pct=args.friction,
    )
    logger.info("--- Exit threshold sweep (+1..+5) ---")
    for row in exit_sweep:
        logger.info(
            "  exit=+%.1f  n=%4d  WR=%s  PF=%s  E=%s  eq=%s",
            row["exit_thresh_pct"],
            row["n"],
            row["win_rate_pct"],
            row["profit_factor"],
            row["expectancy_pct"],
            row["final_equity"],
        )

    fric = None
    if args.friction == 0.0:
        fric = backtest_roc_rebound(
            yahoo,
            symbol="SPY",
            roc_length=args.roc_length,
            entry_thresh_pct=args.entry_thresh,
            exit_thresh_pct=args.exit_thresh,
            friction_pct=0.25,
        )
        logger.info(_summary_row("Yahoo fric0.25", fric))

    ts_result = None
    if not args.skip_timescale:
        tdf = load_timescale_spy()
        if tdf is not None and not tdf.empty:
            ts_result = backtest_roc_rebound(
                tdf,
                symbol="SPY",
                roc_length=args.roc_length,
                entry_thresh_pct=args.entry_thresh,
                exit_thresh_pct=args.exit_thresh,
                friction_pct=args.friction,
            )
            logger.info(_summary_row("TimescaleDB", ts_result))

    elapsed = time.perf_counter() - t0
    payload: Dict[str, Any] = {
        "claimed": CLAIMED,
        "yahoo_primary": {k: v for k, v in primary.items() if k != "trades"},
        "yahoo_trades": primary.get("trades") or [],
        "length_sweep": length_sweep,
        "exit_sweep": exit_sweep,
        "yahoo_friction_0_25": None
        if fric is None
        else {k: v for k, v in fric.items() if k != "trades"},
        "timescale": None
        if ts_result is None
        else {k: v for k, v in ts_result.items() if k != "trades"},
        "elapsed_sec": round(elapsed, 2),
        "bars": {
            "yahoo": int(len(yahoo)),
            "yahoo_first": str(pd.Timestamp(yahoo["ts"].iloc[0]).date()),
            "yahoo_last": str(pd.Timestamp(yahoo["ts"].iloc[-1]).date()),
        },
    }

    summary_path = outdir / "roc_rebound_spy_summary.json"
    trades_path = outdir / "roc_rebound_spy_trades.csv"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(primary.get("trades") or []).to_csv(trades_path, index=False)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s (%d trades)", trades_path, primary.get("n", 0))
    logger.info("Wall-clock %.1fs", elapsed)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SPY 7d ROC rebound claim-check")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--skip-timescale", action="store_true")
    ap.add_argument("--start", type=str, default=None, help="Optional ISO start date")
    ap.add_argument("--roc-length", type=int, default=7)
    ap.add_argument("--entry-thresh", type=float, default=3.0)
    ap.add_argument("--exit-thresh", type=float, default=3.0)
    ap.add_argument("--friction", type=float, default=0.0, help="Round-turn friction %%")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
