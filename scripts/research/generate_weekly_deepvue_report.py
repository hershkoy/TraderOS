#!/usr/bin/env python3
"""TradingView-style HTML report for the weekly Deepvue occupancy book.

Reuses generate_channel_touch_tv_report.render_html. Channel JSON is empty
(this book has no rails). Buy/sell clocks are UTC in the CSV and display as
America/New_York RTH.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_weekly_deepvue_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.generate_channel_touch_tv_report import (
    build_run_meta,
    ensure_report_trade_columns,
    load_spy_close,
    render_html,
    spy_to_raw,
    trades_to_raw,
    write_summary_sidecar,
)
from utils.research.report_paths import dated_outdir, is_iso_date_dir_name, iter_report_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_weekly_deepvue_report")

DEFAULT_TRADES_DIR = ROOT / "reports" / "weekly_deepvue"

DETECTOR_META: Dict[str, Any] = {
    "name": "Deepvue-style weekend scan (causal; not Deepvue)",
    "script": "scripts/research/backtest_weekly_deepvue.py",
    "pine": "n/a (not a channel detector)",
    "version": "research 2026-09-12",
    "notes": (
        "QQQ close > rising 21 EMA; ADV >= $20M; ADR >= 3%; close > 21 EMA; "
        "21d and 63d RS percentile >= 85; RMV = 100 * (5d range / 15d range) <= 45; "
        "near 10 or 21 EMA within 2%; top 6 sectors; max 2/sector; weekly focus 15 take 4; "
        "occupancy max 4 unique symbols. Buy next RTH 09:30 ET 15m mid. "
        "Sell 3% hard stop on 15m bar N fill N+1 mid, or two daily closes below 21 EMA "
        "then next 15m mid. No promote."
    ),
    "params": [],
}

SUMMARY_PARAMS: Dict[str, Any] = {
    "entry": "next_session_open_mid",
    "exit": "hard_stop_15m_n1_mid_or_ma_2close",
    "stop_pct": 0.03,
    "ma_closes": 2,
    "max_open": 4,
    "weekly_focus": 15,
    "take": 4,
    "rs_min_pctile": 85,
    "rmv_max": 45,
    "min_adv": 20_000_000,
    "min_adr_pct": 3.0,
    "qqq_gate": "close_above_rising_21ema",
    "timeframe": "1d_signal_15m_fill",
    "provider": "ALPACA",
    "fallback_provider": "IB",
    "merge_mode": "prefix",
}

SUMMARY_RESULTS: Dict[str, Any] = {
    "n_trades": 179,
    "expectancy": 2.25,
    "profit_factor": 1.85,
    "geo_pf": 0.985,
    "win_rate": 15.1,
    "median_gain_pct": -2.94,
    "fric025_expectancy": 2.00,
    "fric025_profit_factor": 1.70,
}

SUMMARY_NOTES: List[str] = [
    "No promote. Geo-PF < 1; 2022-23 E -2.16 PF 0.17; 83% hard_stop; median is the 3% stop.",
    "Do not grade vs clip books or 1d signal-close lookbacks.",
    "Number to beat for a different 1d live book remains last-15m n=2297 E +0.27 PF 1.09.",
    "HTML default friction 0.25 matches the net book; set friction to 0 for gross.",
    "Occupancy already applied in the CSV (max_open=4); leave Max entries/day at 0.",
]


def latest_weekly_trades(base: Path) -> Path:
    files = list(iter_report_files(base, "weekly_deepvue_trades.csv"))
    if not files:
        raise FileNotFoundError("No weekly_deepvue_trades.csv under %s" % base)

    def _key(path: Path) -> tuple:
        parent = path.parent.name
        day = parent if is_iso_date_dir_name(parent) else ""
        return (day, path.stat().st_mtime)

    return max(files, key=_key)


def write_weekly_summary(trades_path: Path) -> Path:
    path = trades_path.with_name("weekly_deepvue_trades_summary.txt")
    return write_summary_sidecar(
        path,
        title="Causal Deepvue-style weekend scan (not Deepvue; not a promote)",
        params=SUMMARY_PARAMS,
        results=SUMMARY_RESULTS,
        notes=SUMMARY_NOTES,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="TV-style HTML report for weekly Deepvue trades")
    ap.add_argument("--trades", type=Path, default=None)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Write HTML here (default: same folder as the trades CSV)",
    )
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--notional", type=float, default=10_000.0)
    ap.add_argument("--friction-pct", type=float, default=0.25)
    ap.add_argument("--provider", default="ALPACA")
    args = ap.parse_args()

    trades_path = args.trades or latest_weekly_trades(DEFAULT_TRADES_DIR)
    logger.info("Loading trades: %s", trades_path)
    df = ensure_report_trade_columns(pd.read_csv(trades_path))
    if df.empty:
        logger.error("Empty trades CSV")
        return 1

    summary_path = write_weekly_summary(trades_path)
    raw_trades = trades_to_raw(df)

    buy_min = pd.to_datetime(df["buy_date"]).min()
    sell_max = pd.to_datetime(df["sell_date"]).max()
    spy_closes: List[dict] = []
    try:
        spy_close = load_spy_close(
            buy_min - pd.Timedelta(days=5),
            sell_max + pd.Timedelta(days=5),
            provider=args.provider,
        )
        spy_closes = spy_to_raw(spy_close, buy_min, sell_max)
        logger.info(
            "SPY bars for compare: %d (%s to %s)",
            len(spy_closes),
            spy_closes[0]["x"] if spy_closes else "n/a",
            spy_closes[-1]["x"] if spy_closes else "n/a",
        )
    except Exception as exc:
        logger.warning("S&P 500 data skipped: %s", exc)

    defaults = {
        "capital": float(args.initial_capital),
        "sizeMode": "fixed",
        "sizeVal": float(args.notional),
        "friction": float(args.friction_pct),
        "maxPerDay": 0,
        "maxOpen": 0,
        "winCap": 0,
        "excludeSym": "",
    }

    extra_meta = {
        "friction_pct": float(args.friction_pct),
        "provider": args.provider,
        "n_trades_embedded": len(raw_trades),
        "book": "weekly_deepvue",
    }
    run_meta = build_run_meta(trades_path, extra=extra_meta, summary_path=summary_path)
    run_meta["detector"] = {**DETECTOR_META}

    outdir = Path(args.outdir) if args.outdir is not None else trades_path.parent
    outdir = dated_outdir(outdir)
    out_html = outdir / "weekly_deepvue.html"
    out_json = outdir / "weekly_deepvue.json"
    title = "Weekly Deepvue weekend scan — Strategy Report (not a promote)"

    html = render_html(
        raw_trades=raw_trades,
        spy_closes=spy_closes,
        defaults=defaults,
        run_meta=run_meta,
        title=title,
        source=str(trades_path.name),
        comparison=None,
    )
    out_html.write_text(html, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "defaults": defaults,
                "n_trades_embedded": len(raw_trades),
                "n_spy_bars": len(spy_closes),
                "source": str(trades_path),
                "run_meta": run_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s (%d trades embedded)", out_html, len(raw_trades))
    print("\nOpen: %s" % out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
