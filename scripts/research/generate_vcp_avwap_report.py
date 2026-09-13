#!/usr/bin/env python3
"""TradingView-style HTML report for the causal VCP AVWAP occupancy book.

Reuses generate_channel_touch_tv_report.render_html. Channel JSON is empty.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\generate_vcp_avwap_report.py
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
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, apply_friction, summarize_by_year
from scripts.research.generate_channel_touch_tv_report import (
    build_run_meta,
    ensure_report_trade_columns,
    load_spy_close,
    render_html,
    spy_to_raw,
    trades_to_raw,
    write_summary_sidecar,
)
from utils.research.last_15m_volume_geometry import book_stats, geom_mean_year_pf
from utils.research.report_paths import dated_outdir, is_iso_date_dir_name, iter_report_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_vcp_avwap_report")

DEFAULT_TRADES_DIR = ROOT / "reports" / "vcp_avwap"

DETECTOR_META: Dict[str, Any] = {
    "name": "VCP AVWAP breakout (causal 15m; not a promote)",
    "script": "scripts/research/backtest_vcp_avwap.py",
    "pine": "n/a (not a channel detector)",
    "version": "research 2026-09-13",
    "notes": (
        "Playbook #1 translation of strategies/vcp_avwap_breakout.py. "
        "Daily VCP 6-12w + near 52w + AVWAP from prior completed session. "
        "Buy first RTH 15m close above pivot and AVWAP with 1.5x 15m RVOL (signal-close). "
        "Sell ATR*1 hard stop or weekly close below 10w SMA, fill next 15m mid. "
        "No 1h universe; 15m close-cross is the 1h analog. No promote."
    ),
    "params": [],
}

SUMMARY_PARAMS: Dict[str, Any] = {
    "entry": "15m_close_cross_signal_close",
    "exit": "atr1_or_weekly_ma_15m_n1_mid",
    "atr_mult_stop": 1.0,
    "weekly_ma_period": 10,
    "breakout_rvol_min": 1.5,
    "near_52w_threshold": 0.92,
    "vcp_min_weeks": 6,
    "vcp_max_weeks": 12,
    "timeframe": "1d_arm_15m_fill",
    "provider": "ALPACA",
    "fallback_provider": "IB",
    "merge_mode": "prefix",
}


def latest_vcp_trades(base: Path) -> Path:
    files = list(iter_report_files(base, "vcp_avwap_trades.csv"))
    if not files:
        raise FileNotFoundError("No vcp_avwap_trades.csv under %s" % base)

    def _key(path: Path) -> tuple:
        parent = path.parent.name
        day = parent if is_iso_date_dir_name(parent) else ""
        return (day, path.stat().st_mtime)

    return max(files, key=_key)


def _results_from_trades(df: pd.DataFrame, friction_pct: float) -> Dict[str, Any]:
    g = pd.to_numeric(df.get("gain_pct"), errors="coerce") if not df.empty else pd.Series(dtype=float)
    s = book_stats(g)
    years = summarize_by_year(df, gain_col="gain_pct", buckets=YEAR_BUCKETS) if not df.empty else pd.DataFrame()
    geo = geom_mean_year_pf(years)
    net = apply_friction(df, friction_pct) if not df.empty else df
    ns = book_stats(pd.to_numeric(net.get("gain_pct_net"), errors="coerce")) if not df.empty else {}
    return {
        "n_trades": s.get("n"),
        "expectancy": s.get("expectancy_pct"),
        "profit_factor": s.get("profit_factor"),
        "geo_pf": None if geo is None else round(geo, 3),
        "win_rate": s.get("win_rate_pct"),
        "median_gain_pct": s.get("median_pct"),
        "fric025_expectancy": ns.get("expectancy_pct"),
        "fric025_profit_factor": ns.get("profit_factor"),
    }


def write_vcp_summary(trades_path: Path, results: Dict[str, Any]) -> Path:
    path = trades_path.with_name("vcp_avwap_trades_summary.txt")
    notes: List[str] = [
        "Research translation of VCP AVWAP. Realistic 15m signal-close buys and N+1 mid sells.",
        "Do not grade vs clip books or 1d signal-close lookbacks.",
        "Number to beat for a different 1d live book remains last-15m n=2297 E +0.27 PF 1.09.",
        "Occupancy is unique-symbol (max_open 0 = uncapped concurrent); leave Max entries/day at 0.",
    ]
    return write_summary_sidecar(
        path,
        title="Causal VCP AVWAP breakout (15m fills; not a promote)",
        params=SUMMARY_PARAMS,
        results=results,
        notes=notes,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="TV-style HTML report for VCP AVWAP trades")
    ap.add_argument("--trades", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--notional", type=float, default=10_000.0)
    ap.add_argument("--friction-pct", type=float, default=0.25)
    ap.add_argument("--provider", default="ALPACA")
    args = ap.parse_args()

    trades_path = args.trades or latest_vcp_trades(DEFAULT_TRADES_DIR)
    logger.info("Loading trades: %s", trades_path)
    df = ensure_report_trade_columns(pd.read_csv(trades_path))
    if df.empty:
        logger.error("Empty trades CSV")
        return 1

    results = _results_from_trades(df, float(args.friction_pct))
    summary_path = write_vcp_summary(trades_path, results)
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
        "book": "vcp_avwap",
    }
    run_meta = build_run_meta(trades_path, extra=extra_meta, summary_path=summary_path)
    run_meta["detector"] = {**DETECTOR_META}

    outdir = Path(args.outdir) if args.outdir is not None else trades_path.parent
    outdir = dated_outdir(outdir)
    out_html = outdir / "vcp_avwap.html"
    out_json = outdir / "vcp_avwap.json"
    title = "VCP AVWAP breakout — Strategy Report (not a promote)"

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
