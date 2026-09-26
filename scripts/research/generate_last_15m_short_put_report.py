#!/usr/bin/env python3
"""TradingView-style HTML report for last-15m short-put BS (cash-secured %).

Reuses generate_channel_touch_tv_report.render_html (same dark UI as
reports/ascending_channels/current_best). Embeds the ATM IVx1 put book;
comparison table covers stock baseline, IV stress, delta book, Polygon spot.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\generate_last_15m_short_put_report.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.generate_channel_touch_tv_report import (  # noqa: E402
    build_run_meta,
    ensure_report_trade_columns,
    load_spy_close,
    render_html,
    spy_to_raw,
    trades_to_raw,
    write_summary_sidecar,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_last_15m_short_put_report")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-26"
    / "last_15m_short_put_bs_dte45_atm_trades_20260926_213902.csv"
)
DEFAULT_SPOT = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-26"
    / "last_15m_short_put_polygon_spot_20260926_223455.csv"
)

DETECTOR_META: Dict[str, Any] = {
    "name": "Last-15m H2 short put (Black-Scholes research)",
    "script": "scripts/research/overlay_last_15m_short_put_bs.py",
    "pine": "n/a (options overlay on channel-touch signal)",
    "version": "research 2026-09-26",
    "notes": (
        "Same signal as current_best 1d last-15m (open above daily resist). "
        "Sell ATM cash-secured put, DTE 45 (weekday snap), TP at 50% of credit, "
        "stop when underlying tags ATR k=2 hard-stop (constant-IV BS mark). "
        "Primary P&L is cash-secured % = (credit-exit)/K. Occupancy not re-walked. "
        "No promote — see IV-stress and Polygon spot-check in comparison."
    ),
    "params": [
        {"key": "strike_mode", "value": "atm"},
        {"key": "dte", "value": "45"},
        {"key": "tp_frac", "value": "0.50"},
        {"key": "iv", "value": "atr_pct annualized (constant)"},
        {"key": "stop", "value": "underlying ATR k=2 (1.5%-6%)"},
        {"key": "capital", "value": "cash-secured (gain_pct on strike)"},
    ],
}

SUMMARY_PARAMS: Dict[str, Any] = {
    "signal": "last_15m_open_above_resist",
    "instrument": "short_put_atm",
    "dte": 45,
    "tp_frac": 0.50,
    "strike_mode": "atm",
    "iv": "atr_annualized_constant",
    "stop": "underlying_atr_k2",
    "pnl": "cash_secured_pct",
    "occupancy": "not_rewalked",
    "timeframe": "1d_signal_put_overlay",
}

SUMMARY_RESULTS: Dict[str, Any] = {
    "n_trades": 2297,
    "expectancy": 0.51,
    "profit_factor": 1.51,
    "geo_pf": 1.70,
    "win_rate": 48.54,
    "median_gain_pct": None,
    "e_2022": 0.11,
    "pf_2022": 1.08,
    "stock_baseline_e": 0.27,
    "stock_baseline_pf": 1.09,
    "stop_iv_x125_e": -0.27,
    "stop_iv_x125_pf": 0.85,
    "polygon_spot_n_ok": 9,
    "polygon_median_exit_vs_bs": 0.85,
}

SUMMARY_NOTES: List[str] = [
    "No promote. Constant-IV BS looks better than stock; stop-IV x1.25 kills the book.",
    "Polygon free spot (9/12 hard-stops): median live_exit/bs_exit=0.85 — not the x1.25 kill case.",
    "Day-bar close is not the buyback ask; entries also often print under BS credit.",
    "HTML P&L uses cash-secured %; Entry/Exit px show put credit / exit premium.",
    "Do not wire into /hot or current_best live slots.",
    "Number to beat for the stock sleeve remains last-15m n=2297 E +0.27 PF 1.09.",
]


def prepare_put_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Map put overlay columns onto the channel-touch report schema."""
    t = df.copy()
    if "put_ok" in t.columns:
        t = t[t["put_ok"] == True].copy()  # noqa: E712
    if t.empty:
        return t
    if "put_exit_date" in t.columns:
        t["sell_date"] = t["put_exit_date"]
    if "put_hold_days" in t.columns:
        t["hold_days"] = t["put_hold_days"]
    if "put_exit_reason" in t.columns:
        t["exit_reason"] = t["put_exit_reason"]
    # Display option marks; equity still driven by gain_pct (cash-secured %).
    if "put_credit" in t.columns:
        t["buy_price"] = pd.to_numeric(t["put_credit"], errors="coerce")
    if "put_exit_premium" in t.columns:
        t["sell_price"] = pd.to_numeric(t["put_exit_premium"], errors="coerce")
    # Drop stock 15m clocks so the table uses put calendar dates.
    for col in ("buy_time", "sell_time"):
        if col in t.columns:
            t = t.drop(columns=[col])
    t = t.dropna(subset=["buy_date", "sell_date", "buy_price", "sell_price", "gain_pct"])
    return t.reset_index(drop=True)


def build_comparison(spot_path: Optional[Path]) -> Dict[str, Any]:
    rows = [
        {
            "book": "stock last-15m (baseline)",
            "n": 2297,
            "wr": 31.4,
            "e": 0.27,
            "pf": 1.09,
        },
        {
            "book": "put ATM 45d IV×1 (embedded)",
            "n": 2297,
            "wr": 48.5,
            "e": 0.51,
            "pf": 1.51,
            "highlight": True,
        },
        {
            "book": "put Δ−0.30 45d IV×1",
            "n": 2297,
            "wr": 54.0,
            "e": 0.38,
            "pf": 1.62,
        },
        {
            "book": "put ATM stop-IV×1.25",
            "n": 2297,
            "wr": 48.1,
            "e": -0.27,
            "pf": 0.85,
        },
        {
            "book": "put ATM stop-IV×1.5",
            "n": 2297,
            "wr": 47.9,
            "e": -1.05,
            "pf": 0.59,
        },
    ]
    note = (
        "Cash-secured % for puts; stock row is share %. "
        "IV×1.25+ stress marks hard-stop exits only. Not a promote."
    )
    if spot_path is not None and spot_path.exists():
        spot = pd.read_csv(spot_path)
        ok = spot[spot["status"].astype(str) == "ok"]
        if not ok.empty and "exit_vs_bs" in ok.columns and "live_gain_cs" in ok.columns:
            rows.append(
                {
                    "book": "Polygon spot hard-stops (live CS)",
                    "n": int(len(ok)),
                    "wr": round(float((pd.to_numeric(ok["live_gain_cs"], errors="coerce") > 0).mean() * 100), 1),
                    "e": round(float(pd.to_numeric(ok["live_gain_cs"], errors="coerce").mean()), 2),
                    "pf": None,
                }
            )
            med = float(pd.to_numeric(ok["exit_vs_bs"], errors="coerce").median())
            note += " Polygon free spot n=%d median live_exit/bs_exit=%.2f." % (len(ok), med)
    return {
        "title": "Stock vs short-put BS / IV stress / Polygon spot",
        "note": note,
        "rows": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--spot", type=Path, default=DEFAULT_SPOT)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--notional", type=float, default=10_000.0)
    ap.add_argument(
        "--friction-pct",
        type=float,
        default=0.0,
        help="Default 0: put marks already synthetic; stock reports often use 0.25",
    )
    ap.add_argument("--provider", default="ALPACA")
    args = ap.parse_args(argv)

    trades_path = Path(args.trades)
    logger.info("Loading put trades: %s", trades_path)
    raw = pd.read_csv(trades_path)
    df = prepare_put_trades(raw)
    df = ensure_report_trade_columns(df)
    if df.empty:
        logger.error("No put_ok trades after prepare")
        return 1
    logger.info("Prepared n=%d put trades", len(df))

    outdir = Path(args.outdir) if args.outdir is not None else trades_path.parent
    outdir = dated_outdir(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "last_15m_short_put_bs_summary.txt"
    write_summary_sidecar(
        summary_path,
        title="Last-15m short put BS ATM 45d (cash-secured; not a promote)",
        params=SUMMARY_PARAMS,
        results=SUMMARY_RESULTS,
        notes=SUMMARY_NOTES,
    )

    comparison = build_comparison(Path(args.spot) if args.spot else None)
    comparison_path = outdir / "last_15m_short_put_bs_compare.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

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
        logger.info("SPY bars: %d", len(spy_closes))
    except Exception as exc:  # noqa: BLE001
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
        "book": "last_15m_short_put_bs_atm45",
        "pnl_unit": "cash_secured_pct",
    }
    run_meta = build_run_meta(trades_path, extra=extra_meta, summary_path=summary_path)
    run_meta["detector"] = {**DETECTOR_META}
    run_meta["results"] = {**SUMMARY_RESULTS}

    out_html = outdir / "1d_last_15m_short_put_bs.html"
    out_json = outdir / "1d_last_15m_short_put_bs.json"
    title = "Last-15m short put (BS ATM 45d) — Strategy Report (not a promote)"

    html = render_html(
        raw_trades=raw_trades,
        spy_closes=spy_closes,
        defaults=defaults,
        run_meta=run_meta,
        title=title,
        source=str(trades_path.name),
        comparison=comparison,
    )
    out_html.write_text(html, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "defaults": defaults,
                "n_trades_embedded": len(raw_trades),
                "n_spy_bars": len(spy_closes),
                "source": str(trades_path),
                "spot": str(args.spot) if args.spot else None,
                "run_meta": run_meta,
                "comparison": comparison,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s (%d trades)", out_html, len(raw_trades))
    print("\nOpen: %s" % out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
