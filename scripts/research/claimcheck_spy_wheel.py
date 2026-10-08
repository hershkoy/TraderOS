#!/usr/bin/env python3
"""Claim-check SPY Wheel + Covered Strangle (OptionsDX + Theta).

Requires TimescaleDB with OptionsDX SPY quotes (2010-2023). For 2024+ legs,
start Theta Terminal:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_spy_wheel.py --mode both --marks mid
  python scripts\\research\\claimcheck_spy_wheel.py --mode wheel --marks ask_bid
  python scripts\\research\\claimcheck_spy_wheel.py --mode both --end 2015-12-31
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.metrics import passes_phase5_gates, perf_stats  # noqa: E402
from utils.research.spy_wheel import (  # noqa: E402
    CLAIMED,
    HybridSource,
    ODX_END,
    ODX_START,
    ThetaSource,
    closed_to_frame,
    default_cs_params,
    default_wheel_params,
    run_book,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_spy_wheel")

DEFAULT_OUT = ROOT / "reports" / "research" / "spy_wheel"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _write_book(book, outdir: Path) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = book.label
    eq_path = outdir / f"{stem}_equity.csv"
    eq = book.equity.rename("equity").to_frame()
    eq["spy"] = book.spy_equity.reindex(eq.index)
    eq.to_csv(eq_path, index_label="date")
    trades = closed_to_frame(book.closed)
    trades_path = outdir / f"{stem}_trades.csv"
    trades.to_csv(trades_path, index=False)
    events_path = outdir / f"{stem}_events.json"
    events_path.write_text(json.dumps(book.events, indent=2), encoding="utf-8")
    stats = book.stats()
    spy_st = None
    if len(book.spy_equity.dropna()) >= 2:
        spy_st = perf_stats("SPY", book.spy_equity)
        st = perf_stats(
            book.label,
            book.equity,
            spy_cagr=spy_st.cagr,
            spy_sharpe=spy_st.sharpe,
            trade_rets=[c.pnl for c in book.closed],
            notes=book.notes,
        )
        ok, reason = passes_phase5_gates(st, spy_st)
        stats["phase5_ok"] = bool(ok)
        stats["phase5_reason"] = reason
        stats["spy"] = {
            "cagr": spy_st.cagr,
            "max_drawdown": spy_st.max_drawdown,
            "sharpe": spy_st.sharpe,
        }
    stats_path = outdir / f"{stem}_summary.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    LOG.info(
        "%s CAGR=%.2f%% MDD=%.1f%% Sharpe=%.2f closed=%s",
        book.label,
        100.0 * float(stats.get("cagr") or 0),
        100.0 * float(stats.get("max_drawdown") or 0),
        float(stats.get("sharpe") or 0),
        stats.get("n_closed_legs"),
    )
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["wheel", "cs", "both"], default="both")
    p.add_argument(
        "--marks",
        default="mid",
        help="mid | ask_bid | mid,ask_bid",
    )
    p.add_argument("--start", default=str(ODX_START))
    p.add_argument("--end", default="")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_OUT / "cache")
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--odx-only", action="store_true", help="Stop at OptionsDX end")
    args = p.parse_args(argv)

    start = _parse_date(args.start)
    if args.end:
        end = _parse_date(args.end)
    elif args.odx_only:
        end = ODX_END
    else:
        end = date.today()

    marks = [m.strip() for m in str(args.marks).split(",") if m.strip()]
    modes: List[str] = []
    if args.mode in ("wheel", "both"):
        modes.append("wheel")
    if args.mode in ("cs", "both"):
        modes.append("cs")

    theta = ThetaSource(cache_dir=args.cache_dir)
    source = HybridSource(theta=theta)
    if end > ODX_END:
        if not theta.client.is_ready():
            LOG.warning(
                "Theta Terminal not ready; clamping end to OptionsDX %s", ODX_END
            )
            end = ODX_END
        else:
            LOG.info("Theta ready; book may extend past %s", ODX_END)

    t0 = time.time()
    all_stats: List[Dict[str, Any]] = []
    for mode in modes:
        params = default_wheel_params() if mode == "wheel" else default_cs_params()
        for mark in marks:
            LOG.info("Running %s marks=%s %s -> %s", mode, mark, start, end)
            book = run_book(
                mode=params["mode"],
                start=start,
                end=end,
                source=source,
                mark_mode=mark,
                target_dte=int(params["target_dte"]),
                target_delta=float(params["target_delta"]),
                tp_frac=params["tp_frac"],
                cover_frac=float(params["cover_frac"]),
                use_sma_put_gate=bool(params["use_sma_put_gate"]),
                commission_per_contract=float(args.commission),
            )
            stats = _write_book(book, args.outdir)
            all_stats.append(stats)

    summary = {
        "claimed": CLAIMED,
        "window": {"start": str(start), "end": str(end)},
        "elapsed_sec": round(time.time() - t0, 1),
        "books": all_stats,
        "data_notes": [
            "OptionsDX SPY EOD 2010-01-04..2023-12-29 (listed delta)",
            "Theta greeks/eod for 2024+ when Terminal up (pre-2012 hist 403)",
            "Video claims 2007 start; local book cannot cover 2007-2009",
            "No early assignment model; dividends not credited",
        ],
    }
    out_sum = args.outdir / "summary.json"
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_sum.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %s (%.1fs)", out_sum, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
