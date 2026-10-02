#!/usr/bin/env python3
"""Claim-check Volatility Vibes Forward Factor calendars (v2 harness).

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_forward_factor.py --dte-front 30 --dte-back 60 --marks mid_slip --ff-mins none,0.20
  python scripts\\research\\claimcheck_forward_factor.py --structure double_cal --symbols SPY --marks mid_slip --ff-mins 0.20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import (  # noqa: E402
    CLAIMED,
    book_stats,
    load_earnings_map,
    run_ff_book,
    run_ff_panel,
    trades_to_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_forward_factor")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "forward_factor"
DEFAULT_PANEL = "SPY,QQQ,AAPL,TSLA,NVDA,AMD,META,AMZN,GOOGL,NFLX,BA,AES,COIN,PLTR"
EARNINGS_CACHE = DEFAULT_OUT / "earnings_cache.json"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt(label: str, s: Dict[str, Any]) -> str:
    return (
        f"{label}: n={s.get('n')} WR={s.get('win_rate_pct')}% "
        f"meanRet={s.get('mean_ret_pct')}% medRet={s.get('med_ret_pct')}% "
        f"PF={s.get('pf')} PnL=${s.get('total_pnl')} "
        f"avgFF={s.get('avg_ff')} Sharpe~={s.get('sharpe_proxy')} "
        f"avgHold={s.get('avg_hold')}d exits={s.get('exit_reasons')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-08-29")
    ap.add_argument("--symbols", default="SPY")
    ap.add_argument("--marks", default="mid_slip")
    ap.add_argument("--ff-mins", default="none,0.20")
    ap.add_argument("--dte-front", type=int, default=30)
    ap.add_argument("--dte-back", type=int, default=60)
    ap.add_argument("--dte-tol", type=int, default=10)
    ap.add_argument("--structure", default="atm_call", choices=["atm_call", "double_cal"])
    ap.add_argument("--slip-frac", type=float, default=0.10)
    ap.add_argument("--tp", type=float, default=0.25)
    ap.add_argument("--sl", type=float, default=0.40)
    ap.add_argument("--ff-exit", type=float, default=0.05)
    ap.add_argument("--time-stop-dte", type=int, default=15)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--scan", default="monday", choices=["monday", "month_start"])
    ap.add_argument("--monthly-only", action="store_true")
    ap.add_argument("--no-friday-only", action="store_true")
    ap.add_argument("--no-earnings-skip", action="store_true")
    ap.add_argument("--no-dynamic-exit", action="store_true")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    marks = [m.strip() for m in args.marks.split(",") if m.strip()]
    ff_mins: List[float] = []
    for x in args.ff_mins.split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x in ("none", "all", "unfiltered"):
            ff_mins.append(-999.0)
        else:
            ff_mins.append(float(x))

    symbols = (
        [s.strip().upper() for s in DEFAULT_PANEL.split(",") if s.strip()]
        if args.panel
        else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    )

    outdir = args.outdir or (DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d_v2"))
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not ready at %s", client.base_url)
        return 1

    skip_earnings = not args.no_earnings_skip
    earnings_map = None
    if skip_earnings:
        earnings_map = load_earnings_map(symbols, cache_path=EARNINGS_CACHE)

    t0 = time.time()
    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "harness": "v2",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": symbols,
        "structure": args.structure,
        "dte": {"front": args.dte_front, "back": args.dte_back, "tol": args.dte_tol},
        "exits": {
            "tp": args.tp,
            "sl": args.sl,
            "ff_exit": args.ff_exit,
            "time_stop_dte": args.time_stop_dte,
            "dynamic": not args.no_dynamic_exit,
        },
        "slip_frac": args.slip_frac,
        "friday_only": not args.no_friday_only,
        "monthly_only": args.monthly_only,
        "skip_earnings": skip_earnings,
        "books": {},
        "wall_s": None,
    }

    common_kw = dict(
        mark_mode="",  # filled below
        ff_min=0.0,
        dte_front=args.dte_front,
        dte_back=args.dte_back,
        dte_tol=args.dte_tol,
        client=client,
        scan=args.scan,
        friday_only=not args.no_friday_only,
        monthly_only=args.monthly_only,
        structure=args.structure,
        slip_frac=args.slip_frac,
        tp_pct=args.tp,
        sl_pct=args.sl,
        ff_exit=args.ff_exit,
        time_stop_dte=args.time_stop_dte,
        skip_earnings=skip_earnings,
        earnings_map=earnings_map,
        dynamic_exit=not args.no_dynamic_exit,
    )

    for mark in marks:
        for ff_min in ff_mins:
            tag = f"{args.structure}_{mark}_ff{ff_min:g}"
            LOG.info("=== book %s symbols=%s ===", tag, ",".join(symbols))
            kw = dict(common_kw)
            kw["mark_mode"] = mark
            kw["ff_min"] = ff_min
            if len(symbols) == 1:
                book = run_ff_book(start=start, end=end, symbol=symbols[0], **kw)
            else:
                book = run_ff_panel(symbols=symbols, start=start, end=end, **kw)
            stats = book_stats(book.trades)
            summary["books"][tag] = {
                "stats": stats,
                "n_skipped": len(book.skipped),
                "skip_reasons": _count_reasons(book.skipped),
            }
            LOG.info(_fmt(tag, stats))

            stem = (
                f"{'_'.join(symbols[:3])}{'_panel' if len(symbols) > 1 else ''}"
                f"_{args.dte_front}_{args.dte_back}_{tag}"
            )
            trades_to_frame(book.trades).to_csv(outdir / f"{stem}_trades.csv", index=False)
            with (outdir / f"{stem}_skipped.json").open("w", encoding="utf-8") as f:
                json.dump(book.skipped[:5000], f, indent=2, default=str)

    summary["wall_s"] = round(time.time() - t0, 1)
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOG.info("Wrote %s (wall %.1fs)", outdir, summary["wall_s"])
    _print_verdict(summary)
    return 0


def _count_reasons(skipped: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in skipped:
        r = str(s.get("reason") or "unknown")
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _print_verdict(summary: Dict[str, Any]) -> None:
    print("\n=== Forward Factor v2 vs claim ===")
    print("Claim: ATM 60/90 quarter-Kelly ~26.7% CAGR / Sharpe 2.4")
    print(
        "Ours v2: Friday DTE pairs; earnings skip; dynamic exits; "
        f"structure={summary.get('structure')}; mid_slip."
    )
    for tag, body in summary.get("books", {}).items():
        print(_fmt(tag, body.get("stats") or {}), f"| skips={body.get('n_skipped')}")
    print(f"wall_s={summary.get('wall_s')}")


if __name__ == "__main__":
    raise SystemExit(main())
