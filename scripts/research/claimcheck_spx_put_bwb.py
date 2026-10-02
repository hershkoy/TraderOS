#!/usr/bin/env python3
"""Claim-check Carl Allen 21 DTE SPX put Broken Wing Butterfly.

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_spx_put_bwb.py
  python scripts\\research\\claimcheck_spx_put_bwb.py --marks mid --down-day
  python scripts\\research\\claimcheck_spx_put_bwb.py --marks mid_slip0.15 --trend-ma 50
  python scripts\\research\\claimcheck_spx_put_bwb.py --variants friction
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.spx_put_bwb import (  # noqa: E402
    CLAIMED,
    book_stats,
    load_index_eod,
    parse_package_slip,
    run_bwb_book,
    trades_to_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_spx_put_bwb")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_OUT = ROOT / "reports" / "research" / "spx_put_bwb"

# Shared-cache multi-book suites (skip ask_bid — already shown to be COB-unrealistic)
VARIANT_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "friction": [
        {"marks": "mid", "use_stop": True, "down_day": False, "package_slip": 0.0, "trend_ma": None},
        {"marks": "mid", "use_stop": True, "down_day": True, "package_slip": 0.0, "trend_ma": None},
        {"marks": "mid", "use_stop": False, "down_day": False, "package_slip": 0.0, "trend_ma": None},
        {"marks": "mid", "use_stop": True, "down_day": False, "package_slip": 0.15, "trend_ma": None},
        {"marks": "mid", "use_stop": True, "down_day": False, "package_slip": 0.10, "trend_ma": None},
        {"marks": "mid", "use_stop": True, "down_day": False, "package_slip": 0.0, "trend_ma": 50},
        {"marks": "mid", "use_stop": True, "down_day": False, "package_slip": 0.0, "trend_ma": 200},
        {"marks": "mid", "use_stop": True, "down_day": True, "package_slip": 0.15, "trend_ma": 50},
    ],
    # Daily scan; entries only when down-day ∩ SPX>SMA50 (filters in book)
    "daily_regime": [
        {
            "marks": "mid",
            "use_stop": True,
            "stop_mode": "debit",
            "down_day": True,
            "package_slip": 0.0,
            "trend_ma": 50,
            "scan": "daily",
        },
        {
            "marks": "mid",
            "use_stop": True,
            "stop_mode": "debit",
            "down_day": True,
            "package_slip": 0.15,
            "trend_ma": 50,
            "scan": "daily",
        },
        {
            "marks": "mid",
            "use_stop": True,
            "stop_mode": "spot",
            "down_day": True,
            "package_slip": 0.0,
            "trend_ma": 50,
            "scan": "daily",
        },
        {
            "marks": "mid",
            "use_stop": True,
            "stop_mode": "spot",
            "down_day": True,
            "package_slip": 0.15,
            "trend_ma": 50,
            "scan": "daily",
        },
        {
            "marks": "mid",
            "use_stop": True,
            "stop_mode": "spot",
            "spot_stop_buffer": 5.0,
            "down_day": True,
            "package_slip": 0.15,
            "trend_ma": 50,
            "scan": "daily",
        },
        # Monday control: same filters + spot stop for head-to-head
        {
            "marks": "mid",
            "use_stop": True,
            "stop_mode": "spot",
            "down_day": True,
            "package_slip": 0.15,
            "trend_ma": 50,
            "scan": "monday",
        },
    ],
}


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt(label: str, s: Dict[str, Any]) -> str:
    return (
        f"{label}: n={s.get('n')} WR={s.get('win_rate_pct')}% "
        f"meanRoR={s.get('mean_ror_pct')}% medRoR={s.get('med_ror_pct')}% "
        f"PF={s.get('pf')} PnL=${s.get('total_pnl')} "
        f"Sharpe~={s.get('sharpe_proxy')} avgHold={s.get('avg_hold')}d "
        f"avgCredit={s.get('avg_credit')} exits={s.get('exit_reasons')}"
    )


def _tag(
    mark: str,
    *,
    use_stop: bool,
    stop_mode: str,
    down_day: bool,
    package_slip: float,
    trend_ma: Optional[int],
    scan: str,
    spot_stop_buffer: float = 0.0,
) -> str:
    slip = parse_package_slip(mark, package_slip)
    base = mark
    if slip > 0 and not str(mark).startswith("mid_slip"):
        base = f"mid_slip{slip:g}"
    smode = str(stop_mode or "debit").lower()
    if not use_stop:
        smode = "none"
    if smode == "spot":
        stop_part = "spotsl" + (f"p{int(spot_stop_buffer)}" if spot_stop_buffer else "")
    elif smode == "none":
        stop_part = "nosl"
    else:
        stop_part = "sl"
    parts = [base, stop_part]
    if down_day:
        parts.append("downday")
    if trend_ma:
        parts.append(f"ma{int(trend_ma)}")
    if scan and scan != "monday":
        parts.append(scan)
    return "_".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-08-29")
    ap.add_argument("--underlier", default="SPX")
    ap.add_argument("--opt-symbol", default="SPXW", help="SPXW weeklies (21 DTE) or SPX monthly")
    ap.add_argument("--marks", default="mid,ask_bid")
    ap.add_argument("--narrow", type=float, default=25.0)
    ap.add_argument("--wide", type=float, default=50.0)
    ap.add_argument("--dte", type=int, default=21)
    ap.add_argument("--dte-tol", type=int, default=4)
    ap.add_argument("--min-dte-exit", type=int, default=7)
    ap.add_argument("--credit-lo", type=float, default=0.12)
    ap.add_argument("--credit-hi", type=float, default=0.15)
    ap.add_argument("--scan", default="monday", choices=["monday", "daily"])
    ap.add_argument("--down-day", action="store_true", help="Only enter on down closes")
    ap.add_argument("--no-stop", action="store_true", help="Disable 3x-credit stop (claim ~80% WR)")
    ap.add_argument(
        "--package-slip",
        type=float,
        default=0.0,
        help="COB-style mid package slip (open -slip, close +slip); also mid_slip0.15 mark",
    )
    ap.add_argument(
        "--trend-ma",
        type=int,
        default=0,
        help="Require spot > SMA(N) at entry (50 or 200 typical)",
    )
    ap.add_argument(
        "--stop-mode",
        default="debit",
        choices=["debit", "spot", "none"],
        help="debit=3x credit package; spot=underlying close <= short_k (+buffer)",
    )
    ap.add_argument(
        "--spot-stop-buffer",
        type=float,
        default=0.0,
        help="With --stop-mode spot: trigger at short_k + buffer (e.g. 5)",
    )
    ap.add_argument(
        "--variants",
        default="",
        help="Named suite (friction|daily_regime) — shared Theta cache across books",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    if args.variants:
        suite = VARIANT_SUITES.get(args.variants.strip().lower())
        if not suite:
            LOG.error("Unknown --variants %s; choose from %s", args.variants, list(VARIANT_SUITES))
            return 1
        configs = suite
    else:
        marks = [m.strip() for m in args.marks.split(",") if m.strip()]
        stop_mode = "none" if args.no_stop else args.stop_mode
        configs = [
            {
                "marks": m,
                "use_stop": stop_mode != "none",
                "stop_mode": stop_mode,
                "spot_stop_buffer": args.spot_stop_buffer,
                "down_day": args.down_day,
                "package_slip": args.package_slip,
                "trend_ma": args.trend_ma or None,
                "scan": args.scan,
            }
            for m in marks
        ]

    outdir = args.outdir or (DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d"))
    outdir.mkdir(parents=True, exist_ok=True)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not ready at %s", client.base_url)
        return 1

    # Warm shared spot + expirations + quote cache across variants
    need_ma = max((c.get("trend_ma") or 0) for c in configs)
    warm_start = start - timedelta(days=max(40, (need_ma + 40) * 2))
    LOG.info("Loading spot %s %s..%s", args.underlier, warm_start, end)
    spots = load_index_eod(client, args.underlier, warm_start, end)
    LOG.info("Loading %s expirations", args.opt_symbol)
    exps = list_expirations(client, args.opt_symbol)
    cache: Dict[Tuple[Any, ...], Any] = {}
    strike_cache: Dict[date, List[float]] = {}

    t0 = time.time()
    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "underlier": args.underlier,
        "opt_symbol": args.opt_symbol,
        "structure": {"narrow": args.narrow, "wide": args.wide, "dte": args.dte},
        "variants": args.variants or None,
        "books": {},
        "wall_s": None,
        "followups": {
            "live_cob_fill_profile": (
                "TODO later: paper-trade 1-lot SPXW BWB combos and log mid vs fill "
                "to calibrate package_slip (not run in this claim-check)."
            ),
            "intraday_stop_risk": (
                "EOD marks undercount intraday 3x-debit stops; spot-stop is the "
                "structure proxy until 1m path walks exist."
            ),
        },
    }

    for cfg in configs:
        mark = str(cfg["marks"])
        use_stop = bool(cfg.get("use_stop", True))
        stop_mode = str(cfg.get("stop_mode") or ("debit" if use_stop else "none"))
        spot_buf = float(cfg.get("spot_stop_buffer") or 0.0)
        down_day = bool(cfg.get("down_day", False))
        slip = float(cfg.get("package_slip") or 0.0)
        trend_ma = cfg.get("trend_ma")
        trend_ma = int(trend_ma) if trend_ma else None
        scan = str(cfg.get("scan") or args.scan)
        tag = _tag(
            mark,
            use_stop=use_stop,
            stop_mode=stop_mode,
            down_day=down_day,
            package_slip=slip,
            trend_ma=trend_ma,
            scan=scan,
            spot_stop_buffer=spot_buf,
        )
        LOG.info("=== book %s ===", tag)
        book = run_bwb_book(
            start=start,
            end=end,
            underlier=args.underlier,
            opt_symbol=args.opt_symbol,
            mark_mode=mark,
            narrow=args.narrow,
            wide=args.wide,
            target_dte=args.dte,
            dte_tol=args.dte_tol,
            min_dte_exit=args.min_dte_exit,
            credit_lo=args.credit_lo,
            credit_hi=args.credit_hi,
            use_stop=use_stop and stop_mode != "none",
            stop_mode=stop_mode,
            spot_stop_buffer=spot_buf,
            down_day_only=down_day,
            trend_ma=trend_ma,
            package_slip=slip,
            scan=scan,
            client=client,
            spots=spots,
            exps=exps,
            cache=cache,
            strike_cache=strike_cache,
        )
        stats = book_stats(book.trades)
        summary["books"][tag] = {
            "stats": stats,
            "n_skipped": len(book.skipped),
            "skip_reasons": _count_reasons(book.skipped),
            "label": book.label,
            "config": {
                "mark": mark,
                "use_stop": use_stop,
                "stop_mode": stop_mode,
                "spot_stop_buffer": spot_buf,
                "down_day": down_day,
                "package_slip": slip,
                "trend_ma": trend_ma,
                "scan": scan,
            },
        }
        LOG.info(_fmt(tag, stats))
        stem = f"{args.opt_symbol}_{int(args.narrow)}x{int(args.wide)}_{tag}"
        trades_to_frame(book.trades).to_csv(outdir / f"{stem}_trades.csv", index=False)
        with (outdir / f"{stem}_skipped.json").open("w", encoding="utf-8") as f:
            json.dump(book.skipped[:5000], f, indent=2, default=str)

    summary["wall_s"] = round(time.time() - t0, 1)
    if args.variants == "friction":
        out_name = "summary_friction.json"
    elif args.variants == "daily_regime":
        out_name = "summary_daily_regime.json"
    else:
        out_name = "summary.json"
    with (outdir / out_name).open("w", encoding="utf-8") as f:
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
    print("\n=== Carl Allen SPX put BWB vs claim ===")
    print(
        "Claim: ~80% WR no-stop / ~70% with 2x-credit stop; "
        "10-13% of narrow-wing risk in ~2 weeks; 21 DTE 25x50."
    )
    print(
        "Ours: Theta EOD SPXW; TP $0.50; stop=debit 3x or spot<=short_k; "
        "optional mid+/-package slip (COB proxy); $0.65/leg RT."
    )
    for tag, body in summary.get("books", {}).items():
        s = body.get("stats") or {}
        print(_fmt(tag, s), f"| skips={body.get('n_skipped')}")
    print(f"wall_s={summary.get('wall_s')}")


if __name__ == "__main__":
    raise SystemExit(main())
