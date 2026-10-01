#!/usr/bin/env python3
"""Claim-check Amy Meissner TimeZone with ThetaData RUTW listed marks (v2).

v2 proxy: mid+/-slip fills, 1m NBBO entry/exit, IV term-structure gate,
basic +/-2% adjustments. Exact workshop recipe remains unknown.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_amy_timezone.py --window pilot --marks mid_slip
  python scripts\\research\\claimcheck_amy_timezone.py --window 2025 --marks mid,mid_slip,mid_slip10,ask_bid
  python scripts\\research\\claimcheck_amy_timezone.py --window 2025 --marks mid_slip --no-adjust
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, time as dtime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.amy_timezone_proxy import (  # noqa: E402
    CLAIMED,
    DEFAULT_CAPITAL,
    DEFAULT_IV_BACKWARD_MAX,
    DEFAULT_SLIP,
    book_stats,
    run_proxy_book,
    trades_to_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_amy_timezone")

DEFAULT_OUT = ROOT / "reports" / "research" / "amy_timezone"


def _windows(name: str) -> List[Dict[str, Any]]:
    if name == "2025":
        return [
            {
                "key": "2025_jan_aug_weekly",
                "start": date(2025, 1, 2),
                "end": date(2025, 8, 28),
                "cadence": "weekly",
                "claim": CLAIMED["windows"]["2025_jan_aug_weekly"],
            }
        ]
    if name == "3y_monthly":
        return [
            {
                "key": "last_3y_1wk_mo",
                "start": date(2022, 9, 1),
                "end": date(2025, 8, 28),
                "cadence": "monthly",
                "claim": CLAIMED["windows"]["last_3y_1wk_mo"],
            }
        ]
    if name == "both":
        return _windows("2025") + _windows("3y_monthly")
    if name == "pilot":
        return [
            {
                "key": "pilot_2025_q1",
                "start": date(2025, 1, 2),
                "end": date(2025, 3, 31),
                "cadence": "weekly",
                "claim": CLAIMED["windows"]["2025_jan_aug_weekly"],
            }
        ]
    raise ValueError(f"unknown window={name}")


def _parse_clock(s: str) -> dtime:
    parts = str(s).strip().split(":")
    return dtime(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", default="pilot", choices=["pilot", "2025", "3y_monthly", "both"])
    ap.add_argument(
        "--marks",
        default="mid_slip",
        help="comma list: mid, mid_slip, mid_slip10, ask_bid",
    )
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--lots", type=int, default=2)
    ap.add_argument("--slip", type=float, default=DEFAULT_SLIP, help="used when mark=mid_slip")
    ap.add_argument("--no-1m", action="store_true", help="EOD path only")
    ap.add_argument("--no-adjust", action="store_true", help="Disable +/-2%% adjust rolls")
    ap.add_argument("--no-iv-filter", action="store_true")
    ap.add_argument("--iv-backward-max", type=float, default=DEFAULT_IV_BACKWARD_MAX)
    ap.add_argument("--entry-start", default="14:00")
    ap.add_argument("--entry-end", default="15:45")
    ap.add_argument("--no-clamp-sl", action="store_true")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT / date.today().isoformat()
    outdir.mkdir(parents=True, exist_ok=True)
    marks = [m.strip() for m in str(args.marks).split(",") if m.strip()]
    use_1m = not bool(args.no_1m)
    allow_adjust = not bool(args.no_adjust)
    iv_filter = not bool(args.no_iv_filter)
    clamp_sl = not bool(args.no_clamp_sl)
    entry_start = _parse_clock(args.entry_start)
    entry_end = _parse_clock(args.entry_end)

    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "proxy": {
            "version": 2,
            "note": "1m NBBO path; mid+/-slip clamped to NBBO; IV front-back gate; "
            "+/-2% put/calendar rolls (max 3); ATM call calendar + ~3% OTM bull put",
            "capital": args.capital,
            "lots": args.lots,
            "use_1m": use_1m,
            "allow_adjust": allow_adjust,
            "iv_filter": iv_filter,
            "iv_backward_max": args.iv_backward_max,
            "entry_start": args.entry_start,
            "entry_end": args.entry_end,
            "clamp_sl": clamp_sl,
            "slip_default": args.slip,
        },
        "books": {},
    }

    t0 = time.perf_counter()
    for win in _windows(args.window):
        for mm in marks:
            LOG.info("=== %s mark=%s cadence=%s 1m=%s adj=%s ===",
                     win["key"], mm, win["cadence"], use_1m, allow_adjust)
            book = run_proxy_book(
                start=win["start"],
                end=win["end"],
                cadence=win["cadence"],
                mark_mode=mm,
                capital=float(args.capital),
                lots=int(args.lots),
                slip=float(args.slip),
                use_1m=use_1m,
                allow_adjust=allow_adjust,
                iv_filter=iv_filter,
                iv_backward_max=float(args.iv_backward_max),
                entry_start=entry_start,
                entry_end=entry_end,
                clamp_sl=clamp_sl,
            )
            stats = book_stats(book.trades, capital=float(args.capital))
            key = f"{win['key']}__{mm}"
            summary["books"][key] = {
                "window": {
                    "start": win["start"].isoformat(),
                    "end": win["end"].isoformat(),
                    "cadence": win["cadence"],
                },
                "claim": win["claim"],
                "stats": stats,
                "n_skipped": len(book.skipped),
                "skip_reasons": _count_skips(book.skipped),
                "mode": book.mode,
            }
            df = trades_to_frame(book.trades)
            (outdir / f"{key}_trades.csv").write_text("")  # touch
            df.to_csv(outdir / f"{key}_trades.csv", index=False)
            (outdir / f"{key}_skipped.json").write_text(
                json.dumps(book.skipped, indent=2, default=str), encoding="utf-8"
            )
            LOG.info(
                "RESULT %s n=%s wr=%s pf=%s pnl=$%.0f ret=%.1f%% adj_avg=%s skipped=%d",
                key,
                stats["n"],
                None if stats["wr"] is None else f"{100*stats['wr']:.1f}%",
                stats["pf"],
                stats["pnl"],
                stats["ret_pct_capital"],
                stats.get("avg_adjusts"),
                len(book.skipped),
            )

    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    out_json = outdir / "summary_v2.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOG.info("Wrote %s (%.1fs)", out_json, summary["wall_s"])
    _print_verdict(summary)
    return 0


def _count_skips(skipped: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in skipped:
        r = str(s.get("reason") or "unknown")
        out[r] = out.get(r, 0) + 1
    return out


def _print_verdict(summary: Dict[str, Any]) -> None:
    print("\n=== TimeZone proxy v2 vs claimed ===")
    for key, book in summary["books"].items():
        st = book["stats"]
        cl = book["claim"]
        print(
            f"{key}: n={st['n']} (claim {cl.get('n','?')}) "
            f"WR={None if st['wr'] is None else round(100*st['wr'],1)}% "
            f"(claim {100*cl.get('wr',0):.0f}%) "
            f"PF={st['pf']} (claim {cl.get('pf','?')}) "
            f"ret={st['ret_pct_capital']:.1f}% (claim {cl.get('ret_pct_capital','?')}%) "
            f"avgAdj={st.get('avg_adjusts')} avgPnL=${st['avg_pnl']}"
        )
    print("NOTE: still a transparent proxy - workshop strike/adjust recipe unknown.")


if __name__ == "__main__":
    raise SystemExit(main())
