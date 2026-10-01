#!/usr/bin/env python3
"""ThetaData NBBO RTH-open credit spot-check vs Polygon minute-print PCS pilot.

Requires Theta Terminal v3 running:
  python scripts\\pipeline\\thetadata_terminal.py

Usage:
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_post_earnings_pcs_thetadata_nbbo.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.post_earnings_pcs_live import (  # noqa: E402
    book_stats_from_live,
    walk_vertical_tp_sl,
    align_leg_bars,
)
from utils.research.post_earnings_pcs_open_delta import MIN_CREDIT_FRAC  # noqa: E402
from utils.research.post_earnings_pcs_thetadata import (  # noqa: E402
    vertical_nbbo_at_clock,
)
from utils.research.short_put_polygon_spotcheck import bars_from_aggs_response  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("pcs_theta_nbbo")

DEFAULT_CACHE = (
    ROOT
    / "reports"
    / "research"
    / "post_earnings_pcs"
    / "2026-09-30_minute_pilot"
    / "minute_pilot_cache.jsonl"
)


def _load_ok(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("status") == "ok" and obj.get("short_ticker") and obj.get(
                "long_ticker"
            ):
                rows.append(obj)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument(
        "--clock",
        default="09:30",
        help="ET clock for first usable NBBO at/after (e.g. 09:30 or 10:00)",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    clock_tag = str(args.clock).replace(":", "")
    outdir = args.outdir or (
        ROOT
        / "reports"
        / "research"
        / "post_earnings_pcs"
        / ("2026-09-30_thetadata_nbbo_%s" % clock_tag)
    )
    outdir.mkdir(parents=True, exist_ok=True)
    out_cache = outdir / "nbbo_cache.jsonl"

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error(
            "Theta Terminal not reachable at %s — start scripts/pipeline/thetadata_terminal.py",
            client.base_url,
        )
        return 1
    LOG.info("Theta Terminal ready (%s)", client.base_url)

    sample = _load_ok(args.cache)[: int(args.n)]
    LOG.info("Sample n=%d from %s", len(sample), args.cache)

    # optional polygon day bars for exit walk
    from utils.api.polygon.polygon_client import get_polygon_client
    from dotenv import load_dotenv
    import os

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("POLYGON_REQUESTS_PER_MINUTE", "30")
    poly = get_polygon_client()
    day_cache: Dict[str, pd.DataFrame] = {}

    def fetch_day(ticker: str, start: str, end: str) -> pd.DataFrame:
        ck = "%s|%s|%s" % (ticker, start, end)
        if ck in day_cache:
            return day_cache[ck]
        try:
            resp = poly.get_options_aggregates(ticker, start, end, timespan="day")
            df = bars_from_aggs_response(resp if isinstance(resp, dict) else {})
        except Exception as exc:  # noqa: BLE001
            LOG.warning("day fail %s: %s", ticker, exc)
            df = pd.DataFrame()
        day_cache[ck] = df
        return df

    prev: Dict[str, Dict[str, Any]] = {}
    if out_cache.exists() and not args.refresh:
        with out_cache.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    prev[o.get("key") or ""] = o

    rows: List[Dict[str, Any]] = []
    for i, src in enumerate(sample):
        key = src.get("key") or "%s|%s" % (src["symbol"], src["entry_date"])
        if key in prev and not args.refresh:
            rows.append(prev[key])
            LOG.info("[%d/%d] cache %s %s", i + 1, len(sample), key, prev[key].get("status"))
            continue
        LOG.info("[%d/%d] NBBO@%s %s ...", i + 1, len(sample), args.clock, key)
        ed = date.fromisoformat(str(src["entry_date"])[:10])
        nb = vertical_nbbo_at_clock(
            client,
            short_ticker=src["short_ticker"],
            long_ticker=src["long_ticker"],
            entry_date=ed,
            clock=str(args.clock),
        )
        obj: Dict[str, Any] = {
            "key": key,
            "symbol": src["symbol"],
            "entry_date": src["entry_date"],
            "width": src.get("width"),
            "poly_credit_rth_open": src.get("credit_rth_open"),
            "poly_credit_day_close": src.get("credit_day_close"),
            **nb,
        }
        width = float(src.get("width") or 5.0)
        min_c = MIN_CREDIT_FRAC * width
        obj["min_credit"] = min_c
        ba = obj.get("credit_bidask")
        mid = obj.get("credit_mid")
        obj["meets_min_bidask"] = bool(ba is not None and ba >= min_c - 1e-9)
        obj["meets_min_mid"] = bool(mid is not None and mid >= min_c - 1e-9)

        # walk exits from NBBO entry using polygon day closes
        if nb.get("status") == "ok":
            exp = date.fromisoformat(str(src["expiry"])[:10])
            from datetime import timedelta

            start = (ed - timedelta(days=1)).isoformat()
            end = (exp + timedelta(days=2)).isoformat()
            aligned = align_leg_bars(
                fetch_day(src["short_ticker"], start, end),
                fetch_day(src["long_ticker"], start, end),
            )
            if ba is not None and ba > 0:
                wr = walk_vertical_tp_sl(
                    aligned,
                    entry_date=ed,
                    width=width,
                    expiry=exp,
                    entry_short_mid=float(nb["short_bid"]),
                    entry_long_mid=float(nb["long_ask"]),
                    include_entry_day_exit=True,
                    spread_floor=0.0,
                    spread_frac=0.0,
                )
                obj["walk_bidask_status"] = wr.status
                obj["walk_bidask_exit"] = wr.exit_reason
                obj["walk_bidask_hold"] = wr.hold_days
                obj["walk_bidask_ror"] = wr.ror_hs
                obj["walk_bidask_credit"] = wr.credit_hs
            else:
                obj["walk_bidask_status"] = "no_credit"

            if mid is not None and mid > 0:
                wr_m = walk_vertical_tp_sl(
                    aligned,
                    entry_date=ed,
                    width=width,
                    expiry=exp,
                    entry_short_mid=float(nb["short_mid"]),
                    entry_long_mid=float(nb["long_mid"]),
                    include_entry_day_exit=True,
                    spread_floor=0.0,
                    spread_frac=0.0,
                )
                obj["walk_mid_status"] = wr_m.status
                obj["walk_mid_exit"] = wr_m.exit_reason
                obj["walk_mid_hold"] = wr_m.hold_days
                obj["walk_mid_ror"] = wr_m.ror_hs
            else:
                obj["walk_mid_status"] = "no_credit"
        with out_cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")
        rows.append(obj)
        LOG.info(
            "  -> %s ba=%.3f mid=%.3f poly_print=%.3f meet_ba=%s walk_ba=%s",
            obj.get("status"),
            float(obj.get("credit_bidask") or 0),
            float(obj.get("credit_mid") or 0),
            float(obj.get("poly_credit_rth_open") or 0),
            obj.get("meets_min_bidask"),
            obj.get("walk_bidask_exit"),
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "nbbo_spotcheck.csv", index=False)

    ok = [r for r in rows if r.get("status") == "ok"]

    def _med(key: str) -> Any:
        s = pd.Series([r.get(key) for r in ok], dtype=float).dropna()
        return float(s.median()) if len(s) else None

    ba_book = [
        {
            "ror_hs": r.get("walk_bidask_ror"),
            "hold_days": r.get("walk_bidask_hold"),
            "exit_reason": r.get("walk_bidask_exit"),
        }
        for r in ok
        if r.get("walk_bidask_status") == "ok"
    ]
    mid_book = [
        {
            "ror_hs": r.get("walk_mid_ror"),
            "hold_days": r.get("walk_mid_hold"),
            "exit_reason": r.get("walk_mid_exit"),
        }
        for r in ok
        if r.get("walk_mid_status") == "ok"
    ]

    summary = {
        "attempted": len(rows),
        "ok_n": len(ok),
        "status_counts": df["status"].value_counts().to_dict() if not df.empty else {},
        "med_credit_bidask": _med("credit_bidask"),
        "med_credit_mid": _med("credit_mid"),
        "med_poly_print": _med("poly_credit_rth_open"),
        "meets_min_bidask_pct": (
            100.0 * float(pd.Series([r.get("meets_min_bidask") for r in ok]).astype(bool).mean())
            if ok
            else None
        ),
        "meets_min_mid_pct": (
            100.0 * float(pd.Series([r.get("meets_min_mid") for r in ok]).astype(bool).mean())
            if ok
            else None
        ),
        "book_bidask": book_stats_from_live(ba_book),
        "book_mid": book_stats_from_live(mid_book),
        "clock_et": str(args.clock),
        "wall_s": time.perf_counter() - t0,
        "outdir": str(outdir),
        "note": (
            "ThetaData Standard 1m NBBO at/after %s ET; "
            "entry credit_bidask=short_bid-long_ask; "
            "exits still Polygon day closes mid (no NBBO exit tape)."
            % args.clock
        ),
    }
    (outdir / "nbbo_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    def _bl(label: str, b: Dict[str, Any]) -> str:
        return (
            "%s: n=%s WR=%.0f%% hold=%.1fd RoR=%.2f%% PF=%s"
            % (
                label,
                b.get("n"),
                b.get("win_rate_pct") or 0,
                b.get("avg_hold_days") or 0,
                b.get("avg_ror_pct") or 0,
                (
                    ("%.2f" % b["profit_factor"])
                    if isinstance(b.get("profit_factor"), (int, float))
                    else b.get("profit_factor")
                ),
            )
        )

    lines = [
        "Post-earnings PCS ThetaData NBBO @%s ET spot-check" % args.clock,
        "attempted=%d ok=%d statuses=%s"
        % (summary["attempted"], summary["ok_n"], summary["status_counts"]),
        "Med credits: bid/ask=%s mid=%s poly_print=%s"
        % (
            summary["med_credit_bidask"],
            summary["med_credit_mid"],
            summary["med_poly_print"],
        ),
        "Meet min: bidask%%=%s mid%%=%s"
        % (summary["meets_min_bidask_pct"], summary["meets_min_mid_pct"]),
        _bl("Book bid/ask entry", summary["book_bidask"]),
        _bl("Book NBBO mid entry", summary["book_mid"]),
        "wall=%.1fs out=%s" % (summary["wall_s"], outdir),
        summary["note"],
    ]
    text = "\n".join(lines) + "\n"
    (outdir / "nbbo_summary.txt").write_text(text, encoding="utf-8")
    LOG.info("\n%s", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
