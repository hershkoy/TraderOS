#!/usr/bin/env python3
"""Polygon listed vertical mid/half-spread spot-check for post-earnings PCS.

Re-walks TP50 / SL 2x on short+long put day bars for recent BS claim-check rows.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  set POLYGON_REQUESTS_PER_MINUTE=4
  python scripts\\research\\run_post_earnings_pcs_live_realism.py --n 40
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
os.environ.setdefault("POLYGON_REQUESTS_PER_MINUTE", "4")

from utils.api.polygon.polygon_client import get_polygon_client  # noqa: E402
from utils.research.post_earnings_pcs_live import (  # noqa: E402
    align_leg_bars,
    book_stats_from_live,
    resolve_vertical_tickers,
    select_recent_pcs_sample,
    walk_vertical_tp_sl,
)
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.short_put_polygon_spotcheck import bars_from_aggs_response  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("pcs_live_realism")
logging.getLogger("utils.api.polygon.polygon_client").setLevel(logging.WARNING)

DEFAULT_TRADES = (
    ROOT / "reports" / "research" / "post_earnings_pcs" / "2026-09-29" / "trades.csv"
)


def _row_key(symbol: str, entry_date: Any) -> str:
    d = pd.Timestamp(entry_date).strftime("%Y-%m-%d")
    return "%s|%s" % (str(symbol).upper(), d)


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = obj.get("key") or _row_key(obj.get("symbol"), obj.get("bs_entry_date"))
            out[k] = obj
    return out


def _append_cache(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _fetch_bars(client: Any, ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        resp = client.get_options_aggregates(ticker, start, end, timespan="day")
        return bars_from_aggs_response(resp if isinstance(resp, dict) else {})
    except Exception as exc:  # noqa: BLE001
        LOG.warning("aggs %s failed: %s", ticker, exc)
        return pd.DataFrame()


def _fetch_one(
    client: Any,
    row: pd.Series,
    *,
    spread_floor: float,
    spread_frac: float,
) -> Dict[str, Any]:
    sym = str(row["symbol"]).upper()
    entry_d = pd.Timestamp(row["entry_date"]).date()
    exit_bs = pd.Timestamp(row["exit_date"]).date()
    dte = int(row["dte"]) if pd.notna(row.get("dte")) else 37
    short_k = float(row["short_k"])
    long_k = float(row["long_k"])
    width = float(row["width"])

    out: Dict[str, Any] = {
        "key": _row_key(sym, entry_d),
        "symbol": sym,
        "bs_entry_date": entry_d.isoformat(),
        "bs_exit_date": exit_bs.isoformat(),
        "bs_short_k": short_k,
        "bs_long_k": long_k,
        "bs_width": width,
        "bs_credit": float(row["credit"]),
        "bs_ror_pct": float(row["ror_pct"]),
        "bs_hold_days": int(row["hold_days"]),
        "bs_exit_reason": str(row["exit_reason"]),
        "status": "pending",
    }

    short_t, long_t, expiry, sk, lk = resolve_vertical_tickers(
        sym, entry_d, short_k, long_k, dte
    )
    width_live = float(sk) - float(lk)
    if width_live <= 0:
        out["status"] = "bad_width"
        return out

    out.update(
        {
            "live_short_ticker": short_t,
            "live_long_ticker": long_t,
            "live_expiry": expiry,
            "live_short_k": sk,
            "live_long_k": lk,
            "live_width": width_live,
        }
    )

    start = (entry_d - timedelta(days=2)).isoformat()
    end = (
        max(exit_bs, date.fromisoformat(expiry[:10])) + timedelta(days=5)
    ).isoformat()

    short_bars = _fetch_bars(client, short_t, start, end)
    long_bars = _fetch_bars(client, long_t, start, end)
    if short_bars.empty or long_bars.empty:
        out["status"] = "no_bars"
        out["n_short_bars"] = int(len(short_bars))
        out["n_long_bars"] = int(len(long_bars))
        return out

    aligned = align_leg_bars(short_bars, long_bars)
    walk = walk_vertical_tp_sl(
        aligned,
        entry_date=entry_d,
        width=width_live,
        expiry=date.fromisoformat(expiry[:10]),
        spread_floor=spread_floor,
        spread_frac=spread_frac,
        entry_slack_days=0,
    )
    out["status"] = walk.status
    out["n_aligned_bars"] = walk.n_bars
    if walk.entry_date is not None:
        out["live_entry_date"] = walk.entry_date.isoformat()
    if walk.exit_date is not None:
        out["live_exit_date"] = walk.exit_date.isoformat()
    out["credit_mid"] = walk.credit_mid
    out["credit_hs"] = walk.credit_hs
    out["exit_debit_mid"] = walk.exit_debit_mid
    out["exit_debit_hs"] = walk.exit_debit_hs
    out["exit_reason"] = walk.exit_reason
    out["hold_days"] = walk.hold_days
    out["ror_mid"] = walk.ror_mid
    out["ror_hs"] = walk.ror_hs
    out["pnl_mid"] = walk.pnl_mid
    out["pnl_hs"] = walk.pnl_hs
    if walk.credit_mid and float(row["credit"]) > 0:
        out["entry_vs_bs"] = float(walk.credit_mid) / float(row["credit"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-earnings PCS Polygon live realism")
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-entry", default="2024-10-01")
    ap.add_argument("--spread-floor", type=float, default=0.05)
    ap.add_argument("--spread-frac", type=float, default=0.08)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    outdir = args.outdir or dated_outdir(
        ROOT / "reports" / "research" / "post_earnings_pcs"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "pcs_live_realism_cache.jsonl"

    trades = pd.read_csv(args.trades)
    sample = select_recent_pcs_sample(
        trades, n=int(args.n), min_entry=str(args.min_entry)
    )
    LOG.info("Sample n=%d from %s (min_entry=%s)", len(sample), args.trades, args.min_entry)
    if sample.empty:
        LOG.error("No trades in sample window")
        return 1

    cache = {} if args.refresh else _load_cache(cache_path)
    client = get_polygon_client()
    rows: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(sample.iterrows()):
        key = _row_key(row["symbol"], row["entry_date"])
        if key in cache and not args.refresh:
            obj = cache[key]
            rows.append(obj)
            LOG.info("[%d/%d] cache %s status=%s", i + 1, len(sample), key, obj.get("status"))
            continue
        LOG.info("[%d/%d] fetch %s ...", i + 1, len(sample), key)
        obj = _fetch_one(
            client,
            row,
            spread_floor=float(args.spread_floor),
            spread_frac=float(args.spread_frac),
        )
        _append_cache(cache_path, obj)
        cache[key] = obj
        rows.append(obj)
        LOG.info(
            "  -> %s credit_hs=%s ror_hs=%s reason=%s",
            obj.get("status"),
            obj.get("credit_hs"),
            obj.get("ror_hs"),
            obj.get("exit_reason"),
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pcs_live_realism.csv", index=False)

    ok_all = [r for r in rows if r.get("status") == "ok"]
    honest = []
    n_slid = 0
    for r in ok_all:
        be = str(r.get("bs_entry_date") or "")[:10]
        le = str(r.get("live_entry_date") or "")[:10]
        if le and be and le != be:
            n_slid += 1
            continue
        honest.append(r)

    mid_stats = book_stats_from_live(honest, ror_key="ror_mid")
    hs_stats = book_stats_from_live(honest, ror_key="ror_hs")
    bs_matched = []
    for r in honest:
        if r.get("bs_ror_pct") is not None:
            bs_matched.append({"ror_hs": float(r["bs_ror_pct"]), "hold_days": r.get("bs_hold_days"), "exit_reason": r.get("bs_exit_reason")})
    # reuse helper with fake key
    bs_stats = book_stats_from_live(
        [{"ror_hs": x["ror_hs"], "hold_days": x["hold_days"], "exit_reason": x["exit_reason"]} for x in bs_matched],
        ror_key="ror_hs",
    )

    status_counts: Dict[str, int] = {}
    for r in rows:
        st = str(r.get("status") or "?")
        status_counts[st] = status_counts.get(st, 0) + 1

    entry_vs = [
        float(r["entry_vs_bs"])
        for r in honest
        if r.get("entry_vs_bs") is not None
    ]

    summary = {
        "attempted": len(rows),
        "status_counts": status_counts,
        "status_ok": len(ok_all),
        "slid_entry": n_slid,
        "honest_ok": len(honest),
        "live_mid": mid_stats,
        "live_half_spread": hs_stats,
        "bs_same_rows": bs_stats,
        "entry_vs_bs_mid": {
            "n": len(entry_vs),
            "median": float(pd.Series(entry_vs).median()) if entry_vs else None,
            "mean": float(pd.Series(entry_vs).mean()) if entry_vs else None,
        },
        "spread": {"floor": args.spread_floor, "frac": args.spread_frac},
        "wall_s": time.perf_counter() - t0,
        "outdir": str(outdir),
    }
    (outdir / "pcs_live_realism_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "Post-earnings PCS live realism (listed vertical + half-spread)",
        "spread = max(%.2f, %.0f%% * leg mid)" % (args.spread_floor, args.spread_frac * 100),
        "attempted=%d status_ok=%d slid=%d honest=%d"
        % (len(rows), len(ok_all), n_slid, len(honest)),
        "status_counts=%s" % status_counts,
        "LIVE mid:    n=%s WR=%s hold=%s avg_RoR=%s PF=%s"
        % (
            mid_stats.get("n"),
            mid_stats.get("win_rate_pct"),
            mid_stats.get("avg_hold_days"),
            mid_stats.get("avg_ror_pct"),
            mid_stats.get("profit_factor"),
        ),
        "LIVE half-spread: n=%s WR=%s hold=%s avg_RoR=%s PF=%s TP%%=%s SL%%=%s"
        % (
            hs_stats.get("n"),
            hs_stats.get("win_rate_pct"),
            hs_stats.get("avg_hold_days"),
            hs_stats.get("avg_ror_pct"),
            hs_stats.get("profit_factor"),
            hs_stats.get("tp_pct"),
            hs_stats.get("sl_pct"),
        ),
        "BS same rows: n=%s WR=%s hold=%s avg_RoR=%s PF=%s"
        % (
            bs_stats.get("n"),
            bs_stats.get("win_rate_pct"),
            bs_stats.get("avg_hold_days"),
            bs_stats.get("avg_ror_pct"),
            bs_stats.get("profit_factor"),
        ),
        "entry_mid/bs_credit med=%s mean=%s"
        % (
            summary["entry_vs_bs_mid"]["median"],
            summary["entry_vs_bs_mid"]["mean"],
        ),
        "wall=%.1fs out=%s" % (summary["wall_s"], outdir),
    ]
    text = "\n".join(lines) + "\n"
    (outdir / "pcs_live_realism_summary.txt").write_text(text, encoding="utf-8")
    LOG.info("\n%s", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
