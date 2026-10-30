#!/usr/bin/env python3
"""Claim-check Jason Rutschel 13:45 SPXW short iron fly (0DTE).

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_spxw_0dte_iron_fly.py
  python scripts\\research\\claimcheck_spxw_0dte_iron_fly.py --limit 20
  python scripts\\research\\claimcheck_spxw_0dte_iron_fly.py --wing 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.spxw_0dte_iron_fly import (  # noqa: E402
    CLAIMED,
    DEFAULT_CALL_DELTA,
    DEFAULT_PUT_DELTA,
    DEFAULT_VIX_MAX,
    DEFAULT_WEEKDAYS,
    DEFAULT_WING,
    FILL_MID,
    FILL_NATURAL,
    claimed_one_lot_pnl,
    compound_fixed_contracts,
    eligible_days,
    fetch_chain,
    load_index_eod,
    simulate_day,
    spx_close_map,
    summarize_pnls,
    vix_open_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_spxw_0dte_iron_fly")

DEFAULT_OUT = ROOT / "reports" / "research" / "spxw_0dte_iron_fly" / "2026-10-27"
MODES = (FILL_MID, FILL_NATURAL)


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _one_day(
    client: Any,
    day: date,
    *,
    settle: float,
    vix: float,
    cache_dir: Path,
    wing: float,
    put_delta: float,
    call_delta: float,
) -> Dict[str, Any]:
    chain = fetch_chain(client, day, cache_dir)
    if chain is None:
        return {"date": day, "skip": "theta_error"}
    if chain.empty:
        return {"date": day, "skip": "no_chain"}
    out: Dict[str, Any] = {"date": day, "vix": vix, "settle": settle, "modes": {}}
    for mode in MODES:
        row, reason = simulate_day(
            chain,
            settle=settle,
            fill=mode,
            put_delta=put_delta,
            call_delta=call_delta,
            wing=wing,
        )
        if row is None:
            out["modes"][mode] = {"skip": reason}
        else:
            out["modes"][mode] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=_parse_date, default=date(2024, 10, 1))
    ap.add_argument("--end", type=_parse_date, default=date(2026, 10, 24))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wing", type=float, default=DEFAULT_WING)
    ap.add_argument("--put-delta", type=float, default=DEFAULT_PUT_DELTA)
    ap.add_argument("--call-delta", type=float, default=DEFAULT_CALL_DELTA)
    ap.add_argument("--vix-max", type=float, default=DEFAULT_VIX_MAX)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--include-wednesday",
        action="store_true",
        help="Include Wednesdays (claim skips them)",
    )
    args = ap.parse_args()

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not ready on :25503")
        return 1

    outdir: Path = args.outdir
    cache_dir = outdir / "cache"
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    LOG.info("Loading SPXW expirations / SPX / VIX")
    exps = list_expirations(client, "SPXW")
    spx = load_index_eod(client, "SPX", args.start, args.end)
    vix = load_index_eod(client, "VIX", args.start, args.end, yahoo="^VIX")
    if not spx.empty:
        LOG.info(
            "SPX EOD %d rows %s -> %s",
            len(spx),
            spx["date"].iloc[0],
            spx["date"].iloc[-1],
        )
    if not vix.empty:
        LOG.info(
            "VIX EOD %d rows %s -> %s",
            len(vix),
            vix["date"].iloc[0],
            vix["date"].iloc[-1],
        )
    settle_map = spx_close_map(spx)
    vix_map = vix_open_map(vix)
    weekdays = (0, 1, 2, 3, 4) if args.include_wednesday else DEFAULT_WEEKDAYS
    days = eligible_days(
        exps,
        start=args.start,
        end=args.end,
        weekdays=weekdays,
        vix_by_day=vix_map,
        vix_max=args.vix_max,
    )
    days = [d for d in days if d in settle_map]
    if args.limit and args.limit > 0:
        days = days[: int(args.limit)]
    LOG.info(
        "Eligible days %d (start=%s end=%s wing=%.0f vix<%.1f)",
        len(days),
        args.start,
        args.end,
        args.wing,
        args.vix_max,
    )

    results: List[Dict[str, Any]] = []
    skips: Counter = Counter()
    workers = max(1, int(args.workers))

    def _job(day: date) -> Dict[str, Any]:
        return _one_day(
            client,
            day,
            settle=float(settle_map[day]),
            vix=float(vix_map.get(day, float("nan"))),
            cache_dir=cache_dir,
            wing=float(args.wing),
            put_delta=float(args.put_delta),
            call_delta=float(args.call_delta),
        )

    if workers == 1:
        for i, day in enumerate(days, 1):
            row = _job(day)
            results.append(row)
            if i % 25 == 0 or i == len(days):
                LOG.info("Progress %d/%d", i, len(days))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_job, d): d for d in days}
            done = 0
            for fut in as_completed(futs):
                done += 1
                results.append(fut.result())
                if done % 25 == 0 or done == len(days):
                    LOG.info("Progress %d/%d", done, len(days))

    results.sort(key=lambda r: r["date"])
    trade_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    for mode in MODES:
        pnls: List[float] = []
        mode_trades: List[Dict[str, Any]] = []
        for r in results:
            if "skip" in r:
                skips[r["skip"]] += 1
                continue
            m = r.get("modes", {}).get(mode) or {}
            if "skip" in m:
                skips[f"{mode}:{m['skip']}"] += 1
                continue
            trade = {
                "date": r["date"],
                "vix": r.get("vix"),
                "fill": mode,
                **{k: v for k, v in m.items() if k != "fill"},
            }
            mode_trades.append(trade)
            trade_rows.append(trade)
            pnls.append(float(m["pnl"]))
        summary = summarize_pnls(pnls)
        summary["compound"] = compound_fixed_contracts(mode_trades, start_equity=25_000.0)
        summaries[mode] = summary
        LOG.info(
            "%s n=%s WR=%s E=%s avgW=%s avgL=%s total=%s PF=%s CAGR=%s",
            mode,
            summary["n"],
            None if summary["win_rate"] is None else f"{summary['win_rate']:.1%}",
            None if summary["expectancy"] is None else f"{summary['expectancy']:.2f}",
            None if summary["avg_win"] is None else f"{summary['avg_win']:.2f}",
            None if summary["avg_loss"] is None else f"{summary['avg_loss']:.2f}",
            f"{summary['total']:.0f}",
            None if summary["pf"] is None else f"{summary['pf']:.2f}",
            None
            if summary["compound"]["cagr"] is None
            else f"{summary['compound']['cagr']:.1%}",
        )

    wall = time.time() - t0
    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty:
        trades_df["date"] = trades_df["date"].astype(str)
        trades_path = outdir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        LOG.info("Wrote %s (%d rows)", trades_path, len(trades_df))

    payload = {
        "claimed": CLAIMED,
        "claimed_one_lot_pnl_approx": claimed_one_lot_pnl(),
        "params": {
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "wing": args.wing,
            "put_delta": args.put_delta,
            "call_delta": args.call_delta,
            "vix_max": args.vix_max,
            "weekdays": list(weekdays),
            "workers": workers,
            "limit": args.limit,
        },
        "eligible_days": len(days),
        "skips": dict(skips),
        "summaries": summaries,
        "wall_seconds": wall,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    LOG.info("Wrote %s (wall %.1fs)", summary_path, wall)
    print(json.dumps(_jsonable({"summaries": summaries, "skips": dict(skips), "wall": wall}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
