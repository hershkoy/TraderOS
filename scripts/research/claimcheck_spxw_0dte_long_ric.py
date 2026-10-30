#!/usr/bin/env python3
"""Claim-check Jason Rutschel morning SPXW long reverse iron condor (0DTE).

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_spxw_0dte_long_ric.py
  python scripts\\research\\claimcheck_spxw_0dte_long_ric.py --limit 20
  python scripts\\research\\claimcheck_spxw_0dte_long_ric.py --vix-mode eod_low --start 2022-05-11
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.spxw_0dte_iron_fly import (  # noqa: E402
    FILL_MID,
    FILL_NATURAL,
    compound_fixed_contracts,
    fetch_chain,
    load_index_eod,
    spx_close_map,
    summarize_pnls,
    vix_open_map,
)
from utils.research.spxw_0dte_long_ric import (  # noqa: E402
    CLAIMED,
    DEFAULT_CALL_DEBIT,
    DEFAULT_PUT_DEBIT,
    DEFAULT_VIX_DROP,
    DEFAULT_WEEKDAYS,
    DEFAULT_WING,
    VIX_MODE_EOD_LOW,
    VIX_MODE_HOURLY,
    claimed_one_lot_pnl,
    eligible_sessions,
    find_vix_trigger_eod_low,
    find_vix_trigger_hourly,
    load_yahoo_vix_hourly,
    simulate_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_spxw_0dte_long_ric")

DEFAULT_OUT = ROOT / "reports" / "research" / "spxw_0dte_long_ric" / "2026-10-27"
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
    if isinstance(value, (date, dtime)):
        return value.isoformat()
    return value


def _vix_maps(vix: pd.DataFrame) -> Dict[str, Dict[date, float]]:
    opens = vix_open_map(vix)
    closes: Dict[date, float] = {}
    lows: Dict[date, float] = {}
    for _, row in vix.iterrows():
        d = row["date"]
        if isinstance(d, datetime):
            d = d.date()
        if pd.notna(row.get("close")):
            closes[d] = float(row["close"])
        if pd.notna(row.get("low")):
            lows[d] = float(row["low"])
    return {"open": opens, "close": closes, "low": lows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Default: ~720d ago for hourly mode, 2022-05-11 for eod_low",
    )
    ap.add_argument("--end", type=_parse_date, default=date(2026, 10, 24))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wing", type=float, default=DEFAULT_WING)
    ap.add_argument("--put-debit", type=float, default=DEFAULT_PUT_DEBIT)
    ap.add_argument("--call-debit", type=float, default=DEFAULT_CALL_DEBIT)
    ap.add_argument("--vix-drop", type=float, default=DEFAULT_VIX_DROP)
    ap.add_argument(
        "--vix-mode",
        choices=(VIX_MODE_HOURLY, VIX_MODE_EOD_LOW),
        default=VIX_MODE_HOURLY,
    )
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-monday", action="store_true")
    args = ap.parse_args()
    from datetime import timedelta as _td

    if args.start is None:
        args.start = (
            date.today() - _td(days=720)
            if args.vix_mode == VIX_MODE_HOURLY
            else date(2022, 5, 11)
        )

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not ready on :25503")
        return 1

    outdir: Path = args.outdir
    if args.vix_mode == VIX_MODE_EOD_LOW and args.outdir == DEFAULT_OUT:
        outdir = DEFAULT_OUT.parent / f"{DEFAULT_OUT.name}_eod_low"
    cache_dir = outdir / "cache"
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    LOG.info("Loading SPXW expirations / SPX / VIX (%s)", args.vix_mode)
    exps = list_expirations(client, "SPXW")

    import yfinance as yf

    def _yahoo_daily(tick: str, start: date, end: date) -> pd.DataFrame:
        raw = yf.download(
            tick,
            start=start.isoformat(),
            end=(end + _td(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(raw.index).date,
                    "open": pd.to_numeric(raw["Open"], errors="coerce"),
                    "high": pd.to_numeric(raw["High"], errors="coerce"),
                    "low": pd.to_numeric(raw["Low"], errors="coerce"),
                    "close": pd.to_numeric(raw["Close"], errors="coerce"),
                }
            )
            .dropna(subset=["date", "close"])
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

    # Prefer Yahoo daily for indexes — Theta FREE EOD 403s on older months.
    spx = _yahoo_daily("^GSPC", args.start, args.end)
    if spx.empty:
        spx = load_index_eod(client, "SPX", args.start, args.end)
    vix_start = args.start - _td(days=14)
    vix = _yahoo_daily("^VIX", vix_start, args.end)
    if vix.empty:
        vix = load_index_eod(client, "VIX", vix_start, args.end, yahoo="^VIX")
    LOG.info("SPX daily rows: %d  VIX daily rows: %d", len(spx), len(vix))
    settle_map = spx_close_map(spx)
    vm = _vix_maps(vix)
    hourly = pd.DataFrame()
    if args.vix_mode == VIX_MODE_HOURLY:
        hourly = load_yahoo_vix_hourly(args.start, args.end)
        LOG.info("Yahoo VIX 1h bars: %d", len(hourly))

    weekdays = (0, 1, 2, 3, 4) if args.include_monday else DEFAULT_WEEKDAYS
    sessions = eligible_sessions(
        exps, start=args.start, end=args.end, weekdays=weekdays
    )
    sessions = [d for d in sessions if d in settle_map and d in vm["open"]]

    triggers: List[Dict[str, Any]] = []
    skip_trig: Counter = Counter()
    for d in sessions:
        prior_days = [x for x in vm["close"] if x < d]
        if not prior_days:
            skip_trig["no_prior"] += 1
            continue
        prior = vm["close"][max(prior_days)]
        open_ = vm["open"][d]
        if args.vix_mode == VIX_MODE_HOURLY:
            entry, reason = find_vix_trigger_hourly(
                d,
                prior_close=prior,
                session_open=open_,
                hourly=hourly,
                min_drop=float(args.vix_drop),
            )
        else:
            low = vm["low"].get(d)
            if low is None:
                skip_trig["no_low"] += 1
                continue
            entry, reason = find_vix_trigger_eod_low(
                prior_close=prior,
                session_open=open_,
                session_low=float(low),
                min_drop=float(args.vix_drop),
            )
        if entry is None:
            skip_trig[reason] += 1
            continue
        triggers.append(
            {
                "date": d,
                "entry": entry,
                "vix_open": open_,
                "vix_prior": prior,
                "settle": float(settle_map[d]),
            }
        )

    if args.limit and args.limit > 0:
        triggers = triggers[: int(args.limit)]
    LOG.info(
        "Triggered days %d / sessions %d (mode=%s drop=%.2f%%)",
        len(triggers),
        len(sessions),
        args.vix_mode,
        100.0 * float(args.vix_drop),
    )

    results: List[Dict[str, Any]] = []
    skips: Counter = Counter(skip_trig)
    workers = max(1, int(args.workers))

    def _job(trig: Dict[str, Any]) -> Dict[str, Any]:
        day = trig["date"]
        clock: dtime = trig["entry"]
        chain = fetch_chain(client, day, cache_dir, clock=clock)
        if chain is None:
            return {"date": day, "skip": "theta_error", **trig}
        if chain.empty:
            return {"date": day, "skip": "no_chain", **trig}
        out: Dict[str, Any] = {**trig, "modes": {}}
        for mode in MODES:
            row, reason = simulate_day(
                chain,
                settle=float(trig["settle"]),
                fill=mode,
                wing=float(args.wing),
                put_debit=float(args.put_debit),
                call_debit=float(args.call_debit),
                clock=clock,
            )
            if row is None:
                out["modes"][mode] = {"skip": reason}
            else:
                out["modes"][mode] = row
        return out

    if workers == 1:
        for i, trig in enumerate(triggers, 1):
            results.append(_job(trig))
            if i % 25 == 0 or i == len(triggers):
                LOG.info("Progress %d/%d", i, len(triggers))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_job, t): t for t in triggers}
            done = 0
            for fut in as_completed(futs):
                done += 1
                results.append(fut.result())
                if done % 25 == 0 or done == len(triggers):
                    LOG.info("Progress %d/%d", done, len(triggers))

    results.sort(key=lambda r: r["date"])
    trade_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    for mode in MODES:
        pnls: List[float] = []
        mode_trades: List[Dict[str, Any]] = []
        for r in results:
            if "skip" in r and "modes" not in r:
                skips[r["skip"]] += 1
                continue
            m = (r.get("modes") or {}).get(mode) or {}
            if "skip" in m:
                skips[f"{mode}:{m['skip']}"] += 1
                continue
            if not m:
                continue
            entry = r.get("entry")
            trade = {
                "date": r["date"],
                "entry": entry.isoformat() if isinstance(entry, dtime) else entry,
                "vix_open": r.get("vix_open"),
                "vix_prior": r.get("vix_prior"),
                "fill": mode,
                **{k: v for k, v in m.items() if k != "fill"},
            }
            mode_trades.append(trade)
            trade_rows.append(trade)
            pnls.append(float(m["pnl"]))
        summary = summarize_pnls(pnls)
        summary["compound"] = compound_fixed_contracts(
            mode_trades, start_equity=25_000.0
        )
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
            "put_debit": args.put_debit,
            "call_debit": args.call_debit,
            "vix_drop": args.vix_drop,
            "vix_mode": args.vix_mode,
            "weekdays": list(weekdays),
            "workers": workers,
            "limit": args.limit,
        },
        "sessions": len(sessions),
        "triggered": len(triggers),
        "skips": dict(skips),
        "summaries": summaries,
        "wall_seconds": wall,
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    LOG.info("Wrote %s (wall %.1fs)", summary_path, wall)
    print(
        json.dumps(
            _jsonable(
                {
                    "triggered": len(triggers),
                    "summaries": summaries,
                    "skips": dict(skips),
                    "wall": wall,
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
