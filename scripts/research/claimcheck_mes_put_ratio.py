#!/usr/bin/env python3
"""Claim-check dtoptions 56 DTE Friday 2x1 put ratio (SPXW proxy for /MES).

Requires Theta Terminal on :25503:
  python scripts\\pipeline\\thetadata_terminal.py

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\claimcheck_mes_put_ratio.py
  python scripts\\research\\claimcheck_mes_put_ratio.py --start 2022-01-01 --end 2022-04-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.mes_put_ratio import (  # noqa: E402
    CLAIMED,
    candidate_expiries,
    fetch_leg_eod,
    fetch_put_chain,
    friday_sessions,
    index_closes,
    load_spx_eod,
    marks_from_legs,
    pick_ratio_strikes,
    settlement_spot,
    simulate_trade,
    summarize_trades,
    trades_to_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("claimcheck_mes_put_ratio")

DEFAULT_OUT = ROOT / "reports" / "research" / "mes_put_ratio"

VARIANTS: List[Dict[str, Any]] = [
    {"name": "mid_expiry", "mode": "mid", "leg_slip": 0.0, "tp_frac": None, "exit_dte": None},
    {"name": "mid_expiry_slip05", "mode": "mid", "leg_slip": 0.05, "tp_frac": None, "exit_dte": None},
    {"name": "mid_expiry_slip10", "mode": "mid", "leg_slip": 0.10, "tp_frac": None, "exit_dte": None},
    {"name": "mid_tp50", "mode": "mid", "leg_slip": 0.0, "tp_frac": 0.5, "exit_dte": None},
    {"name": "mid_tp50_dte21", "mode": "mid", "leg_slip": 0.0, "tp_frac": 0.5, "exit_dte": 21},
    {"name": "mid_tp50_dte21_slip05", "mode": "mid", "leg_slip": 0.05, "tp_frac": 0.5, "exit_dte": 21},
    {"name": "natural_expiry", "mode": "natural", "leg_slip": 0.0, "tp_frac": None, "exit_dte": None},
    {"name": "natural_tp50_dte21", "mode": "natural", "leg_slip": 0.0, "tp_frac": 0.5, "exit_dte": 21},
]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _fetch_one_chain(client, symbol: str, exp: date, day: date, cache: Path):
    t0 = time.time()
    df = fetch_put_chain(client, symbol=symbol, expiration=exp, day=day, cache_dir=cache)
    return day, exp, df, time.time() - t0


def _fetch_one_leg(client, symbol: str, exp: date, strike: float, start: date, end: date, cache: Path):
    df = fetch_leg_eod(
        client,
        symbol=symbol,
        expiration=exp,
        strike=strike,
        start=start,
        end=end,
        cache_dir=cache,
    )
    return exp, strike, df


def prepare_setups(
    client,
    fridays: Sequence[date],
    expirations: Sequence[date],
    *,
    symbol: str,
    cache: Path,
    workers: int,
    long_delta: float,
    short_delta: float,
    target_dte: int,
) -> List[Dict[str, Any]]:
    """Pick the listed Friday expiry nearest 56 DTE that has a chain that day."""
    pending: Dict[date, List[date]] = {
        day: candidate_expiries(expirations, day, target_dte=target_dte, tol=21)
        for day in fridays
    }
    setups: Dict[date, Dict[str, Any]] = {}
    pass_n = 0
    while True:
        batch = [(day, cands[0]) for day, cands in pending.items() if cands and day not in setups]
        if not batch:
            break
        pass_n += 1
        LOG.info("Chain pass %s: %s candidates", pass_n, len(batch))
        found: Dict[date, Any] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [
                pool.submit(_fetch_one_chain, client, symbol, exp, day, cache)
                for day, exp in batch
            ]
            done = 0
            for fut in as_completed(futs):
                day, exp, df, elapsed = fut.result()
                found[day] = (exp, df, elapsed)
                done += 1
                if done % 25 == 0 or done == len(batch):
                    LOG.info(
                        "Chains %s/%s (last %.1fs, rows %s)",
                        done,
                        len(batch),
                        elapsed,
                        0 if df is None else len(df),
                    )
        for day, exp in batch:
            exp_got, df, _elapsed = found[day]
            picked = pick_ratio_strikes(df, long_delta=long_delta, short_delta=short_delta)
            if picked is not None:
                setups[day] = {
                    "entry_date": day,
                    "expiration": exp_got,
                    "skip": None,
                    "dte": (exp_got - day).days,
                    **picked,
                }
            else:
                pending[day] = pending[day][1:]

    for day, cands in pending.items():
        if day not in setups:
            setups[day] = {
                "entry_date": day,
                "expiration": cands[0] if cands else None,
                "skip": "no_strikes",
                "marks": [],
            }

    needed: List[Tuple[date, float, date, date]] = []
    for setup in setups.values():
        if setup.get("skip"):
            continue
        exp = setup["expiration"]
        needed.append((exp, float(setup["long_k"]), setup["entry_date"], exp))
        needed.append((exp, float(setup["short_k"]), setup["entry_date"], exp))

    unique: Dict[Tuple[date, float], List[date]] = {}
    for exp, strike, start, end in needed:
        key = (exp, float(strike))
        if key not in unique:
            unique[key] = [start, end]
        else:
            unique[key][0] = min(unique[key][0], start)
            unique[key][1] = max(unique[key][1], end)
    LOG.info("Fetching %s leg paths", len(unique))
    legs: Dict[Tuple[date, float], Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [
            pool.submit(_fetch_one_leg, client, symbol, exp, strike, span[0], span[1], cache)
            for (exp, strike), span in unique.items()
        ]
        done = 0
        for fut in as_completed(futs):
            exp, strike, df = fut.result()
            legs[(exp, float(strike))] = df
            done += 1
            if done % 40 == 0 or done == len(unique):
                LOG.info("Legs %s/%s", done, len(unique))

    ordered = [setups[day] for day in sorted(setups)]
    for setup in ordered:
        if setup.get("skip"):
            continue
        exp = setup["expiration"]
        long_eod = legs.get((exp, float(setup["long_k"])))
        short_eod = legs.get((exp, float(setup["short_k"])))
        setup["marks"] = marks_from_legs(setup["entry_date"], exp, long_eod, short_eod)
    return ordered


def run_variant(setups: Sequence[Dict[str, Any]], spot_by_day: Dict[date, float], variant: Dict[str, Any]):
    trades = []
    for setup in setups:
        if setup.get("skip"):
            continue
        exp = setup["expiration"]
        trades.append(
            simulate_trade(
                entry_date=setup["entry_date"],
                expiration=exp,
                spot_entry=float(setup["spot"]),
                long_k=float(setup["long_k"]),
                short_k=float(setup["short_k"]),
                long_delta=float(setup["long_delta"]),
                short_delta=float(setup["short_delta"]),
                marks=setup["marks"],
                settle_spot=settlement_spot(spot_by_day, exp),
                mode=variant["mode"],
                leg_slip=float(variant["leg_slip"]),
                tp_frac=variant["tp_frac"],
                exit_dte=variant["exit_dte"],
                stop_credit_mult=variant.get("stop_credit_mult"),
                tp_at_limit=bool(variant.get("tp_at_limit")),
            )
        )
    return trades


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Claim-check 56 DTE Friday 2x1 put ratio")
    p.add_argument("--symbol", default="SPXW")
    p.add_argument("--start", default="2021-10-01")
    p.add_argument("--end", default="2026-10-01")
    p.add_argument("--long-delta", type=float, default=0.14)
    p.add_argument("--short-delta", type=float, default=0.10)
    p.add_argument("--target-dte", type=int, default=56)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--outdir", default="")
    args = p.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out = Path(args.outdir) if args.outdir else DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d")
    cache = DEFAULT_OUT / "_cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal is not up on :25503")
        return 1

    t0 = time.time()
    spot = load_spx_eod(client, start, end + timedelta(days=80))
    spot_by_day = index_closes(spot)
    if not spot_by_day:
        LOG.error("No SPX closes")
        return 1
    sessions = sorted(spot_by_day)
    LOG.info("SPX sessions %s -> %s (%s)", sessions[0], sessions[-1], len(sessions))
    expirations = list_expirations(client, args.symbol)
    fridays = friday_sessions(sessions, start, end)
    LOG.info("Friday sessions: %s", len(fridays))
    setups = prepare_setups(
        client,
        fridays,
        expirations,
        symbol=args.symbol,
        cache=cache,
        workers=args.workers,
        long_delta=args.long_delta,
        short_delta=args.short_delta,
        target_dte=args.target_dte,
    )
    priced = sum(1 for s in setups if not s.get("skip"))
    dtes = [int(s["dte"]) for s in setups if not s.get("skip") and s.get("dte")]
    if dtes:
        exact = sum(1 for d in dtes if d == args.target_dte)
        LOG.info(
            "Priced setups %s / %s | DTE min %s med %s max %s | exact %sDTE %s",
            priced,
            len(setups),
            min(dtes),
            sorted(dtes)[len(dtes) // 2],
            max(dtes),
            args.target_dte,
            exact,
        )
    else:
        LOG.info("Priced setups %s / %s", priced, len(setups))

    summary: Dict[str, Any] = {
        "claimed": CLAIMED,
        "symbol": args.symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "schedule": len(fridays),
        "priced": priced,
        "elapsed_sec": None,
        "variants": {},
    }
    for variant in VARIANTS:
        trades = run_variant(setups, spot_by_day, variant)
        stats = summarize_trades(trades)
        credit_trades = [t for t in trades if not t.skipped and t.credit_pts > 0]
        stats_credit = summarize_trades(credit_trades)
        summary["variants"][variant["name"]] = {
            "params": {k: variant[k] for k in ("mode", "leg_slip", "tp_frac", "exit_dte")},
            "all": stats,
            "credit_only": stats_credit,
        }
        frame = trades_to_frame(trades)
        frame.to_csv(out / f"{variant['name']}_trades.csv", index=False)
        LOG.info(
            "%s n=%s WR=%s losses=%s PF=%s total=%s 2022losses=%s",
            variant["name"],
            stats.get("n"),
            stats.get("win_rate_pct"),
            stats.get("losses"),
            stats.get("pf"),
            stats.get("total_pnl"),
            stats.get("losses_entry_2022"),
        )

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    LOG.info("Wrote %s in %ss", out, summary["elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
