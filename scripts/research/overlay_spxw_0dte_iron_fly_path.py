#!/usr/bin/env python3
"""Path MTM + Phase-5 vs SPY overlay for the SPXW 0DTE short iron fly.

Requires Theta Terminal on :25503 and the claim-check trades.csv.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\pipeline\\thetadata_terminal.py
  python scripts\\research\\overlay_spxw_0dte_iron_fly_path.py
  python scripts\\research\\overlay_spxw_0dte_iron_fly_path.py --limit 20
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
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.spxw_0dte_iron_fly import FILL_MID, FILL_NATURAL  # noqa: E402
from utils.research.spxw_0dte_iron_fly_path import (  # noqa: E402
    analyze_trade_path,
    phase5_report,
    summarize_path_book,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("overlay_spxw_iron_fly_path")

DEFAULT_TRADES = (
    ROOT / "reports" / "research" / "spxw_0dte_iron_fly" / "2026-10-27" / "trades.csv"
)
DEFAULT_OUT = (
    ROOT / "reports" / "research" / "spxw_0dte_iron_fly" / "2026-10-27" / "path_phase5"
)
START_EQUITY = 25_000.0


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
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _load_spy(start: date, end: date) -> pd.Series:
    import yfinance as yf

    raw = yf.download(
        "SPY",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = pd.to_numeric(raw["Close"], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index).normalize())
    return s.dropna()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fill", choices=(FILL_MID, FILL_NATURAL, "both"), default="both")
    ap.add_argument("--mark", choices=(FILL_MID, FILL_NATURAL), default=FILL_MID,
                    help="Intraday MTM mark (default mid; natural = cover asks)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal not ready on :25503")
        return 1

    trades = pd.read_csv(args.trades)
    fills = [FILL_MID, FILL_NATURAL] if args.fill == "both" else [args.fill]
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / "cache"

    t0 = time.time()
    all_summaries: Dict[str, Any] = {}
    phase5_all: Dict[str, Any] = {}

    for fill in fills:
        sub = trades.loc[trades["fill"] == fill].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.date
        sub = sub.sort_values("date")
        rows_in = sub.to_dict(orient="records")
        if args.limit and args.limit > 0:
            rows_in = rows_in[: int(args.limit)]
        LOG.info("Path overlay fill=%s mark=%s n=%d", fill, args.mark, len(rows_in))

        results: List[Dict[str, Any]] = []
        workers = max(1, int(args.workers))

        def _job(row: Dict[str, Any]) -> Dict[str, Any]:
            return analyze_trade_path(
                client, row, cache_dir=cache_dir, mark=args.mark
            )

        if workers == 1:
            for i, row in enumerate(rows_in, 1):
                results.append(_job(row))
                if i % 25 == 0 or i == len(rows_in):
                    LOG.info("[%s] Progress %d/%d", fill, i, len(rows_in))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_job, r): r for r in rows_in}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    results.append(fut.result())
                    if done % 25 == 0 or done == len(rows_in):
                        LOG.info("[%s] Progress %d/%d", fill, done, len(rows_in))

        results.sort(key=lambda r: str(r.get("date")))
        skips = {}
        usable = []
        for r in results:
            if "skip" in r:
                skips[r["skip"]] = skips.get(r["skip"], 0) + 1
            else:
                usable.append(r)

        path_summary = summarize_path_book(usable)
        path_summary["skips"] = skips
        all_summaries[fill] = path_summary
        LOG.info(
            "[%s] path n=%s MAE_med=%s UW_half=%s TP_hit=%s settle_E=%s tp_E=%s slip_med$=%s",
            fill,
            path_summary.get("n"),
            None
            if path_summary.get("mae_median") is None
            else f"{path_summary['mae_median']:.0f}",
            None
            if path_summary.get("underwater_half_frac") is None
            else f"{path_summary['underwater_half_frac']:.1%}",
            None
            if path_summary.get("tp_hit_frac") is None
            else f"{path_summary['tp_hit_frac']:.1%}",
            None
            if not path_summary.get("settle")
            else f"{path_summary['settle'].get('expectancy'):.1f}",
            None
            if not path_summary.get("tp50")
            else f"{path_summary['tp50'].get('expectancy'):.1f}",
            None
            if path_summary.get("entry_slip_dollars_median") is None
            else f"{path_summary['entry_slip_dollars_median']:.0f}",
        )

        # Phase-5 on settle and tp50 books
        if usable:
            d0 = min(r["date"] for r in usable)
            d1 = max(r["date"] for r in usable)
            if isinstance(d0, str):
                d0 = date.fromisoformat(d0[:10])
                d1 = date.fromisoformat(str(d1)[:10])
            spy = _load_spy(d0, d1)
            sessions = [ts.date() for ts in spy.index]
            settle_trades = [
                {"date": r["date"], "pnl": float(r["settle_pnl"])} for r in usable
            ]
            tp_trades = [
                {"date": r["date"], "pnl": float(r["tp_pnl"])} for r in usable
            ]
            p5_settle = phase5_report(
                f"iron_fly_{fill}_settle",
                settle_trades,
                sessions,
                spy,
                start_equity=START_EQUITY,
            )
            p5_tp = phase5_report(
                f"iron_fly_{fill}_tp50",
                tp_trades,
                sessions,
                spy,
                start_equity=START_EQUITY,
            )
            phase5_all[fill] = {"settle": p5_settle, "tp50": p5_tp}
            LOG.info(
                "[%s] Phase5 settle: %s Sharpe=%.2f vs SPY %.2f MDD %.1f%% vs %.1f%%",
                fill,
                p5_settle["gate"],
                p5_settle["strat"]["sharpe"],
                p5_settle["spy"]["sharpe"],
                100.0 * abs(p5_settle["strat"]["max_drawdown"]),
                100.0 * abs(p5_settle["spy"]["max_drawdown"]),
            )
            LOG.info(
                "[%s] Phase5 tp50: %s Sharpe=%.2f MDD %.1f%%",
                fill,
                p5_tp["gate"],
                p5_tp["strat"]["sharpe"],
                100.0 * abs(p5_tp["strat"]["max_drawdown"]),
            )

        # Write per-fill path trades
        flat = []
        for r in usable:
            row = {k: v for k, v in r.items()}
            for k in ("mae_ts", "mfe_ts", "tp_ts", "date"):
                if k in row and row[k] is not None:
                    row[k] = str(row[k])
            flat.append(row)
        pd.DataFrame(flat).to_csv(outdir / f"path_{fill}.csv", index=False)

    wall = time.time() - t0
    payload = {
        "params": {
            "trades": str(args.trades),
            "fill": args.fill,
            "mark": args.mark,
            "limit": args.limit,
            "workers": args.workers,
            "start_equity": START_EQUITY,
        },
        "path": all_summaries,
        "phase5": phase5_all,
        "wall_seconds": wall,
        "verdict": _verdict(phase5_all, all_summaries),
    }
    out_path = outdir / "summary_phase5.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    LOG.info("Wrote %s (wall %.1fs)", out_path, wall)
    print(json.dumps(_jsonable({"verdict": payload["verdict"], "phase5": phase5_all, "path": {
        k: {
            "n": v.get("n"),
            "mae_median": v.get("mae_median"),
            "underwater_half_frac": v.get("underwater_half_frac"),
            "winner_underwater_half_frac": v.get("winner_underwater_half_frac"),
            "tp_hit_frac": v.get("tp_hit_frac"),
            "entry_slip_dollars_median": v.get("entry_slip_dollars_median"),
            "entry_slip_dollars_p90": v.get("entry_slip_dollars_p90"),
            "exit_mid_minus_nat_median": v.get("exit_mid_minus_nat_median"),
            "settle_E": (v.get("settle") or {}).get("expectancy"),
            "tp50_E": (v.get("tp50") or {}).get("expectancy"),
        }
        for k, v in all_summaries.items()
    }, "wall": wall}), indent=2))
    return 0


def _verdict(phase5: Dict[str, Any], path: Dict[str, Any]) -> Dict[str, Any]:
    nat = phase5.get(FILL_NATURAL) or {}
    settle = nat.get("settle") or {}
    path_nat = path.get(FILL_NATURAL) or {}
    gate_pass = bool(settle.get("gate_pass"))
    uw = path_nat.get("winner_underwater_half_frac")
    # Soft: PASS gate but path shows many winners underwater -> do not wire
    path_soft = gate_pass and uw is not None and float(uw) >= 0.25
    # Research soft-promote when Phase-5 passes and path is not terrible.
    # Still not wired: needs live/paper SPXW BAG fill evidence.
    soft_promote = gate_pass and not path_soft
    return {
        "promote": False,
        "soft_promote": soft_promote,
        "wire": False,
        "gate_pass_natural_settle": gate_pass,
        "gate": settle.get("gate"),
        "path_soft_do_not_wire": path_soft,
        "winner_underwater_half_frac": uw,
        "reason": (
            "Phase-5 PASS on natural settle; path OK - soft-promote, not wired (need live BAG fills)"
            if soft_promote
            else (
                "Phase-5 PASS but path MAE soft - do not wire"
                if path_soft
                else f"Phase-5 {settle.get('gate') or 'n/a'}"
            )
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
