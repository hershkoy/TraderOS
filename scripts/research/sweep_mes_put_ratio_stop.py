#!/usr/bin/env python3
"""Credit-multiple stop on the 56 DTE Friday 2x1 put ratio.

``stop_credit_mult`` 3.0 closes when the close-side loss reaches 300% of the
entry credit (pnl <= -3 * credit). The exit is that day's close mark, so a
gap can finish worse than the multiple. Reuses the cached SPXW chains.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\sweep_mes_put_ratio_stop.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.claimcheck_mes_put_ratio import (  # noqa: E402
    prepare_setups,
    run_variant,
)
from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402
from utils.research.forward_factor import list_expirations  # noqa: E402
from utils.research.mes_put_ratio import (  # noqa: E402
    friday_sessions,
    index_closes,
    load_spx_eod,
    overlapping_book_equity,
    package_value,
    summarize_trades,
    transacted_prices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("sweep_mes_put_ratio_stop")

DEFAULT_CACHE = ROOT / "reports" / "research" / "mes_put_ratio" / "_cache"
DEFAULT_OUT = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-03_stop"

# name, tp_frac (None = hold), stop multiple of credit (None = no stop)
LEVELS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("tp50", 0.50, None),
    ("tp50_sl2", 0.50, 2.0),
    ("tp50_sl3", 0.50, 3.0),
    ("tp50_sl4", 0.50, 4.0),
    ("hold_sl3", None, 3.0),
    ("hold_sl4", None, 4.0),
]

FOCUS = {
    "2025-02-07",
    "2025-02-14",
    "2025-02-21",
    "2025-03-21",
    "2025-03-28",
    "2024-07-12",
    "2022-08-19",
}


def _open_marks(
    setup: Dict[str, Any],
    *,
    exit_day: date,
    credit: float,
) -> List[Tuple[date, float]]:
    rows: List[Tuple[date, float]] = []
    for mark in setup.get("marks") or []:
        day = mark["day"]
        if day <= setup["entry_date"] or day >= exit_day:
            continue
        long_px, short_px = transacted_prices(mark, mode="mid", leg_slip=0.0, closing=True)
        pnl_pts = credit - package_value(long_px, short_px)
        rows.append((day, pnl_pts * 100.0))
    return rows


def _book_path(setups: Sequence[Dict[str, Any]], trades) -> Dict[str, Any]:
    by_entry = {t.entry_date: t for t in trades if not t.skipped and t.exit_date}
    payloads = []
    for setup in setups:
        if setup.get("skip"):
            continue
        trade = by_entry.get(setup["entry_date"])
        if trade is None or trade.exit_date is None:
            continue
        payloads.append(
            {
                "entry": trade.entry_date,
                "exit": trade.exit_date,
                "realized": float(trade.pnl_dollars),
                "marks": _open_marks(setup, exit_day=trade.exit_date, credit=float(trade.credit_pts)),
            }
        )
    return overlapping_book_equity(payloads)


def _kept_map(trades) -> Dict[date, Any]:
    return {t.entry_date: t for t in trades if not t.skipped and t.exit_date}


def _row(trade) -> Dict[str, Any]:
    return {
        "entry": trade.entry_date.isoformat(),
        "exit": trade.exit_date.isoformat() if trade.exit_date else None,
        "pnl": round(trade.pnl_dollars, 2),
        "mae": round(trade.mae_pts * 100.0, 2),
        "credit": round(trade.credit_pts * 100.0, 2),
        "hold": trade.hold_days,
        "reason": trade.exit_reason,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    start = date(2021, 10, 1)
    end = date(2026, 10, 1)
    client = get_thetadata_client()
    if not client.is_ready():
        LOG.error("Theta Terminal is not up on :25503")
        return 1
    spot = load_spx_eod(client, start, end + timedelta(days=80))
    spot_by_day = index_closes(spot)
    sessions = sorted(spot_by_day)
    expirations = list_expirations(client, "SPXW")
    fridays = friday_sessions(sessions, start, end)
    setups = prepare_setups(
        client,
        fridays,
        expirations,
        symbol="SPXW",
        cache=DEFAULT_CACHE,
        workers=4,
        long_delta=0.14,
        short_delta=0.10,
        target_dte=56,
    )
    books: Dict[str, Dict[date, Any]] = {}
    summary: Dict[str, Any] = {"levels": {}}
    for name, frac, stop in LEVELS:
        trades = run_variant(
            setups,
            spot_by_day,
            {
                "mode": "mid",
                "leg_slip": 0.0,
                "tp_frac": frac,
                "exit_dte": None,
                "stop_credit_mult": stop,
            },
        )
        books[name] = _kept_map(trades)
        stats = summarize_trades(trades)
        kept = list(books[name].values())
        mae = sorted(t.mae_pts * 100.0 for t in kept)
        path = _book_path(setups, trades)
        focus = [_row(t) for t in kept if t.entry_date.isoformat() in FOCUS]
        stops = [_row(t) for t in kept if t.exit_reason == "stop"]
        summary["levels"][name] = {
            "tp_frac": frac,
            "stop_credit_mult": stop,
            "stats": stats,
            "mae_worst": round(mae[0], 2) if mae else None,
            "n_mae_lt_5k": sum(1 for x in mae if x < -5000),
            "n_mae_lt_10k": sum(1 for x in mae if x < -10000),
            "n_mae_lt_18k": sum(1 for x in mae if x < -18000),
            "n_stop": len(stops),
            "stop_pnl": round(sum(r["pnl"] for r in stops), 2),
            "book": path,
            "focus": sorted(focus, key=lambda row: row["entry"]),
            "stops": sorted(stops, key=lambda row: row["entry"]),
        }
        LOG.info(
            "%s n=%s losses=%s stops=%s total=%s mae_worst=%s book_dd=%s",
            name,
            stats.get("n"),
            stats.get("losses"),
            len(stops),
            stats.get("total_pnl"),
            summary["levels"][name]["mae_worst"],
            path.get("max_dd"),
        )

    # Each stop vs the same entry held, and vs 50% take-profit with no stop.
    recovery: Dict[str, List[Dict[str, Any]]] = {}
    tp50 = books["tp50"]
    for name in ("tp50_sl2", "tp50_sl3", "tp50_sl4", "hold_sl3", "hold_sl4"):
        rows = []
        for stopped in summary["levels"][name]["stops"]:
            day = date.fromisoformat(stopped["entry"])
            base = tp50.get(day)
            rows.append(
                {
                    "entry": stopped["entry"],
                    "stop_exit": stopped["exit"],
                    "stop_pnl": stopped["pnl"],
                    "stop_mae": stopped["mae"],
                    "credit": stopped["credit"],
                    "tp50_pnl": None if base is None else round(base.pnl_dollars, 2),
                    "tp50_reason": None if base is None else base.exit_reason,
                    "tp50_mae": None if base is None else round(base.mae_pts * 100.0, 2),
                }
            )
        recovery[name] = rows
    # One unmanaged hold pass so recovery can see the expiry result.
    hold_trades = run_variant(
        setups,
        spot_by_day,
        {"mode": "mid", "leg_slip": 0.0, "tp_frac": None, "exit_dte": None},
    )
    hold_map = _kept_map(hold_trades)
    for name, rows in recovery.items():
        for row in rows:
            held = hold_map.get(date.fromisoformat(row["entry"]))
            row["hold_pnl"] = None if held is None else round(held.pnl_dollars, 2)
            row["hold_reason"] = None if held is None else held.exit_reason
            row["recovered_on_hold"] = bool(held is not None and held.pnl_dollars > 0)
            row["recovered_on_tp50"] = bool(row["tp50_pnl"] is not None and row["tp50_pnl"] > 0)
        green_hold = [r for r in rows if r["recovered_on_hold"]]
        red_hold = [r for r in rows if r["hold_pnl"] is not None and r["hold_pnl"] <= 0]
        green_tp = [r for r in rows if r["recovered_on_tp50"]]
        summary["levels"][name]["recovery"] = {
            "n_stop": len(rows),
            "n_hold_green": len(green_hold),
            "n_hold_red": len(red_hold),
            "n_tp50_green": len(green_tp),
            "stop_pnl": round(sum(r["stop_pnl"] for r in rows), 2),
            "same_entries_hold_pnl": round(sum(r["hold_pnl"] or 0.0 for r in rows), 2),
            "same_entries_tp50_pnl": round(sum(r["tp50_pnl"] or 0.0 for r in rows), 2),
            "rows": rows,
        }
        LOG.info(
            "%s stops=%s hold_green=%s hold_red=%s tp50_green=%s stop$=%s hold$=%s tp50$=%s",
            name,
            len(rows),
            len(green_hold),
            len(red_hold),
            len(green_tp),
            summary["levels"][name]["recovery"]["stop_pnl"],
            summary["levels"][name]["recovery"]["same_entries_hold_pnl"],
            summary["levels"][name]["recovery"]["same_entries_tp50_pnl"],
        )

    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
            return None
        return obj

    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    out = DEFAULT_OUT / "stop_sweep.json"
    out.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
