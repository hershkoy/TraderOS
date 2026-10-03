#!/usr/bin/env python3
"""Take-profit sweep for the 56 DTE Friday 2x1 put ratio.

Reuses the cached SPXW chains from claimcheck_mes_put_ratio.py.
``tp_frac`` is the package value still left, as a fraction of the entry credit.
0.50 means exit once half the credit has been kept.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\sweep_mes_put_ratio_tp.py
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
LOG = logging.getLogger("sweep_mes_put_ratio_tp")

DEFAULT_CACHE = ROOT / "reports" / "research" / "mes_put_ratio" / "_cache"
DEFAULT_OUT = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-02_tp"

# Remaining package value / entry credit. None holds to expiration.
# 0.75 = bank 25% of the credit, 0.50 = bank half, 0.25 = bank 75%.
TP_LEVELS: List[Tuple[str, Optional[float]]] = [
    ("hold", None),
    ("tp25", 0.75),
    ("tp50", 0.50),
    ("tp75", 0.25),
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
    """MTM dollars on sessions still open, before the exit day."""
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
    summary: Dict[str, Any] = {"levels": {}}
    for name, frac in TP_LEVELS:
        trades = run_variant(
            setups,
            spot_by_day,
            {"mode": "mid", "leg_slip": 0.0, "tp_frac": frac, "exit_dte": None},
        )
        stats = summarize_trades(trades)
        kept = [t for t in trades if not t.skipped and t.pnl_dollars is not None]
        mae = sorted(t.mae_pts * 100.0 for t in kept)
        path = _book_path(setups, trades)
        focus = []
        for trade in kept:
            if trade.entry_date.isoformat() in FOCUS:
                focus.append(
                    {
                        "entry": trade.entry_date.isoformat(),
                        "exit": trade.exit_date.isoformat() if trade.exit_date else None,
                        "pnl": round(trade.pnl_dollars, 2),
                        "mae": round(trade.mae_pts * 100.0, 2),
                        "hold": trade.hold_days,
                        "reason": trade.exit_reason,
                    }
                )
        summary["levels"][name] = {
            "tp_frac": frac,
            "capture": None if frac is None else round(1.0 - frac, 2),
            "stats": stats,
            "mae_p50": round(mae[len(mae) // 2], 2) if mae else None,
            "mae_p10": round(mae[max(0, int(len(mae) * 0.1) - 1)], 2) if mae else None,
            "mae_worst": round(mae[0], 2) if mae else None,
            "n_mae_lt_5k": sum(1 for x in mae if x < -5000),
            "n_mae_lt_10k": sum(1 for x in mae if x < -10000),
            "n_mae_lt_18k": sum(1 for x in mae if x < -18000),
            "book": path,
            "focus": sorted(focus, key=lambda row: row["entry"]),
        }
        LOG.info(
            "%s n=%s losses=%s total=%s mae_worst=%s book_dd=%s",
            name,
            stats.get("n"),
            stats.get("losses"),
            stats.get("total_pnl"),
            summary["levels"][name]["mae_worst"],
            path.get("max_dd"),
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
    (DEFAULT_OUT / "tp_sweep.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    print(DEFAULT_OUT / "tp_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
