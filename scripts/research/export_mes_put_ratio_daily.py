#!/usr/bin/env python3
"""Daily marked-to-market path for the MES-sized 56 DTE put ratio.

Writes SPX-dollar equity plus a one-MES margin model. The HTML report scales
both by the contract count. Reuses the cached SPXW chains.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\export_mes_put_ratio_daily.py
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
    daily_book_path,
    friday_sessions,
    index_closes,
    load_spx_eod,
    package_value,
    transacted_prices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("export_mes_put_ratio_daily")

DEFAULT_CACHE = ROOT / "reports" / "research" / "mes_put_ratio" / "_cache"
DEFAULT_OUT = ROOT / "reports" / "research" / "mes_put_ratio" / "2026-10-03_stop" / "mes_daily.json"

# name, label, tp_frac, stop multiple, fill the take-profit at the limit
BOOKS: List[Tuple[str, str, Optional[float], Optional[float], bool]] = [
    ("hold", "Hold to expiry", None, None, False),
    ("tp25", "Bank 25% at the limit", 0.75, None, True),
    ("tp50", "Bank 50% at the limit", 0.50, None, True),
    ("tp50_close", "Bank 50% at the close mid", 0.50, None, False),
    ("tp75", "Bank 75% at the limit", 0.25, None, True),
    ("natural_tp50", "Natural bid/ask, 50% at the limit", 0.50, None, True),
    ("tp50_sl3", "50% limit + 3x stop", 0.50, 3.0, True),
]


def _payloads(setups: Sequence[Dict[str, Any]], trades, *, mode: str = "mid") -> List[Dict[str, Any]]:
    by_entry = {t.entry_date: t for t in trades if not t.skipped and t.exit_date}
    rows = []
    for setup in setups:
        if setup.get("skip"):
            continue
        trade = by_entry.get(setup["entry_date"])
        if trade is None or trade.exit_date is None:
            continue
        credit = float(trade.credit_pts) * 100.0
        marks = []
        for mark in setup.get("marks") or []:
            day = mark["day"]
            if day <= trade.entry_date or day >= trade.exit_date:
                continue
            long_px, short_px = transacted_prices(mark, mode=mode, leg_slip=0.0, closing=True)
            pnl_pts = float(trade.credit_pts) - package_value(long_px, short_px)
            marks.append((day, pnl_pts * 100.0))
        rows.append(
            {
                "entry": trade.entry_date,
                "exit": trade.exit_date,
                "realized": float(trade.pnl_dollars),
                "credit": credit,
                "marks": marks,
            }
        )
    return rows


def _pack(curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dates": [row["date"] for row in curve],
        "equity": [row["equity"] for row in curve],
        "open": [row["open"] for row in curve],
        "n": [row["n"] for row in curve],
        "closed": [row["closed"] for row in curve],
        "margin": [row["margin"] for row in curve],
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
    books: Dict[str, Any] = {}
    for name, label, frac, stop, at_limit in BOOKS:
        mode = "natural" if name.startswith("natural") else "mid"
        trades = run_variant(
            setups,
            spot_by_day,
            {
                "mode": mode,
                "leg_slip": 0.0,
                "tp_frac": frac,
                "exit_dte": None,
                "stop_credit_mult": stop,
                "tp_at_limit": at_limit,
            },
        )
        curve = daily_book_path(_payloads(setups, trades, mode=mode))
        packed = _pack(curve)
        packed["label"] = label
        books[name] = packed
        if curve:
            worst_i = min(range(len(curve)), key=lambda i: curve[i]["equity"])
            margin_i = max(range(len(curve)), key=lambda i: curve[i]["margin"])
            LOG.info(
                "%s days=%s end=%s worst_open=%s on %s max_margin=%s on %s max_n=%s",
                name,
                len(curve),
                curve[-1]["equity"],
                curve[worst_i]["open"],
                curve[worst_i]["date"],
                curve[margin_i]["margin"],
                curve[margin_i]["date"],
                max(row["n"] for row in curve),
            )
    out = {
        "fee_per_package": 3.90,
        "spx_mult": 100,
        "mes_mult": 5,
        "books": books,
    }
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(out), encoding="utf-8")
    print(DEFAULT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
