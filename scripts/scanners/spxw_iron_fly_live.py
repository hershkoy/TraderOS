#!/usr/bin/env python3
"""SPXW 0DTE iron fly scanner for /hot + Telegram (timeframe=if).

Includes day filters (skip Wed, VIX open <20) and L5/S3/P2 sit-out.
Fridays: on ENTER NOW, auto-park paper BAG with --place --no-transmit.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\spxw_iron_fly_live.py
  python scripts\\scanners\\spxw_iron_fly_live.py --dry-run
  python scripts\\scanners\\spxw_iron_fly_live.py --no-auto-park
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.db.timescaledb_client import get_timescaledb_client
from utils.notify.alerts import send_alert
from utils.scanning.channel_touch_candidates_store import ChannelTouchCandidatesStore
from utils.scanning.spxw_iron_fly_live import (
    SYMBOL,
    TIMEFRAME_IF,
    candidate_db_row,
    evaluate_live,
    in_watch_window,
    run_friday_auto_park,
    should_auto_park_friday,
    today_et,
)
from utils.scanning.spxw_iron_fly_l5 import record_signal_day, load_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("spxw_iron_fly_live")


def run(
    *,
    dry_run: bool = False,
    no_telegram: bool = False,
    no_auto_park: bool = False,
) -> int:
    client = get_timescaledb_client()
    store = ChannelTouchCandidatesStore(client)
    settings = store.load_settings()

    existing = [
        r
        for r in store.load_rows()
        if str(r.get("stock") or "").upper() == SYMBOL
        and str(r.get("timeframe") or "") == TIMEFRAME_IF
    ]
    prev: Optional[Dict[str, Any]] = existing[0] if existing else None

    row, alerts = evaluate_live(prev_row=prev)
    if not row:
        logger.info("No IF row (idle / outside window)")
        if not dry_run and not in_watch_window():
            store.replace_candidates([], timeframe=TIMEFRAME_IF)
        return 0

    take = bool(row.get("_take")) if "_take" in row else bool(row.get("wait_ok") or row.get("hot"))
    # Persist L5 decision for today once we're in/near the window (no PnL yet).
    if not dry_run and in_watch_window():
        st = load_state()
        record_signal_day(st, today_et(), taken=bool(take), pnl=None)

    db_rows = [candidate_db_row(row)]
    logger.info(
        "IF status=%s hot=%s wait_ok=%s vix=%s l5=%s take=%s alerts=%d",
        row.get("status"),
        row.get("hot"),
        row.get("wait_ok"),
        row.get("support"),
        (row.get("_l5") or {}).get("reason") if isinstance(row.get("_l5"), dict) else row.get("_note"),
        take,
        len(alerts),
    )

    buy_ok = bool(row.get("wait_ok"))
    park_result: Optional[Dict[str, Any]] = None
    if not no_auto_park and should_auto_park_friday(buy_ok=buy_ok):
        park_result = run_friday_auto_park(dry_run=dry_run)
        logger.info("Friday auto-park status=%s", park_result.get("status"))
        if (
            park_result.get("status") == "parked_no_transmit"
            and not dry_run
        ):
            alerts.append(
                {
                    "kind": "arm",
                    "text": (
                        "SPXW iron fly Friday BAG parked in TWS (transmit=False). "
                        "Review legs/credit and click Transmit."
                    ),
                }
            )
        elif (
            park_result.get("status") not in ("dry_run", "skipped")
            and park_result.get("status") != "parked_no_transmit"
            and not dry_run
        ):
            alerts.append(
                {
                    "kind": "arm",
                    "text": (
                        f"SPXW iron fly Friday auto-park FAILED "
                        f"({park_result.get('status')}). "
                        f"Manual: python scripts\\trading\\spxw_iron_fly_combo.py "
                        f"--place --no-transmit"
                    ),
                }
            )

    if dry_run:
        for a in alerts:
            logger.info("[dry-run telegram %s] %s", a.get("kind"), a.get("text"))
        return 0

    store.replace_candidates(db_rows, timeframe=TIMEFRAME_IF)

    from utils.scanning.channel_touch_notify_prefs import want_notify

    if no_telegram:
        return 0

    for a in alerts:
        kind = a.get("kind")
        text = a.get("text") or ""
        if kind == "arm":
            event = "armed"
        elif kind == "approach":
            event = "approaching"
        else:
            continue
        want_tg = want_notify(settings, "if", event, "telegram")
        want_desk = want_notify(settings, "if", event, "browser")
        if kind == "approach":
            want_tg = want_tg or want_notify(settings, "if", "hot", "telegram")
            want_desk = want_desk or want_notify(settings, "if", "hot", "browser")
        if not want_tg and not want_desk:
            continue
        try:
            send_alert(text, desktop=want_desk, telegram=want_tg)
            logger.info("Alert sent (%s tg=%s desk=%s)", kind, want_tg, want_desk)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alert failed: %s", exc)

    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="SPXW iron fly /hot scanner")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument(
        "--no-auto-park",
        action="store_true",
        help="Disable Friday ENTER NOW --place --no-transmit auto-park",
    )
    args = ap.parse_args()
    raise SystemExit(
        run(
            dry_run=args.dry_run,
            no_telegram=args.no_telegram,
            no_auto_park=args.no_auto_park,
        )
    )


if __name__ == "__main__":
    main()
