#!/usr/bin/env python3
"""QQQ LEAPS gap scanner for /hot + Telegram (timeframe=lg).

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\qqq_leaps_gap_live.py
  python scripts\\scanners\\qqq_leaps_gap_live.py --dry-run
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
from utils.scanning.qqq_leaps_gap_live import (
    SYMBOL,
    TIMEFRAME_LG,
    candidate_db_row,
    evaluate_live,
    fetch_alpaca_qqq_marks,
    in_watch_window,
    load_qqq_daily,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qqq_leaps_gap_live")


def run(*, dry_run: bool = False, no_telegram: bool = False) -> int:
    client = get_timescaledb_client()
    store = ChannelTouchCandidatesStore(client)
    settings = store.load_settings()

    daily = load_qqq_daily(client)
    if daily.empty:
        logger.error("No QQQ 1d bars in TimescaleDB")
        return 1

    marks = fetch_alpaca_qqq_marks()
    last = marks.get("last")
    session_open = marks.get("open")
    logger.info(
        "QQQ last=%s open=%s daily_bars=%d watch=%s",
        last,
        session_open,
        len(daily),
        in_watch_window(),
    )

    existing = [
        r
        for r in store.load_rows()
        if str(r.get("stock") or "").upper() == SYMBOL
        and str(r.get("timeframe") or "") == TIMEFRAME_LG
    ]
    prev: Optional[Dict[str, Any]] = existing[0] if existing else None

    row, alerts = evaluate_live(
        daily,
        last_price=last,
        session_open=session_open,
        prev_row=prev,
    )
    if not row:
        logger.info("No LG row (idle / insufficient history)")
        if not dry_run:
            # Keep sticky rows only while watch window; else clear
            if not in_watch_window():
                store.replace_candidates([], timeframe=TIMEFRAME_LG)
        return 0

    notify = str(row.get("hot_notified_on") or "")
    st = str(row.get("status") or "")
    watching = notify.endswith(":approach") or notify.endswith(":arm")
    keep = st in ("armed", "waiting") and (
        watching or bool(row.get("hot")) or in_watch_window()
    )
    db_rows = [candidate_db_row(row)] if keep else []

    logger.info(
        "LG status=%s score=%s buy_ok=%s hot=%s gap=%s last=%.2f target=%s keep=%s alerts=%d",
        row.get("status"),
        row.get("wait_bars"),
        row.get("wait_ok"),
        row.get("hot"),
        row.get("overshoot_prior"),
        float(row.get("last_price") or 0),
        row.get("resist"),
        keep,
        len(alerts),
    )

    if dry_run:
        for a in alerts:
            logger.info("[dry-run telegram %s] %s", a.get("kind"), a.get("text"))
        return 0

    store.replace_candidates(db_rows, timeframe=TIMEFRAME_LG)

    from utils.scanning.channel_touch_notify_prefs import want_notify

    if no_telegram:
        return 0

    for a in alerts:
        kind = a.get("kind")
        text = a.get("text") or ""
        if kind == "arm":
            event = "armed"
        elif kind == "approach":
            # soft near-trigger = approaching + hot prefs
            event = "approaching"
        else:
            continue
        want_tg = want_notify(settings, "lg", event, "telegram")
        want_desk = want_notify(settings, "lg", event, "browser")
        # Also honor hot prefs for approach (getting-close)
        if kind == "approach":
            want_tg = want_tg or want_notify(settings, "lg", "hot", "telegram")
            want_desk = want_desk or want_notify(settings, "lg", "hot", "browser")
        if not want_tg and not want_desk:
            continue
        try:
            send_alert(text, desktop=want_desk, telegram=want_tg)
            logger.info("Alert sent (%s tg=%s desk=%s)", kind, want_tg, want_desk)
        except Exception as exc:
            logger.warning("Alert failed: %s", exc)

    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="QQQ LEAPS gap /hot scanner")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run(dry_run=args.dry_run, no_telegram=args.no_telegram))


if __name__ == "__main__":
    main()
