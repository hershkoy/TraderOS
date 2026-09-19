#!/usr/bin/env python3
"""Deep Pullback SPY scanner for /hot + Telegram.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\scanners\\deep_pullback_spy.py
  python scripts\\scanners\\deep_pullback_spy.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.db.timescaledb_client import get_timescaledb_client
from utils.notify.alerts import send_alert
from utils.scanning.channel_touch_15m import fetch_alpaca_last_prices
from utils.scanning.channel_touch_candidates_store import ChannelTouchCandidatesStore
from utils.scanning.deep_pullback_spy import (
    SYMBOL,
    TIMEFRAME_DP,
    candidate_db_row,
    evaluate_live,
    load_spy_daily,
    mark_entered,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deep_pullback_spy")


def _open_bought(store: ChannelTouchCandidatesStore) -> Optional[Dict[str, Any]]:
    for t in store.load_bought(active_only=True):
        if str(t.get("stock") or "").upper() == SYMBOL and str(t.get("timeframe") or "") == TIMEFRAME_DP:
            return t
    return None


def run(*, dry_run: bool = False, no_telegram: bool = False) -> int:
    client = get_timescaledb_client()
    store = ChannelTouchCandidatesStore(client)
    settings = store.load_settings()

    daily = load_spy_daily(client)
    if daily.empty:
        logger.error("No SPY 1d bars in TimescaleDB")
        return 1

    prices = fetch_alpaca_last_prices([SYMBOL], batch_size=1)
    last = prices.get(SYMBOL)
    logger.info("SPY last=%s daily_bars=%d", last, len(daily))

    existing = [
        r
        for r in store.load_rows()
        if str(r.get("stock") or "").upper() == SYMBOL
        and str(r.get("timeframe") or "") == TIMEFRAME_DP
    ]
    prev = existing[0] if existing else None

    bought = _open_bought(store)
    if bought and prev is not None:
        prev = mark_entered(
            prev,
            entry_px=float(bought.get("entry_px") or bought.get("last_price") or 0),
            entry_ts=bought.get("entry_ts") or datetime.now(timezone.utc),
        )
        prev["status"] = "filled"
    elif bought and prev is None:
        prev = {
            "status": "filled",
            "fill_px": float(bought.get("entry_px") or 0),
            "h2_time": bought.get("entry_ts"),
        }

    row, alerts = evaluate_live(daily, last_price=last, prev_row=prev)
    if not row:
        logger.info("No DP row (insufficient history)")
        if not dry_run:
            store.replace_candidates([], timeframe=TIMEFRAME_DP)
        return 0

    # Keep after approaching/arm so minute SPY pulls refresh current vs target
    feat_score = int(row.get("wait_bars") or 0)
    st = str(row.get("status") or "")
    notify = str(row.get("hot_notified_on") or "")
    watching = notify.endswith(":approach") or notify.endswith(":arm")
    keep = st in ("armed", "filled", "exit_signal") or (
        st == "waiting" and (feat_score >= 2 or watching)
    ) or watching
    db_rows = [candidate_db_row(row)] if keep else []

    logger.info(
        "DP status=%s score=%s buy_ok=%s hot=%s last=%.2f target=%s keep=%s alerts=%d",
        row.get("status"),
        row.get("wait_bars"),
        row.get("wait_ok"),
        row.get("hot"),
        float(row.get("last_price") or 0),
        row.get("resist"),
        keep,
        len(alerts),
    )

    if dry_run:
        for a in alerts:
            logger.info("[dry-run telegram %s] %s", a.get("kind"), a.get("text"))
        return 0

    store.replace_candidates(db_rows, timeframe=TIMEFRAME_DP)

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
        elif kind == "exit":
            event = "sell"
        else:
            continue
        want_tg = want_notify(settings, "dp", event, "telegram")
        want_desk = want_notify(settings, "dp", event, "browser")
        if not want_tg and not want_desk:
            continue
        try:
            send_alert(text, desktop=want_desk, telegram=want_tg)
            logger.info("Alert sent (%s tg=%s desk=%s)", kind, want_tg, want_desk)
        except Exception as exc:
            logger.warning("Alert failed: %s", exc)

    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep Pullback SPY /hot scanner")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run(dry_run=args.dry_run, no_telegram=args.no_telegram))


if __name__ == "__main__":
    main()
