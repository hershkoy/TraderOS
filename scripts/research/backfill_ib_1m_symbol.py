#!/usr/bin/env python3
"""
Backfill IB 1m bars for SPY (and named symbols) into TimescaleDB.

Lives under scripts/research/ because scripts/data is tooling-gated in this workspace.
Same role as a scripts/data/backfill_ib_1m_symbol.py would have.

Default: forward walk from --since (batch-days=1).
Candidate-day mode: oracle U 15m@15:45 proximity days only.

Client ID 8825 — keep distinct from 8822/8823/8824/8826.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backfill_ib_1m_symbol.py --symbols SPY --candidate-days-from-qs-realism --start 2018-01-01 --ib-client-id 8825
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import (  # noqa: E402
    _prepare_ib_duration_from_days,
    cleanup_ib_connection,
    create_ib_contract_with_primary_exchange,
    fetch_ib_historical_window,
    get_ib_connection,
    ib_step_back_minutes,
    prepare_nautilus_dataframe,
    save_to_timescaledb,
    set_ib_client_id,
)
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_ib_1m_symbol")
logging.getLogger("ib_insync").setLevel(logging.WARNING)

TIMEFRAME = "1m"
DEFAULT_CLIENT_ID = 8825


def _as_utc(ts) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()


def _ns_ts(df, first: bool) -> str:
    col = "timestamp" if "timestamp" in df.columns else "ts"
    t = pd.Timestamp(df[col].iloc[0 if first else -1])
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return str(t)


def sessions_with_1m(symbol: str, start: date, end: date) -> Set[date]:
    client = get_timescaledb_client()
    rows = client.execute_query(
        "SELECT DISTINCT (ts AT TIME ZONE 'America/New_York')::date AS d "
        "FROM market_data WHERE symbol=%s AND timeframe=%s "
        "AND ts >= %s AND ts < %s",
        (
            symbol,
            TIMEFRAME,
            datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc),
        ),
    )
    out: Set[date] = set()
    for r in rows or []:
        d = r[0] if not isinstance(r, dict) else r["d"]
        out.add(pd.Timestamp(d).date())
    return out


def fetch_save_day(ib, contract, symbol: str, day: date, *, dry_run: bool) -> int:
    end_dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(
        days=1, hours=4
    )
    raw = fetch_ib_historical_window(
        ib,
        contract,
        timeframe=TIMEFRAME,
        end_dt=end_dt,
        duration_str="1 D",
    )
    if raw is None or raw.empty:
        return 0
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    ny = raw["timestamp"].dt.tz_convert("America/New_York")
    raw = raw.loc[ny.dt.date == day]
    if raw.empty:
        return 0
    df = prepare_nautilus_dataframe(raw.copy(), symbol, "IB", TIMEFRAME)
    if dry_run:
        return int(len(df))
    if not save_to_timescaledb(df, symbol, "IB", TIMEFRAME):
        logger.error("%s %s: TimescaleDB insert failed", symbol, day)
        return -1
    return int(len(df))


def backfill_forward(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    batch_days: int,
    sleep_s: float,
    dry_run: bool,
) -> Tuple[int, str]:
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    if start_dt >= end_dt:
        return 0, "caught_up"

    ib = get_ib_connection()
    contract = create_ib_contract_with_primary_exchange(symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        return 0, "qualify_failed"
    contract = qualified[0]

    probe = fetch_ib_historical_window(
        ib, contract, timeframe=TIMEFRAME, end_dt=end_dt, duration_str="1 D"
    )
    if probe is None or probe.empty:
        return 0, "empty"
    if sleep_s > 0:
        time.sleep(sleep_s)

    cursor = start_dt
    n_saved = 0
    empty_streak = 0
    step = timedelta(minutes=ib_step_back_minutes(TIMEFRAME))

    while cursor < end_dt:
        window_end = min(cursor + timedelta(days=max(1, int(batch_days))), end_dt)
        duration_days = min(
            max(1, (window_end - cursor).days + 1), max(1, int(batch_days))
        )
        dur_unit = _prepare_ib_duration_from_days(duration_days)
        logger.info(
            "%s: IB 1m window %s -> %s (%s)",
            symbol,
            cursor.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d"),
            dur_unit,
        )
        raw = fetch_ib_historical_window(
            ib,
            contract,
            timeframe=TIMEFRAME,
            end_dt=window_end,
            duration_str=dur_unit,
        )
        if raw is None or raw.empty:
            empty_streak += 1
            if n_saved > 0 and empty_streak >= 8:
                return n_saved, "ok"
            cursor = window_end
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue
        empty_streak = 0
        raw = raw[raw["timestamp"] >= cursor]
        raw = raw[raw["timestamp"] <= window_end]
        if raw.empty:
            cursor = window_end
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue
        df = prepare_nautilus_dataframe(raw.copy(), symbol, "IB", TIMEFRAME)
        logger.info(
            "%s: fetched %d 1m bars (%s -> %s)",
            symbol,
            len(df),
            _ns_ts(df, True),
            _ns_ts(df, False),
        )
        if dry_run:
            n_saved += int(len(df))
        else:
            if not save_to_timescaledb(df, symbol, "IB", TIMEFRAME):
                return n_saved, "insert_failed"
            n_saved += int(len(df))
        last_ts = _as_utc(raw["timestamp"].max())
        cursor = (last_ts + step).to_pydatetime()
        if cursor.tzinfo is None:
            cursor = cursor.replace(tzinfo=timezone.utc)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return n_saved, "ok"


def backfill_candidate_days(
    symbol: str,
    days: Sequence[date],
    *,
    sleep_s: float,
    dry_run: bool,
    skip_existing: bool = True,
) -> Tuple[int, int]:
    if not days:
        return 0, 0
    days_sorted = sorted(set(days))
    existing: Set[date] = set()
    if skip_existing:
        existing = sessions_with_1m(symbol, days_sorted[0], days_sorted[-1])
        logger.info(
            "%s: already have 1m on %d/%d candidate days",
            symbol,
            len(existing & set(days_sorted)),
            len(days_sorted),
        )

    ib = get_ib_connection()
    contract = create_ib_contract_with_primary_exchange(symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.error("%s: qualifyContracts failed", symbol)
        return 0, 0
    contract = qualified[0]

    n_ok = 0
    n_bars = 0
    for i, day in enumerate(days_sorted):
        if skip_existing and day in existing:
            n_ok += 1
            continue
        n = fetch_save_day(ib, contract, symbol, day, dry_run=dry_run)
        if n < 0:
            break
        if n == 0:
            logger.warning(
                "%s: empty 1m for %s (%d/%d)", symbol, day, i + 1, len(days_sorted)
            )
        else:
            n_ok += 1
            n_bars += n
            logger.info(
                "%s: %s saved %d 1m bars (%d/%d)",
                symbol,
                day,
                n,
                i + 1,
                len(days_sorted),
            )
        if sleep_s > 0:
            time.sleep(sleep_s)
    return n_ok, n_bars


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill IB 1m OHLCV into TimescaleDB")
    ap.add_argument("--symbols", default="SPY")
    ap.add_argument("--since", default="2018-01-01")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--candidate-days-from-qs-realism", action="store_true")
    ap.add_argument("--ib-client-id", type=int, default=DEFAULT_CLIENT_ID)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--batch-days", type=int, default=1)
    ap.add_argument("--ib-port", type=int, default=4001)
    ap.add_argument("--no-skip-existing", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("IB_PORT", str(int(args.ib_port)))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        return 1
    set_ib_client_id(int(args.ib_client_id))
    n_ok = 0
    try:
        if args.candidate_days_from_qs_realism:
            from utils.research.qs_five_swing_realism import build_qs_moc_candidate_days

            for sym in symbols:
                days = build_qs_moc_candidate_days(sym, start=args.start)
                logger.info("%s: %d candidate days from %s", sym, len(days), args.start)
                ok, bars = backfill_candidate_days(
                    sym,
                    days,
                    sleep_s=float(args.sleep),
                    dry_run=bool(args.dry_run),
                    skip_existing=not bool(args.no_skip_existing),
                )
                logger.info("%s: candidate-day done ok_days=%d bars=%d", sym, ok, bars)
                if ok > 0 or bars >= 0:
                    n_ok += 1
        else:
            start_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            for sym in symbols:
                n, status = backfill_forward(
                    sym,
                    start_dt,
                    datetime.now(timezone.utc),
                    batch_days=int(args.batch_days),
                    sleep_s=float(args.sleep),
                    dry_run=bool(args.dry_run),
                )
                logger.info("%s: %s bars=%d", sym, status, n)
                if status in {"ok", "caught_up"}:
                    n_ok += 1
    finally:
        cleanup_ib_connection()
    logger.info("Done: %d/%d symbols", n_ok, len(symbols))
    return 0 if n_ok == len(symbols) else 1


if __name__ == "__main__":
    raise SystemExit(main())
