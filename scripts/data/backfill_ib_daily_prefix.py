#!/usr/bin/env python3
"""
Backfill IB daily (1d) history as a prefix for ALPACA symbols that start late.

Writes provider='IB' rows into TimescaleDB (does NOT overwrite ALPACA).
Research loaders can then stitch with:
  load_ohlcv_many(..., provider='ALPACA', fallback_provider='IB', merge_mode='prefix')

Requires IB Gateway on 127.0.0.1:4001.

Fast path (default): one reqHistoricalData per symbol ending at Alpaca start
(prefix only). Avoids fetch_max_from_ib's full-history walk.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_daily_prefix.py --limit 5 --start 2018-11-01
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_daily_prefix.py --start 2018-11-01 --sleep 0.25
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import (  # noqa: E402
    cleanup_ib_connection,
    create_ib_contract_with_primary_exchange,
    fetch_max_from_ib,
    get_ib_connection,
    prepare_nautilus_dataframe,
)
from utils.db.timescaledb_client import get_timescaledb_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_ib_daily_prefix")
logging.getLogger("utils.data.fetch_data").setLevel(logging.WARNING)
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)
logging.getLogger("ib_insync").setLevel(logging.WARNING)

_CONTRACT_CACHE = {}
NO_GAP_FILE = ROOT / "logs" / "data" / "ib_prefix_no_gap_symbols.txt"


def _load_no_gap_symbols() -> set:
    if not NO_GAP_FILE.exists():
        return set()
    return {ln.strip().upper() for ln in NO_GAP_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _mark_no_gap(symbol: str) -> None:
    NO_GAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_no_gap_symbols()
    sym = symbol.strip().upper()
    if sym in existing:
        return
    with NO_GAP_FILE.open("a", encoding="utf-8") as fh:
        fh.write(sym + "\n")


def list_symbols_needing_prefix(
    *,
    target_start: datetime,
    min_alpaca_start: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Tuple[str, datetime]]:
    """
    Return (symbol, alpaca_min_ts) for ALPACA 1d names whose history starts after target_start.
    Prefer symbols that have little/no IB 1d before alpaca_min.
    """
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB connection failed")
    no_gap = _load_no_gap_symbols()
    sql = """
        WITH alp AS (
            SELECT symbol, MIN(ts) AS amin
            FROM market_data
            WHERE provider = 'ALPACA' AND timeframe = '1d'
            GROUP BY symbol
            HAVING MIN(ts) > %s
        ),
        ib AS (
            SELECT symbol, MIN(ts) AS imin
            FROM market_data
            WHERE provider = 'IB' AND timeframe = '1d'
            GROUP BY symbol
        )
        SELECT a.symbol, a.amin
        FROM alp a
        LEFT JOIN ib i ON i.symbol = a.symbol
        -- Need fill when no IB yet, or IB starts at/after Alpaca (no usable prefix)
        WHERE i.symbol IS NULL OR i.imin >= a.amin
        ORDER BY a.amin ASC, a.symbol ASC
    """
    cur = client.connection.cursor()
    try:
        cur.execute(sql, (target_start,))
        rows = cur.fetchall()
    finally:
        cur.close()
    if no_gap:
        rows = [r for r in rows if str(r[0]).upper() not in no_gap]
    if min_alpaca_start is not None:
        rows = [r for r in rows if r[1] >= min_alpaca_start]
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    return [(str(s), ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts) for s, ts in rows]


def _duration_str(start: datetime, end: datetime) -> str:
    days = max(1, (end - start).days + 7)  # small pad
    if days > 365:
        years = max(1, (days + 364) // 365)
        return f"{years} Y"
    return f"{days} D"


def _qualify(symbol: str):
    if symbol in _CONTRACT_CACHE:
        return _CONTRACT_CACHE[symbol]
    ib = get_ib_connection()
    contract = create_ib_contract_with_primary_exchange(symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        return None
    _CONTRACT_CACHE[symbol] = qualified[0]
    return qualified[0]


def _lookup_alpaca_min(symbols: List[str]) -> dict:
    """Map symbol -> earliest ALPACA 1d ts (or empty)."""
    if not symbols:
        return {}
    client = get_timescaledb_client()
    if not client.ensure_connection():
        return {}
    cur = client.connection.cursor()
    try:
        cur.execute(
            """
            SELECT symbol, MIN(ts)
            FROM market_data
            WHERE provider = 'ALPACA' AND timeframe = '1d' AND symbol = ANY(%s)
            GROUP BY symbol
            """,
            (symbols,),
        )
        return {str(s): (ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts) for s, ts in cur.fetchall()}
    finally:
        cur.close()


def fetch_ib_daily_oneshot(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """
    Single historical request for daily bars in [start, end).

    Uses endDateTime='' (now) + duration back through `start`, then trims to the
    Alpaca prefix. Asking IB for a mid-history endDateTime is flaky (0 bars /
    timeouts on many names).
    """
    if end <= start:
        logger.warning("%s: empty window start=%s end=%s", symbol, start, end)
        return None

    ib = get_ib_connection()
    contract = _qualify(symbol)
    if contract is None:
        logger.warning("%s: no qualified IB contract", symbol)
        return None

    start_utc = start if start.tzinfo is not None else start.replace(tzinfo=timezone.utc)
    end_utc = end if end.tzinfo is not None else end.replace(tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    dur = _duration_str(start_utc, now_utc)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=dur,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        logger.warning("%s: IB returned 0 bars (dur=%s)", symbol, dur)
        return None

    batch_df = pd.DataFrame([b.__dict__ for b in bars])[["date", "open", "high", "low", "close", "volume"]]
    batch_df.rename(columns={"date": "timestamp"}, inplace=True)
    batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"])
    if batch_df["timestamp"].dt.tz is None:
        batch_df["timestamp"] = batch_df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    else:
        batch_df["timestamp"] = batch_df["timestamp"].dt.tz_convert("UTC")

    before = len(batch_df)
    batch_df = batch_df[(batch_df["timestamp"] >= start_utc) & (batch_df["timestamp"] < end_utc)]
    if batch_df.empty:
        logger.warning(
            "%s: %d IB bars but none in [%s, %s) -- no prefix gap",
            symbol,
            before,
            start_utc.date(),
            end_utc.date(),
        )
        _mark_no_gap(symbol)
        return None
    return prepare_nautilus_dataframe(batch_df, symbol, "IB", "1d")


def backfill_symbol(
    symbol: str,
    start: datetime,
    end: Optional[datetime] = None,
    *,
    oneshot: bool = True,
) -> int:
    """Fetch IB daily and insert. Returns bars saved."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("TimescaleDB connection failed")

    if oneshot and end is not None:
        df = fetch_ib_daily_oneshot(symbol, start, end)
    else:
        df = fetch_max_from_ib(symbol, "1d", start_date=start)
        if df is not None and not df.empty and end is not None:
            end_utc = end if end.tzinfo is not None else end.replace(tzinfo=timezone.utc)
            end_ns = int(pd.Timestamp(end_utc).value)
            df = df.loc[df["ts_event"].astype("int64") < end_ns].copy()

    if df is None or df.empty:
        logger.warning("%s: no IB daily bars", symbol)
        return 0

    if "ts_event" not in df.columns:
        logger.error("%s: unexpected IB frame columns %s", symbol, list(df.columns))
        return 0

    ok = client.insert_market_data(df, symbol, "IB", "1d")
    if not ok:
        logger.error("%s: insert_market_data failed (%d bars)", symbol, len(df))
        return 0

    ts0 = pd.Timestamp(int(df["ts_event"].min()), unit="ns", tz="UTC")
    ts1 = pd.Timestamp(int(df["ts_event"].max()), unit="ns", tz="UTC")
    logger.info("%s: saved %d IB 1d bars (%s -> %s)", symbol, len(df), ts0.date(), ts1.date())
    return int(len(df))


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill IB 1d prefix for late-starting ALPACA symbols")
    ap.add_argument("--start", default="2018-11-01", help="Target history start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Optional hard end date (else Alpaca min per symbol)")
    ap.add_argument("--limit", type=int, default=None, help="Max symbols (smoke test)")
    ap.add_argument("--symbols", default="", help="Comma list override")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Pause between IB symbols (pacing; default 0.25)",
    )
    ap.add_argument(
        "--full-history",
        action="store_true",
        help="Use slow fetch_max_from_ib walk instead of one-shot prefix",
    )
    ap.add_argument("--dry-run", action="store_true", help="List symbols only")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cli_end = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else None
    )

    if args.symbols.strip():
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        amin_map = _lookup_alpaca_min(syms)
        needed = []
        for s in syms:
            amin = amin_map.get(s)
            if amin is None:
                # No Alpaca row: fetch through cli_end or "now"
                amin = cli_end or datetime.now(timezone.utc)
            needed.append((s, amin))
    else:
        logger.info("Finding ALPACA 1d symbols starting after %s ...", args.start)
        t0 = time.perf_counter()
        needed = list_symbols_needing_prefix(target_start=start, limit=args.limit)
        logger.info("Found %d symbols needing IB prefix (%.1fs)", len(needed), time.perf_counter() - t0)

    if not needed:
        print("No symbols need IB prefix backfill")
        return 0

    oneshot = not args.full_history
    print(f"Will process {len(needed)} symbols (dry_run={args.dry_run}, oneshot={oneshot}, sleep={args.sleep})")
    for s, amin in needed[:10]:
        print(f"  {s} alpaca_min={amin}")
    if len(needed) > 10:
        print(f"  ... +{len(needed)-10} more")

    if args.dry_run:
        return 0

    saved = 0
    failed = 0
    bars_total = 0
    t_run = time.perf_counter()
    try:
        # Warm IB connection once
        get_ib_connection()
        for i, (sym, amin) in enumerate(needed, start=1):
            end = cli_end if cli_end is not None else amin
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            try:
                n = backfill_symbol(sym, start, end=end, oneshot=oneshot)
                if n > 0:
                    saved += 1
                    bars_total += n
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                logger.error("%s failed: %s", sym, exc)
            if args.sleep and i < len(needed):
                time.sleep(float(args.sleep))
            if i % 25 == 0:
                elapsed = time.perf_counter() - t_run
                rate = i / elapsed if elapsed > 0 else 0.0
                eta_min = ((len(needed) - i) / rate / 60.0) if rate > 0 else float("nan")
                logger.info(
                    "Progress %d/%d saved=%d failed=%d bars=%d rate=%.2f/s eta=%.0fmin",
                    i,
                    len(needed),
                    saved,
                    failed,
                    bars_total,
                    rate,
                    eta_min,
                )
    finally:
        try:
            cleanup_ib_connection()
        except Exception:
            pass

    print(f"\nDone: saved_symbols={saved} failed={failed} bars={bars_total}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
