#!/usr/bin/env python3
"""
Backfill IB daily (1d) history as a prefix for ALPACA symbols that start late.

Writes provider='IB' rows into TimescaleDB (does NOT overwrite ALPACA).
Research loaders can then stitch with:
  load_ohlcv_many(..., provider='ALPACA', fallback_provider='IB', merge_mode='prefix')

Requires IB Gateway. Auto-detects the API port. Default client id is 8827
so this job does not sit on live 15m (8826) or the 15m/5m universe backfills
(8822/8823). Do not run while those clients are connected.

A symbol is selected when the earliest stored 1d bar (ALPACA or IB) is after
--start. The 2026-08-26 run used 2018-11-01, so names that already have an IB
prefix back to 2018 stay skipped until --start moves earlier. Twenty years
back from 2026 is --start 2006-01-01 (also the new default). Each request is
at most --chunk-years (10) and walks backward from the first stored bar.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_daily_prefix.py --limit 5 --start 2018-11-01
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\data\\backfill_ib_daily_prefix.py --start 2018-11-01 --sleep 0.25
"""
from __future__ import annotations

import argparse
import signal
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
    _ib_end_datetime_utc,
    cleanup_ib_connection,
    create_ib_contract_with_primary_exchange,
    fetch_max_from_ib,
    get_ib_connection,
    prepare_nautilus_dataframe,
    set_ib_client_id,
)
from utils.research.ib_daily_prefix import (  # noqa: E402
    DEFAULT_CHUNK_YEARS,
    PidLock,
    duration_str as _duration_str,
    in_rth_yield_window,
    iter_prefix_chunks,
    make_should_stop,
    next_rth_yield_dt,
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
STOP_FILE = ROOT / "logs" / "data" / "ib_1d_prefix.stop"
LOCK_FILE = ROOT / "logs" / "data" / "ib_1d_prefix.lock"
CHUNK_YEARS = DEFAULT_CHUNK_YEARS
CHUNK_SLEEP = 0.0
SHOULD_STOP = lambda: False


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
    Return (symbol, earliest_stored_ts) for ALPACA 1d names whose earliest
    stored 1d bar (ALPACA or IB) is after target_start.
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
        ),
        ib AS (
            SELECT symbol, MIN(ts) AS imin
            FROM market_data
            WHERE provider = 'IB' AND timeframe = '1d'
            GROUP BY symbol
        )
        SELECT a.symbol, LEAST(a.amin, COALESCE(i.imin, a.amin)) AS earliest
        FROM alp a
        LEFT JOIN ib i ON i.symbol = a.symbol
        WHERE LEAST(a.amin, COALESCE(i.imin, a.amin)) > %s
        ORDER BY earliest ASC, a.symbol ASC
    """
    cur = client.connection.cursor()
    try:
        cur.execute(sql, (target_start,))
        rows = cur.fetchall()
    finally:
        cur.close()
    # 2006-01-01 is a Sunday and the 2nd was a holiday, so a filled name
    # starts 2006-01-03. Treat the first 10 days as done or the next run
    # requests every finished symbol again.
    grace_end = target_start + timedelta(days=10)
    rows = [r for r in rows if r[1] is not None and r[1] > grace_end]
    if no_gap:
        rows = [r for r in rows if str(r[0]).upper() not in no_gap]
    if min_alpaca_start is not None:
        rows = [r for r in rows if r[1] >= min_alpaca_start]
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    return [(str(s), ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts) for s, ts in rows]


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
    """Map symbol -> earliest stored 1d ts across ALPACA and IB."""
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
            WHERE provider IN ('ALPACA', 'IB') AND timeframe = '1d' AND symbol = ANY(%s)
            GROUP BY symbol
            """,
            (symbols,),
        )
        return {str(s): (ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts) for s, ts in cur.fetchall()}
    finally:
        cur.close()


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_ib_daily_window(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    end_now: bool,
) -> Optional[pd.DataFrame]:
    """One IB daily request trimmed to [start, end).

    None means try again later (timeout, or a reply that never reached this
    window). An empty frame means IB returned history and none of it belongs
    in the window. raw_min / raw_max are stored on the frame attrs.
    """
    if end <= start:
        return pd.DataFrame()

    ib = get_ib_connection()
    try:
        ib.RequestTimeout = 180
    except Exception:
        pass
    contract = _qualify(symbol)
    if contract is None:
        logger.warning("%s: no qualified IB contract", symbol)
        return None

    start_utc = _as_utc(start)
    end_utc = _as_utc(end)
    if end_now:
        end_str = ""
        dur = _duration_str(start_utc, datetime.now(timezone.utc))
    else:
        request_end = end_utc + timedelta(days=1)
        end_str = _ib_end_datetime_utc(request_end)
        dur = _duration_str(start_utc, request_end)
    logger.info(
        "%s: IB 1d %s -> %s dur=%s end=%s",
        symbol, start_utc.date(), end_utc.date(), dur, end_str or "now",
    )

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_str,
            durationStr=dur,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
            timeout=180,
        )
    except Exception as exc:
        logger.error("%s: IB historical request failed: %s", symbol, exc)
        return None
    if not bars:
        logger.warning("%s: IB returned 0 bars (dur=%s); will retry, not a no-gap", symbol, dur)
        return None

    batch_df = pd.DataFrame([b.__dict__ for b in bars])[["date", "open", "high", "low", "close", "volume"]]
    batch_df.rename(columns={"date": "timestamp"}, inplace=True)
    batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"])
    if batch_df["timestamp"].dt.tz is None:
        batch_df["timestamp"] = batch_df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    else:
        batch_df["timestamp"] = batch_df["timestamp"].dt.tz_convert("UTC")

    raw_min = batch_df["timestamp"].min()
    raw_max = batch_df["timestamp"].max()
    trimmed = batch_df[(batch_df["timestamp"] >= start_utc) & (batch_df["timestamp"] < end_utc)].copy()
    if trimmed.empty:
        logger.warning(
            "%s: %d IB bars %s..%s but none in [%s, %s)",
            symbol, len(batch_df), raw_min.date(), raw_max.date(), start_utc.date(), end_utc.date(),
        )
        empty = pd.DataFrame()
        empty.attrs["raw_min"] = raw_min
        empty.attrs["raw_max"] = raw_max
        return empty
    out = prepare_nautilus_dataframe(trimmed, symbol, "IB", "1d")
    if out is not None:
        out.attrs["raw_min"] = raw_min
        out.attrs["raw_max"] = raw_max
    return out


def _insert_prefix(symbol: str, df: pd.DataFrame, client) -> int:
    if df is None or df.empty:
        return 0
    if not client.insert_market_data(df, symbol, "IB", "1d"):
        logger.error("%s: insert_market_data failed (%d bars)", symbol, len(df))
        return 0
    ts0 = pd.Timestamp(int(df["ts_event"].min()), unit="ns", tz="UTC")
    ts1 = pd.Timestamp(int(df["ts_event"].max()), unit="ns", tz="UTC")
    logger.info("%s: saved %d IB 1d bars (%s -> %s)", symbol, len(df), ts0.date(), ts1.date())
    return int(len(df))


def _backfill_chunks(symbol: str, start: datetime, end: datetime, client) -> int:
    """Fill [start, end).

    One request ending now covers the 20-year gap. A dated endDateTime often
    times out, and an empty reply is a retry, not a permanent no-gap. No-gap
    is only when a now-request's oldest bar is the history already stored.
    """
    start_utc = _as_utc(start)
    end_utc = _as_utc(end)
    if SHOULD_STOP():
        return 0
    df = fetch_ib_daily_window(symbol, start_utc, end_utc, end_now=True)
    if df is None:
        return 0
    if df.empty:
        raw_min = df.attrs.get("raw_min")
        if raw_min is not None and raw_min > end_utc + timedelta(days=7):
            logger.warning(
                "%s: IB reply starts %s, after stored %s; trying a dated end",
                symbol, raw_min.date(), end_utc.date(),
            )
            if SHOULD_STOP():
                return 0
            older = fetch_ib_daily_window(symbol, start_utc, end_utc, end_now=False)
            if older is None or older.empty:
                logger.info("%s: dated request also missed [%s, %s); will retry", symbol, start_utc.date(), end_utc.date())
                return 0
            return _insert_prefix(symbol, older, client)
        _mark_no_gap(symbol)
        logger.info("%s: no IB bars before %s", symbol, end_utc.date())
        return 0
    saved = _insert_prefix(symbol, df, client)
    if saved <= 0:
        return 0
    oldest = pd.Timestamp(int(df["ts_event"].min()), unit="ns", tz="UTC").to_pydatetime()
    if oldest <= start_utc + timedelta(days=10) or SHOULD_STOP():
        return saved
    older = fetch_ib_daily_window(symbol, start_utc, oldest, end_now=False)
    if older is None or older.empty:
        logger.info("%s: older prefix before %s not in this reply; will retry", symbol, oldest.date())
        return saved
    saved += _insert_prefix(symbol, older, client)
    return saved


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
        return _backfill_chunks(symbol, start, end, client)
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
    ap.add_argument("--start", default="2006-01-01", help="Target history start (YYYY-MM-DD). 2006-01-01 is 20 years back from 2026")
    ap.add_argument("--end", default=None, help="Optional hard end date (else Alpaca min per symbol)")
    ap.add_argument("--limit", type=int, default=None, help="Max symbols (smoke test)")
    ap.add_argument("--symbols", default="", help="Comma list override")
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Pause between IB requests (pacing; default 1s)",
    )
    ap.add_argument(
        "--full-history",
        action="store_true",
        help="Use slow fetch_max_from_ib walk instead of chunked prefix",
    )
    ap.add_argument(
        "--chunk-years",
        type=int,
        default=DEFAULT_CHUNK_YEARS,
        help="Max years per IB request (default 10)",
    )
    ap.add_argument(
        "--client-id",
        type=int,
        default=8827,
        help="IB API client id (default 8827; keep off 8822/8823/8826)",
    )
    ap.add_argument("--dry-run", action="store_true", help="List symbols only")
    ap.add_argument("--stop-file", default=str(STOP_FILE), help="Stop after the current symbol when this file exists")
    ap.add_argument("--lock-file", default=str(LOCK_FILE), help="Pid lock so the hourly watchdog does not double-connect")
    ap.add_argument(
        "--skip-if-job-running",
        default="",
        help="Exit 0 if this CronRunner job still holds its lock",
    )
    ap.add_argument(
        "--allow-rth",
        action="store_true",
        help="Allow running during Mon-Fri 09:15-16:30 ET (default: stop)",
    )
    ap.add_argument(
        "--no-until",
        action="store_true",
        help="Do not stop at the next weekday 09:15 ET",
    )
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
        print(f"  {s} stored_min={amin}")
    if len(needed) > 10:
        print(f"  ... +{len(needed)-10} more")

    if args.dry_run:
        return 0

    if str(args.skip_if_job_running).strip():
        from utils.cron.manager import CronManager

        alive, pid = CronManager(root=ROOT).lock_status(str(args.skip_if_job_running).strip())
        if alive:
            logger.info("Skip: CronRunner job %s still running (pid %s)", args.skip_if_job_running, pid)
            return 0

    global CHUNK_YEARS, CHUNK_SLEEP, SHOULD_STOP
    CHUNK_YEARS = max(1, int(args.chunk_years))
    CHUNK_SLEEP = float(args.sleep or 0.0)
    set_ib_client_id(int(args.client_id))

    stop_file = Path(args.stop_file)
    until = None if args.no_until else next_rth_yield_dt()
    flag = {"stop": False}

    def _on_signal(signum, _frame):
        logger.info("Signal %s: will stop after the current symbol", signum)
        flag["stop"] = True

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, _on_signal)
        except (OSError, ValueError):
            pass
    SHOULD_STOP = make_should_stop(
        stop_file=stop_file,
        until=until,
        allow_rth=bool(args.allow_rth),
        flag=flag,
    )
    if SHOULD_STOP():
        why = "RTH yield window" if in_rth_yield_window() else "stop file or --until"
        logger.info("Not starting (%s)", why)
        return 0
    if until is not None:
        logger.info("Will stop at %s", until.isoformat())

    lock = PidLock(Path(args.lock_file), "backfill_ib_daily_prefix.py")
    if not lock.acquire():
        logger.info("Another 1d prefix backfill holds the lock; nothing to do")
        return 0

    saved = 0
    failed = 0
    bars_total = 0
    stopped = False
    t_run = time.perf_counter()
    try:
        get_ib_connection()
        for i, (sym, amin) in enumerate(needed, start=1):
            if SHOULD_STOP():
                stopped = True
                logger.info("Stopping before %s (%d/%d)", sym, i, len(needed))
                break
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
        lock.release()

    print(f"\nDone: saved_symbols={saved} failed={failed} bars={bars_total} stopped={stopped}")
    if failed and not stopped:
        logger.info(
            "%d symbols saved nothing this pass; the next run retries names still short of the start",
            failed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
