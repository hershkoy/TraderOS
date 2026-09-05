"""IB market_data coverage without a full-table GROUP BY.

After a reboot TimescaleDB is healthy but cold. ``MIN/MAX(ts), COUNT(*)
GROUP BY symbol`` on IB 15m (~58M bars) exceeds a 120s statement_timeout.
The composite index ``(symbol, provider, timeframe, ts DESC)`` answers
first/last with per-symbol ``ORDER BY ts LIMIT 1``. Symbol discovery uses
``MAX(ts)`` plus ``DISTINCT symbol`` on the newest ~21 days so Timescale
can exclude old chunks (a skip-scan without a time bound merge-appends
every weekly chunk).

``first_ts`` is cached under ``logs/data/ib_{timeframe}_coverage_cache.json``
so later cold starts only refresh ``last_ts`` (newest chunks).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from utils.db.timescaledb_client import get_timescaledb_client

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = "IB"
DEFAULT_CACHE_DIR = ROOT / "logs" / "data"
ExecuteFn = Callable[[str, Sequence], Sequence]


RECENT_SYMBOL_DAYS = 21


def cache_path_for(timeframe: str, cache_dir: Optional[Path] = None) -> Path:
    folder = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return folder / ("ib_%s_coverage_cache.json" % str(timeframe).strip().lower())


def max_ts_sql() -> str:
    return "SELECT MAX(ts) FROM market_data WHERE provider = %s AND timeframe = %s"


def distinct_recent_symbols_sql() -> str:
    return (
        "SELECT DISTINCT symbol FROM market_data "
        "WHERE provider = %s AND timeframe = %s AND ts >= %s "
        "ORDER BY symbol"
    )


def edge_ts_sql(newest: bool) -> str:
    order = "DESC" if newest else "ASC"
    return (
        "SELECT ts FROM market_data "
        "WHERE symbol = %s AND provider = %s AND timeframe = %s "
        "ORDER BY ts " + order + " LIMIT 1"
    )


def range_edge_sql() -> str:
    return (
        "SELECT MIN(ts), MAX(ts) FROM market_data "
        "WHERE symbol = %s AND provider = %s AND timeframe = %s "
        "AND ts >= %s AND ts < %s"
    )


def load_first_ts_cache(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = raw.get("rows") if isinstance(raw, dict) else None
    if not isinstance(rows, dict):
        return {}
    out: Dict[str, str] = {}
    for sym, rec in rows.items():
        if not isinstance(rec, dict):
            continue
        first = rec.get("first_ts")
        if first:
            out[str(sym).upper()] = str(first)
    return out


def save_coverage_cache(path: Path, timeframe: str, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).upper()
        rows[sym] = {
            "first_ts": None if pd.isna(row.get("first_ts")) else str(row["first_ts"]),
            "last_ts": None if pd.isna(row.get("last_ts")) else str(row["last_ts"]),
        }
    payload = {
        "provider": PROVIDER,
        "timeframe": str(timeframe),
        "updated": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "rows": rows,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def iter_ib_symbols(execute: ExecuteFn, timeframe: str) -> List[str]:
    """Symbols with IB bars in the newest ~3 weeks (chunk exclusion; not all history)."""
    max_rows = list(execute(max_ts_sql(), (PROVIDER, timeframe)) or [])
    if not max_rows or max_rows[0][0] is None:
        return []
    as_of = pd.Timestamp(max_rows[0][0])
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("utc")
    else:
        as_of = as_of.tz_convert("utc")
    start = (as_of - pd.Timedelta(days=RECENT_SYMBOL_DAYS)).to_pydatetime()
    rows = list(execute(distinct_recent_symbols_sql(), (PROVIDER, timeframe, start)) or [])
    symbols = []
    for row in rows:
        if row and row[0]:
            symbols.append(str(row[0]).upper())
    logger.info(
        "%s recent-symbol DISTINCT %d names since %s (as_of %s)",
        timeframe,
        len(symbols),
        start,
        as_of,
    )
    return symbols


def fetch_edge_ts(
    execute: ExecuteFn,
    symbol: str,
    timeframe: str,
    *,
    newest: bool,
) -> Optional[pd.Timestamp]:
    rows = list(execute(edge_ts_sql(newest), (str(symbol).upper(), PROVIDER, timeframe)) or [])
    if not rows or rows[0][0] is None:
        return None
    ts = pd.Timestamp(rows[0][0])
    if ts.tzinfo is None:
        ts = ts.tz_localize("utc")
    else:
        ts = ts.tz_convert("utc")
    return ts


def coverage_frame(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["symbol", "first_ts", "last_ts", "n_bars"])
    if df.empty:
        return df
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def load_ib_coverage(
    timeframe: str,
    *,
    cache_dir: Optional[Path] = None,
    execute: Optional[ExecuteFn] = None,
) -> pd.DataFrame:
    """Per-symbol first/last IB timestamp for one timeframe (no GROUP BY)."""
    own_cursor = None
    if execute is None:
        client = get_timescaledb_client()
        if not client.ensure_connection():
            raise RuntimeError("Failed to connect to TimescaleDB")
        own_cursor = client.connection.cursor()

        def execute(sql: str, params: Sequence) -> Sequence:
            own_cursor.execute(sql, tuple(params))
            return own_cursor.fetchall()

    try:
        cache_path = cache_path_for(timeframe, cache_dir)
        first_cache = load_first_ts_cache(cache_path)
        symbols = iter_ib_symbols(execute, timeframe)
        if not symbols and first_cache:
            logger.warning(
                "%s recent DISTINCT empty; using %d cached symbols",
                timeframe,
                len(first_cache),
            )
            symbols = sorted(first_cache)
        rows: List[Dict] = []
        for i, sym in enumerate(symbols, start=1):
            cached_first = first_cache.get(sym)
            if cached_first:
                first_ts = pd.Timestamp(cached_first)
                if first_ts.tzinfo is None:
                    first_ts = first_ts.tz_localize("utc")
                else:
                    first_ts = first_ts.tz_convert("utc")
            else:
                first_ts = fetch_edge_ts(execute, sym, timeframe, newest=False)
            last_ts = fetch_edge_ts(execute, sym, timeframe, newest=True)
            rows.append(
                {
                    "symbol": sym,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "n_bars": None,
                }
            )
            if i % 200 == 0:
                logger.info("%s coverage %d/%d", timeframe, i, len(symbols))
        df = coverage_frame(rows)
        if not df.empty:
            save_coverage_cache(cache_path, timeframe, df)
            logger.info(
                "Wrote %s coverage cache (%d symbols) %s",
                timeframe,
                len(df),
                cache_path,
            )
        return df
    finally:
        if own_cursor is not None:
            own_cursor.close()


def load_ib_coverage_range(
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    symbols: Optional[Sequence[str]] = None,
    execute: Optional[ExecuteFn] = None,
) -> pd.DataFrame:
    """Per-symbol first/last IB timestamp inside [start, end). No COUNT(*)."""
    own_cursor = None
    if execute is None:
        client = get_timescaledb_client()
        if not client.ensure_connection():
            raise RuntimeError("Failed to connect to TimescaleDB")
        own_cursor = client.connection.cursor()

        def execute(sql: str, params: Sequence) -> Sequence:
            own_cursor.execute(sql, tuple(params))
            return own_cursor.fetchall()

    try:
        names = [str(s).upper() for s in symbols] if symbols is not None else iter_ib_symbols(
            execute, timeframe
        )
        sql = range_edge_sql()
        rows: List[Dict] = []
        for sym in names:
            fetched = list(execute(sql, (sym, PROVIDER, timeframe, start_dt, end_dt)) or [])
            if not fetched:
                continue
            first_ts, last_ts = fetched[0][0], fetched[0][1]
            if first_ts is None and last_ts is None:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "n_bars": None,
                }
            )
        return coverage_frame(rows)
    finally:
        if own_cursor is not None:
            own_cursor.close()
