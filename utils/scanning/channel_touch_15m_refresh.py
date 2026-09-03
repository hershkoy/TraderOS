"""Intraday IB 15m gap refresh for the channel-touch hot/armed list.

Pulls completed 15m bars from Gateway for a small symbol list (not the
~1478 universe), upserts TimescaleDB, then the scanner re-runs the H5
fill check on that last closed bar.

Client id 8826 (8821 = single-symbol backfill, 8822 = overnight universe,
8823 was the previous live id; Gateway can wedge a timed-out client).
"""
from __future__ import annotations

import logging
import socket
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.data.fetch_data import (
    _fetch_ib_intraday_batched,
    cleanup_ib_connection,
    create_ib_contract_with_primary_exchange,
    get_ib_connection,
    prepare_nautilus_dataframe,
    save_to_timescaledb,
    set_ib_client_id,
)
from utils.db.timescaledb_client import get_timescaledb_client
from utils.scanning.channel_touch_15m import (
    LIVE_15M_DEFAULTS,
    drop_incomplete_15m_bars,
)

logger = logging.getLogger(__name__)

LIVE_IB_CLIENT_ID = int(LIVE_15M_DEFAULTS["ib_client_id"])
GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4001


def gateway_tcp_open(host: str = GATEWAY_HOST, port: int = GATEWAY_PORT, timeout: float = 1.0) -> bool:
    """True if something is listening on the Gateway API port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(float(timeout))
    try:
        sock.connect((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def probe_ib_connected(*, client_id: int = LIVE_IB_CLIENT_ID) -> bool:
    """Real IB connect only. Do not raw-TCP 4001 first (Gateway handshake hang).

    Caller must cleanup_ib_connection().
    """
    set_ib_client_id(int(client_id))
    try:
        # Pass 4001 so get_ib_connection skips detect_ib_port (extra client 98 handshake).
        # start_loop=False matches tests/utils/ib_conn.py (startLoop + connect times out).
        ib = get_ib_connection(
            port=GATEWAY_PORT, client_id=int(client_id), start_loop=False
        )
    except Exception:
        logger.exception("IB connect failed (client %s)", client_id)
        cleanup_ib_connection()
        return False
    if ib is None or not ib.isConnected():
        logger.warning("IB isConnected() is false (client %s)", client_id)
        cleanup_ib_connection()
        return False
    return True


def last_ts_map_from_rows(
    rows: Sequence[dict], symbols: Sequence[str]
) -> Dict[str, pd.Timestamp]:
    """Watchlist as_of per symbol (naive timestamps treated as UTC)."""
    want = {str(s).upper() for s in symbols if str(s).strip()}
    out: Dict[str, pd.Timestamp] = {}
    for row in rows:
        stock = str(row.get("stock") or "").upper()
        if stock not in want or stock in out:
            continue
        if str(row.get("timeframe") or "15m") != "15m":
            continue
        raw = row.get("as_of")
        if raw in (None, ""):
            continue
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        out[stock] = ts
    return out


def last_ts_by_symbol(symbols: Sequence[str]) -> Dict[str, pd.Timestamp]:
    """MAX(ts) for a small IB 15m list (not a full-universe GROUP BY)."""
    clean = [str(s).upper() for s in symbols if str(s).strip()]
    if not clean:
        return {}
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")
    placeholders = ",".join(["%s"] * len(clean))
    sql = (
        "SELECT symbol, MAX(ts) AS last_ts FROM market_data "
        "WHERE provider = %s AND timeframe = %s AND symbol IN (%s) "
        "GROUP BY symbol"
    ) % ("%s", "%s", placeholders)
    cur = client.connection.cursor()
    try:
        cur.execute("SET LOCAL statement_timeout = '30s'")
        cur.execute(sql, tuple(["IB", "15m"] + clean))
        rows = cur.fetchall()
    finally:
        cur.close()
    out: Dict[str, pd.Timestamp] = {}
    for symbol, last_ts in rows:
        if last_ts is None:
            continue
        ts = pd.Timestamp(last_ts)
        out[str(symbol).upper()] = ts
    return out


def rewind_start(last_ts: pd.Timestamp, overlap_bars: int) -> datetime:
    ts = pd.Timestamp(last_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    delta = timedelta(minutes=15 * max(0, int(overlap_bars)))
    return (ts - delta).to_pydatetime()


def _fetch_window_start(
    last_ts: Optional[pd.Timestamp],
    *,
    now: datetime,
    overlap_bars: int,
    max_days: float,
) -> datetime:
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    floor = (now_ts - timedelta(days=float(max_days))).to_pydatetime()
    if last_ts is None or pd.isna(last_ts):
        return floor
    start = rewind_start(last_ts, overlap_bars)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if start < floor.replace(tzinfo=start.tzinfo):
        return floor
    return start


def fetch_and_store_symbol_gap(
    symbol: str,
    start_dt: datetime,
    *,
    now: Optional[datetime] = None,
    dry_run: bool = False,
) -> Tuple[int, Optional[str]]:
    """Fetch IB 15m from start_dt, drop the in-progress bar, upsert TimescaleDB.

    Returns (n_completed_bars_saved, last_completed_ts_iso).
    """
    ib = get_ib_connection(port=GATEWAY_PORT, start_loop=False)
    if ib is None or not ib.isConnected():
        raise RuntimeError("IB not connected")
    contract = create_ib_contract_with_primary_exchange(symbol)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.error("%s: qualifyContracts returned empty", symbol)
        return 0, None
    contract = qualified[0]
    end_dt = now or datetime.now(timezone.utc)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    raw = _fetch_ib_intraday_batched(symbol, "15m", ib, contract, start_dt, end_dt)
    if raw is None or raw.empty:
        logger.warning("%s: no new 15m bars from IB", symbol)
        return 0, None
    completed = drop_incomplete_15m_bars(raw, now=end_dt)
    if completed is None or completed.empty:
        logger.info("%s: IB returned only the in-progress 15m bar", symbol)
        return 0, None
    df = prepare_nautilus_dataframe(completed.copy(), symbol, "IB", "15m")
    last_ts = None
    if "ts_event" in df.columns and not df.empty:
        last_ts = str(pd.Timestamp(int(df["ts_event"].max()), unit="ns", tz="UTC"))
    if dry_run:
        logger.info("%s: dry-run, skip insert n=%d last=%s", symbol, len(df), last_ts)
        return int(len(df)), last_ts
    if not save_to_timescaledb(df, symbol, "IB", "15m"):
        logger.error("%s: TimescaleDB insert failed", symbol)
        return 0, last_ts
    return int(len(df)), last_ts


def refresh_ib_15m_symbols(
    symbols: Sequence[str],
    *,
    client_id: int = LIVE_IB_CLIENT_ID,
    sleep_s: float = 0.35,
    overlap_bars: int = 2,
    max_days: float = 5.0,
    close_lag_sec: float = 8.0,
    last_ts_map: Optional[Dict[str, pd.Timestamp]] = None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
    disconnect: bool = True,
) -> List[str]:
    """Fetch+store completed 15m bars. Returns symbols that saved at least one bar.

    Probes Gateway first. Cleans up the IB connection when finished unless
    ``disconnect`` is False (keep the session for a following batch).
    """
    wanted = [str(s).upper() for s in symbols if str(s).strip()]
    if not wanted:
        return []
    if close_lag_sec and float(close_lag_sec) > 0:
        time.sleep(float(close_lag_sec))
    now_ts = now or datetime.now(timezone.utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=timezone.utc)
    ts_map: Dict[str, pd.Timestamp] = dict(last_ts_map) if last_ts_map is not None else {}
    missing = [s for s in wanted if s not in ts_map]
    if missing:
        try:
            ts_map.update(last_ts_by_symbol(missing))
        except Exception:
            logger.exception(
                "last_ts_by_symbol failed for %d symbols; using max_days window",
                len(missing),
            )
    saved: List[str] = []
    if not probe_ib_connected(client_id=int(client_id)):
        raise RuntimeError(
            "IB API handshake failed (client %s)" % client_id
        )
    try:
        for i, sym in enumerate(wanted, start=1):
            start_dt = _fetch_window_start(
                ts_map.get(sym),
                now=now_ts,
                overlap_bars=int(overlap_bars),
                max_days=float(max_days),
            )
            logger.info(
                "[%d/%d] IB 15m refresh %s from %s",
                i,
                len(wanted),
                sym,
                start_dt.isoformat(),
            )
            try:
                n_bars, last = fetch_and_store_symbol_gap(
                    sym, start_dt, now=now_ts, dry_run=bool(dry_run)
                )
            except Exception as exc:
                logger.exception("%s: live 15m refresh failed: %s", sym, exc)
                continue
            if n_bars > 0:
                saved.append(sym)
                logger.info("%s: stored %d completed 15m bars last=%s", sym, n_bars, last)
            if i < len(wanted) and float(sleep_s) > 0:
                time.sleep(float(sleep_s))
    finally:
        if disconnect:
            cleanup_ib_connection()
    logger.info("IB 15m live refresh saved=%d / %d", len(saved), len(wanted))
    return saved
