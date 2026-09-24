"""Ingest TradingView dump_tv_bars.mjs JSON for COMEX GC1! 1h into TimescaleDB."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.db.timescaledb_client import get_timescaledb_client

DUMP = Path(r"d:\WORK\Projs\IB\backTraderTest\reports\tv_dumps\gc1_1h.json")
SYMBOL = "GC1!"
PROVIDER = "TRADINGVIEW"
TIMEFRAME = "1h"


def bars_to_df(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bars = payload.get("bars") or []
    if not bars:
        raise SystemExit(f"no bars in {path}")
    rows = []
    for row in bars:
        if not row or len(row) < 5:
            continue
        ts = pd.Timestamp(int(row[0]), unit="s", tz="UTC")
        vol = float(row[5]) if len(row) > 5 and row[5] is not None else 0.0
        rows.append(
            {
                "ts_event": int(ts.value),  # ns
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": int(max(vol, 0)),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts_event"]).sort_values("ts_event")
    return df.reset_index(drop=True)


def main():
    df = bars_to_df(DUMP)
    t0 = pd.Timestamp(int(df["ts_event"].iloc[0]), unit="ns", tz="UTC")
    t1 = pd.Timestamp(int(df["ts_event"].iloc[-1]), unit="ns", tz="UTC")
    print(f"parsed n={len(df)} {t0} -> {t1}")
    client = get_timescaledb_client()
    assert client.ensure_connection()
    ok = client.insert_market_data(df, SYMBOL, PROVIDER, TIMEFRAME)
    print(f"insert_market_data -> {ok}")
    cur = client.connection.cursor()
    cur.execute(
        """
        SELECT COUNT(*)::bigint, MIN(ts), MAX(ts)
        FROM market_data
        WHERE symbol = %s AND timeframe = %s AND provider = %s
        """,
        (SYMBOL, TIMEFRAME, PROVIDER),
    )
    print("db:", cur.fetchone())


if __name__ == "__main__":
    main()
