"""Incremental GLD 1h backfill: always start from last stored bar."""
from datetime import datetime, timezone

from utils.data.fetch_data import fetch_from_alpaca, save_to_timescaledb
from utils.db.timescaledb_client import get_timescaledb_client


def last_ts(symbol="GLD", timeframe="1h"):
    c = get_timescaledb_client()
    assert c.ensure_connection()
    cur = c.connection.cursor()
    cur.execute(
        "SELECT MAX(ts) FROM market_data WHERE symbol=%s AND timeframe=%s",
        (symbol, timeframe),
    )
    row = cur.fetchone()
    return row[0] if row else None


def main():
    symbol = "GLD"
    timeframe = "1h"
    rounds = 40
    for i in range(rounds):
        mx = last_ts(symbol, timeframe)
        if mx is None:
            start = "2020-01-01"
        else:
            # resume one hour after last
            if mx.tzinfo is None:
                mx = mx.replace(tzinfo=timezone.utc)
            start = (mx.astimezone(timezone.utc)).strftime("%Y-%m-%d")
        print(f"round {i+1}/{rounds} start={start} last={mx}")
        try:
            df = fetch_from_alpaca(symbol, bars=2000, timeframe=timeframe, start_date=start)
        except Exception as e:
            print(f"  FAIL {e}")
            break
        if df is None or getattr(df, "empty", True):
            print("  empty - done")
            break
        n = len(df)
        ok = save_to_timescaledb(df, symbol, "ALPACA", timeframe)
        print(f"  fetched={n} save={ok}")
        new_mx = last_ts(symbol, timeframe)
        if new_mx == mx:
            print("  no progress - stop")
            break
        # if we reached near now, stop
        if new_mx and (datetime.now(timezone.utc) - new_mx.astimezone(timezone.utc)).days < 3:
            print(f"  caught up to {new_mx}")
            break
    print("final last_ts=", last_ts(symbol, timeframe))


if __name__ == "__main__":
    main()
