from datetime import datetime, timezone

from utils.db.timescaledb_client import get_timescaledb_client
from utils.charting.ohlcv_window import load_ohlcv_window

client = get_timescaledb_client()
around = datetime(2026, 9, 10, 19, 45, tzinfo=timezone.utc)
raw = client.get_market_data_window("SPY", "1d", around=around, before=100, after=20)
print("raw window", None if raw is None else len(raw), raw["timestamp"].min() if raw is not None and len(raw) else None, raw["timestamp"].max() if raw is not None and len(raw) else None)
if raw is not None and not raw.empty:
    print("providers", raw["provider"].value_counts().to_dict() if "provider" in raw.columns else None)
    print("head ts", raw["timestamp"].head(3).tolist())
    print("tail ts", raw["timestamp"].tail(3).tolist())

win = load_ohlcv_window("SPY", "1d", around="2026-09-10 19:45:00", before=100, after=20)
print("load_ohlcv", len(win["df"]), win["df"].index.min() if len(win["df"]) else None, win["df"].index.max() if len(win["df"]) else None)
