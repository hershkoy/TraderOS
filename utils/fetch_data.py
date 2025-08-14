# fetch_data.py
from dotenv import load_dotenv
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from utils.timescaledb_client import get_timescaledb_client

# ─────────────────────────────
# CONFIG
# ─────────────────────────────

load_dotenv()

api_key    = os.getenv("ALPACA_API_KEY_ID")
secret_key = os.getenv("ALPACA_API_SECRET")

ALPACA_BAR_CAP = 10_000
IB_BAR_CAP     = 3_000

SAVE_DIR = Path("./data")
SAVE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────
def prepare_nautilus_dataframe(df, symbol, provider, timeframe):
    """
    Transform raw DataFrame to NautilusTrader-compatible format.
    
    Required columns: ts_event, open, high, low, close, volume, instrument_id, venue_id, timeframe
    """
    # Ensure timestamp column exists and is properly formatted
    if 'timestamp' in df.columns:
        # Convert to UTC if not already
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        elif df['timestamp'].dt.tz != timezone.utc:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    else:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    # Convert timestamp to nanoseconds for ts_event
    df['ts_event'] = df['timestamp'].astype('int64')
    
    # Add required metadata columns
    df['instrument_id'] = symbol
    df['venue_id'] = provider.upper()
    df['timeframe'] = timeframe
    
    # Select and order columns according to Nautilus requirements
    required_columns = ['ts_event', 'open', 'high', 'low', 'close', 'volume', 'instrument_id', 'venue_id', 'timeframe']
    
    # Ensure all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df[required_columns]

def save_to_timescaledb(df, symbol, provider, timeframe):
    """Save DataFrame to TimescaleDB."""
    try:
        client = get_timescaledb_client()
        if client.insert_market_data(df, symbol, provider, timeframe):
            print(f"✔ Saved {len(df)} records to TimescaleDB for {symbol} {timeframe}")
            return True
        else:
            print(f"✗ Failed to save to TimescaleDB for {symbol} {timeframe}")
            return False
    except Exception as e:
        print(f"✗ Error saving to TimescaleDB: {e}")
        return False

# ─────────────────────────────
# ALPACA LOGIC
# ─────────────────────────────
def fetch_from_alpaca(symbol, bars, timeframe):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    print(f"→ Fetching {bars} bars from Alpaca for {symbol} @ {timeframe}...")

    tf_map = {
        "1h": TimeFrame.Hour,
        "1d": TimeFrame.Day,
    }

    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe for Alpaca: {timeframe}")

    client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
    )

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=bars if timeframe == "1h" else 24 * bars)

    req = StockBarsRequest(
        feed=DataFeed.IEX,
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe],
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        limit=min(bars, ALPACA_BAR_CAP),
    )

    # Get raw data
    raw_data = client.get_stock_bars(req)
    df = raw_data.df.reset_index()
    
    # Transform to Nautilus format
    df = prepare_nautilus_dataframe(df, symbol, "ALPACA", timeframe)
    
    # Save to TimescaleDB
    save_to_timescaledb(df, symbol, "ALPACA", timeframe)

# ─────────────────────────────
# IBKR LOGIC
# ─────────────────────────────
def fetch_from_ib(symbol, bars, timeframe):
    from ib_insync import IB, Stock

    print(f"→ Fetching {bars} bars from IBKR for {symbol} @ {timeframe}...")

    if timeframe not in ["1h", "1d"]:
        raise ValueError(f"Unsupported timeframe for IBKR: {timeframe}")

    ib = IB()
    ib.connect("127.0.0.1", 4001, clientId=42)

    contract = Stock(symbol, "SMART", "USD")
    bar_size = "1 hour" if timeframe == "1h" else "1 day"
    dur_unit = f"{min(bars, IB_BAR_CAP)} {'H' if timeframe == '1h' else 'D'}"

    bars_data = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=dur_unit,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    ib.disconnect()

    # Convert to DataFrame
    df = pd.DataFrame([b.__dict__ for b in bars_data])[['date', 'open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={"date": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d  %H:%M:%S").dt.tz_localize("US/Eastern").dt.tz_convert("UTC")

    # Transform to Nautilus format
    df = prepare_nautilus_dataframe(df, symbol, "IB", timeframe)
    
    # Save to TimescaleDB
    save_to_timescaledb(df, symbol, "IB", timeframe)

# ─────────────────────────────
# MAIN ENTRY
# ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical bars from Alpaca or IBKR.")
    parser.add_argument("--symbol", required=True, help="Symbol to fetch (e.g. NFLX)")
    parser.add_argument("--provider", required=True, choices=["alpaca", "ib"], help="Data provider")
    parser.add_argument("--timeframe", required=True, choices=["1h", "1d"], help="Timeframe to fetch")
    parser.add_argument("--bars", type=int, default=1000, help="Number of bars to fetch")

    args = parser.parse_args()

    symbol    = args.symbol.upper()
    provider  = args.provider.lower()
    timeframe = args.timeframe
    bars      = args.bars

    if provider == "alpaca":
        bars = min(bars, ALPACA_BAR_CAP)
        fetch_from_alpaca(symbol, bars, timeframe)
    elif provider == "ib":
        bars = min(bars, IB_BAR_CAP)
        fetch_from_ib(symbol, bars, timeframe)
