# fetch_data.py
from dotenv import load_dotenv
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import time
import logging
try:
    from utils.timescaledb_client import get_timescaledb_client
except ImportError:
    # Fallback for when running from utils directory
    from timescaledb_client import get_timescaledb_client

# ─────────────────────────────
# CONFIG
# ─────────────────────────────

load_dotenv()

api_key    = os.getenv("ALPACA_API_KEY_ID")
secret_key = os.getenv("ALPACA_API_SECRET")

ALPACA_BAR_CAP = 10_000
IB_BAR_CAP     = 3_000

# Rate limiting and throttling settings
ALPACA_REQUEST_DELAY = 0.1  # 100ms between requests
IB_REQUEST_DELAY = 0.5      # 500ms between requests
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

SAVE_DIR = Path("./data")
SAVE_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # Connect to the database before inserting data
        if not client.connect():
            logger.error(f"✗ Failed to connect to TimescaleDB")
            return False
        

            
        if client.insert_market_data(df, symbol, provider, timeframe):
            logger.info(f"✔ Saved {len(df)} records to TimescaleDB for {symbol} {timeframe}")
            client.disconnect()
            return True
        else:
            logger.error(f"✗ Failed to save to TimescaleDB for {symbol} {timeframe}")
            client.disconnect()
            return False
    except Exception as e:
        logger.error(f"✗ Error saving to TimescaleDB: {e}")
        return False

def handle_rate_limiting(response, provider):
    """Handle rate limiting and throttling responses."""
    if hasattr(response, 'status_code'):
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', RETRY_DELAY))
            logger.warning(f"Rate limited by {provider}. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return True
        elif response.status_code >= 500:
            logger.warning(f"Server error from {provider}. Retrying...")
            time.sleep(RETRY_DELAY)
            return True
    return False

# ─────────────────────────────
# ALPACA LOGIC
# ─────────────────────────────
def fetch_from_alpaca(symbol, bars, timeframe):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    if bars == "max":
        logger.info(f"→ Fetching maximum available bars from Alpaca for {symbol} @ {timeframe}...")
        return fetch_max_from_alpaca(symbol, timeframe)
    else:
        logger.info(f"→ Fetching {bars} bars from Alpaca for {symbol} @ {timeframe}...")
        return fetch_single_alpaca_request(symbol, bars, timeframe)

def fetch_single_alpaca_request(symbol, bars, timeframe):
    """Fetch a single request from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

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

    # Get raw data with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            raw_data = client.get_stock_bars(req)
            df = raw_data.df.reset_index()
            
            # Transform to Nautilus format
            df = prepare_nautilus_dataframe(df, symbol, "ALPACA", timeframe)
            
            # Save to TimescaleDB
            save_to_timescaledb(df, symbol, "ALPACA", timeframe)
            return df
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                logger.warning(f"Rate limited by Alpaca. Waiting {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            elif attempt < MAX_RETRIES - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to fetch data from Alpaca after {MAX_RETRIES} attempts: {e}")
                raise

def fetch_max_from_alpaca(symbol, timeframe):
    """Fetch maximum available historical data from Alpaca by looping through requests."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

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

    all_data = []
    end_time = datetime.now(timezone.utc)
    total_bars_fetched = 0
    
    logger.info(f"Starting maximum data fetch for {symbol} @ {timeframe}")
    
    while True:
        # Calculate start time for this batch
        if timeframe == "1h":
            start_time = end_time - timedelta(hours=ALPACA_BAR_CAP)
        else:  # 1d
            start_time = end_time - timedelta(days=ALPACA_BAR_CAP)
        
        req = StockBarsRequest(
            feed=DataFeed.IEX,
            symbol_or_symbols=symbol,
            timeframe=tf_map[timeframe],
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            limit=ALPACA_BAR_CAP,
        )

        # Get data with retry logic
        batch_data = None
        for attempt in range(MAX_RETRIES):
            try:
                raw_data = client.get_stock_bars(req)
                batch_df = raw_data.df.reset_index()
                
                if batch_df.empty:
                    logger.info(f"No more data available. Total bars fetched: {total_bars_fetched}")
                    break
                
                batch_data = batch_df
                break
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Rate limited by Alpaca. Waiting {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                elif attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to fetch batch from Alpaca after {MAX_RETRIES} attempts: {e}")
                    raise
        
        if batch_data is None or batch_data.empty:
            break
        
        # Add to collection
        all_data.append(batch_data)
        total_bars_fetched += len(batch_data)
        
        logger.info(f"Fetched batch of {len(batch_data)} bars. Total: {total_bars_fetched}")
        
        # Update end_time for next batch (go further back in time)
        end_time = start_time
        
        # Add delay between requests to respect rate limits
        time.sleep(ALPACA_REQUEST_DELAY)
        
        # Safety check to prevent infinite loops
        if total_bars_fetched > 1000000:  # 1M bars limit
            logger.warning("Reached safety limit of 1M bars. Stopping fetch.")
            break
    
    if not all_data:
        logger.warning("No data was fetched from Alpaca")
        return None
    
    # Combine all batches
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    logger.info(f"Combined {len(combined_df)} unique bars from {len(all_data)} batches")
    
    # Transform to Nautilus format
    df = prepare_nautilus_dataframe(combined_df, symbol, "ALPACA", timeframe)
    
    # Save to TimescaleDB
    save_to_timescaledb(df, symbol, "ALPACA", timeframe)
    
    return df

# ─────────────────────────────
# IBKR LOGIC
# ─────────────────────────────
def fetch_from_ib(symbol, bars, timeframe):
    from ib_insync import IB, Stock

    if bars == "max":
        logger.info(f"→ Fetching maximum available bars from IBKR for {symbol} @ {timeframe}...")
        return fetch_max_from_ib(symbol, timeframe)
    else:
        logger.info(f"→ Fetching {bars} bars from IBKR for {symbol} @ {timeframe}...")
        return fetch_single_ib_request(symbol, bars, timeframe)

def fetch_single_ib_request(symbol, bars, timeframe):
    """Fetch a single request from IBKR."""
    from ib_insync import IB, Stock

    if timeframe not in ["1h", "1d"]:
        raise ValueError(f"Unsupported timeframe for IBKR: {timeframe}")

    ib = IB()
    ib.connect("127.0.0.1", 4001, clientId=42)

    contract = Stock(symbol, "SMART", "USD")
    bar_size = "1 hour" if timeframe == "1h" else "1 day"
    dur_unit = f"{min(bars, IB_BAR_CAP)} {'H' if timeframe == '1h' else 'D'}"

    for attempt in range(MAX_RETRIES):
        try:
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
            
            return df
            
        except Exception as e:
            ib.disconnect()
            if "rate limit" in str(e).lower() or "throttle" in str(e).lower():
                logger.warning(f"Rate limited by IBKR. Waiting {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            elif attempt < MAX_RETRIES - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to fetch data from IBKR after {MAX_RETRIES} attempts: {e}")
                raise

def fetch_max_from_ib(symbol, timeframe):
    """Fetch maximum available historical data from IBKR by looping through requests."""
    from ib_insync import IB, Stock

    if timeframe not in ["1h", "1d"]:
        raise ValueError(f"Unsupported timeframe for IBKR: {timeframe}")

    all_data = []
    total_bars_fetched = 0
    end_date = ""  # Start from most recent data
    
    logger.info(f"Starting maximum data fetch for {symbol} @ {timeframe}")
    
    while True:
        ib = IB()
        ib.connect("127.0.0.1", 4001, clientId=42)

        contract = Stock(symbol, "SMART", "USD")
        bar_size = "1 hour" if timeframe == "1h" else "1 day"
        dur_unit = f"{IB_BAR_CAP} {'H' if timeframe == '1h' else 'D'}"

        # Get data with retry logic
        batch_data = None
        for attempt in range(MAX_RETRIES):
            try:
                bars_data = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_date,
                    durationStr=dur_unit,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
                
                if not bars_data:
                    logger.info(f"No more data available. Total bars fetched: {total_bars_fetched}")
                    break
                
                # Convert to DataFrame
                batch_df = pd.DataFrame([b.__dict__ for b in bars_data])[['date', 'open', 'high', 'low', 'close', 'volume']]
                batch_df.rename(columns={"date": "timestamp"}, inplace=True)
                batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], format="%Y%m%d  %H:%M:%S").dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
                
                batch_data = batch_df
                break
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "throttle" in str(e).lower():
                    logger.warning(f"Rate limited by IBKR. Waiting {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                elif attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to fetch batch from IBKR after {MAX_RETRIES} attempts: {e}")
                    ib.disconnect()
                    raise
        
        ib.disconnect()
        
        if batch_data is None or batch_data.empty:
            break
        
        # Add to collection
        all_data.append(batch_data)
        total_bars_fetched += len(batch_data)
        
        logger.info(f"Fetched batch of {len(batch_data)} bars. Total: {total_bars_fetched}")
        
        # Update end_date for next batch (go further back in time)
        if len(batch_data) > 0:
            end_date = batch_data['timestamp'].min().strftime("%Y%m%d %H:%M:%S")
        
        # Add delay between requests to respect rate limits
        time.sleep(IB_REQUEST_DELAY)
        
        # Safety check to prevent infinite loops
        if total_bars_fetched > 1000000:  # 1M bars limit
            logger.warning("Reached safety limit of 1M bars. Stopping fetch.")
            break
    
    if not all_data:
        logger.warning("No data was fetched from IBKR")
        return None
    
    # Combine all batches
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    logger.info(f"Combined {len(combined_df)} unique bars from {len(all_data)} batches")
    
    # Transform to Nautilus format
    df = prepare_nautilus_dataframe(combined_df, symbol, "IB", timeframe)
    
    # Save to TimescaleDB
    save_to_timescaledb(df, symbol, "IB", timeframe)
    
    return df

# ─────────────────────────────
# MAIN ENTRY
# ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical bars from Alpaca or IBKR.")
    parser.add_argument("--symbol", required=True, help="Symbol to fetch (e.g. NFLX)")
    parser.add_argument("--provider", required=True, choices=["alpaca", "ib"], help="Data provider")
    parser.add_argument("--timeframe", required=True, choices=["1h", "1d"], help="Timeframe to fetch")
    parser.add_argument("--bars", default=1000, help="Number of bars to fetch or 'max' for maximum available")

    args = parser.parse_args()

    symbol    = args.symbol.upper()
    provider  = args.provider.lower()
    timeframe = args.timeframe
    bars      = args.bars

    # Handle bars argument
    if bars == "max":
        bars = "max"
    else:
        try:
            bars = int(bars)
            if bars <= 0:
                raise ValueError("Bars must be a positive integer or 'max'")
        except ValueError:
            logger.error("Bars must be a positive integer or 'max'")
            exit(1)

    try:
        if provider == "alpaca":
            if bars == "max":
                fetch_from_alpaca(symbol, bars, timeframe)
            else:
                bars = min(bars, ALPACA_BAR_CAP)
                fetch_from_alpaca(symbol, bars, timeframe)
        elif provider == "ib":
            if bars == "max":
                fetch_from_ib(symbol, bars, timeframe)
            else:
                bars = min(bars, IB_BAR_CAP)
                fetch_from_ib(symbol, bars, timeframe)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        exit(1)
