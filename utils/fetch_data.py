# fetch_data.py
from dotenv import load_dotenv
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import time
import logging
import random
import threading
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

# Constants
ALPACA_BAR_CAP = 10000  # Alpaca's limit per request
IB_BAR_CAP = 3000       # IBKR's limit per request
MAX_RETRIES = 3
RETRY_DELAY = 5
IB_REQUEST_DELAY = 1     # Delay between IBKR requests

# Global IBKR connection manager
_ib_connection = None
_ib_lock = threading.Lock()

def get_ib_connection():
    """Get or create a shared IBKR connection"""
    global _ib_connection
    
    with _ib_lock:
        if _ib_connection is None or not _ib_connection.isConnected():
            from ib_insync import IB
            _ib_connection = IB()
            client_id = random.randint(1000, 9999)
            logger.info(f"Creating new IBKR connection with client ID {client_id}")
            
            # Try to connect with retry logic
            max_connection_attempts = 3
            for attempt in range(max_connection_attempts):
                try:
                    _ib_connection.connect("127.0.0.1", 4001, clientId=client_id)
                    logger.info(f"IBKR connection established with client ID {client_id}")
                    break
                except Exception as e:
                    if attempt < max_connection_attempts - 1:
                        logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(2)
                    else:
                        logger.error(f"Failed to connect to IBKR after {max_connection_attempts} attempts: {e}")
                        raise
        
        return _ib_connection

def close_ib_connection():
    """Close the global IBKR connection"""
    global _ib_connection
    
    with _ib_lock:
        if _ib_connection and _ib_connection.isConnected():
            try:
                _ib_connection.disconnect()
                logger.info("Global IBKR connection closed")
            except Exception as e:
                logger.warning(f"Error closing IBKR connection: {e}")
            finally:
                _ib_connection = None

def cleanup_ib_connection():
    """Cleanup function to be called when done with IBKR operations"""
    close_ib_connection()

SAVE_DIR = Path("./data")
SAVE_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter out unwanted IBKR messages
ib_logger = logging.getLogger('ib_insync.wrapper')
ib_logger.setLevel(logging.WARNING)  # Only show warnings and errors, not info

# Filter out position messages specifically
class PositionFilter(logging.Filter):
    def filter(self, record):
        return not (record.getMessage().startswith('position:') or 
                   'Position(...)' in record.getMessage() or
                   'Market data farm connection is OK' in record.getMessage() or
                   'HMDS data farm connection is OK' in record.getMessage() or
                   'Sec-def data farm connection is OK' in record.getMessage())

ib_logger.addFilter(PositionFilter())

# ─────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────

def format_timestamp_for_alpaca(dt):
    """
    Format timestamp for Alpaca API.
    Alpaca expects RFC3339 format without microseconds.
    """
    # Remove microseconds and format as RFC3339
    dt_clean = dt.replace(microsecond=0)
    return dt_clean.isoformat()

def format_date_for_alpaca(dt):
    """
    Format date for Alpaca API using simple YYYY-MM-DD format.
    This avoids timestamp parsing issues.
    """
    return dt.strftime("%Y-%m-%d")

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

def detect_max_available_bars(symbol, timeframe):
    """
    Intelligently detect how many bars are actually available from Alpaca
    and return the count along with the actual time range.
    """
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
    
    try:
        # Test different data feeds to see which gives more data
        feeds_to_test = [DataFeed.IEX, DataFeed.SIP]  # Try both IEX and SIP feeds
        
        best_result = None
        best_feed = None
        
        for feed in feeds_to_test:
            logger.info(f"🔍 Testing {feed.value} feed for {symbol} @ {timeframe}...")
            
            end_time = datetime.now(timezone.utc)
            # Go back much further to see what's actually available
            if timeframe == "1h":
                start_time = end_time - timedelta(days=2000)  # Try 5+ years
            else:  # 1d
                start_time = end_time - timedelta(days=3000)  # Try 8+ years
            
            req = StockBarsRequest(
                feed=feed,
                symbol_or_symbols=symbol,
                timeframe=tf_map[timeframe],
                start=format_date_for_alpaca(start_time),
                end=format_date_for_alpaca(end_time),
                limit=ALPACA_BAR_CAP,  # Use the maximum allowed
            )
            
            try:
                raw_data = client.get_stock_bars(req)
                df = raw_data.df.reset_index()
                
                if not df.empty:
                    actual_bars = len(df)
                    logger.info(f"📊 {feed.value} feed: got {actual_bars} bars from {start_time.date()} to {end_time.date()}")
                    logger.info(f"📅 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    
                    # Track the best result
                    if best_result is None or actual_bars > best_result[0]:
                        best_result = (actual_bars, start_time, end_time)
                        best_feed = feed
                        
                        # If we got the full 10,000 bars, this feed is working well
                        if actual_bars >= ALPACA_BAR_CAP:
                            logger.info(f"🎯 {feed.value} feed hit Alpaca's limit of {ALPACA_BAR_CAP} bars!")
                            break
                else:
                    logger.warning(f"❌ {feed.value} feed: No data available")
                    
            except Exception as e:
                logger.warning(f"❌ {feed.value} feed failed: {e}")
                continue
        
        if best_result is None:
            logger.error("❌ All data feeds failed!")
            return None
            
        actual_bars, start_time, end_time = best_result
        logger.info(f"🏆 Best result: {best_feed.value} feed with {actual_bars} bars")
        
        # If we got the full 10,000 bars, there might be more data available
        if actual_bars >= ALPACA_BAR_CAP:
            logger.info(f"🎯 Hit Alpaca's limit of {ALPACA_BAR_CAP} bars. There might be more data available.")
            logger.info(f"💡 Consider using multiple requests or the old batch method for truly maximum data.")
            return (ALPACA_BAR_CAP, start_time, end_time)
        
        # If we got less than the limit, that's probably all the data available
        logger.info(f"📈 Got {actual_bars} bars, which appears to be the maximum available for {symbol} @ {timeframe}")
        
        # Calculate expected vs actual
        if timeframe == "1h":
            expected_per_year = 8.5 * 5 * 52  # 8.5 hours × 5 days × 52 weeks
            years_covered = (end_time - start_time).days / 365.25
            expected_total = int(expected_per_year * years_covered)
            logger.warning(f"⚠️  Expected ~{expected_total} bars for {years_covered:.1f} years, but only got {actual_bars}")
            logger.warning(f"⚠️  This suggests data filtering or API limitations")
        
        return (actual_bars, start_time, end_time)
            
    except Exception as e:
        logger.warning(f"Error detecting max bars for {symbol} @ {timeframe}: {e}")
        # Return conservative defaults
        if timeframe == "1h":
            return (2000, datetime.now(timezone.utc) - timedelta(days=1000), datetime.now(timezone.utc))
        else:
            return (1000, datetime.now(timezone.utc) - timedelta(days=1000), datetime.now(timezone.utc))

def detect_max_available_bars_with_range(symbol, timeframe):
    """
    Wrapper function that maintains backward compatibility.
    """
    return detect_max_available_bars(symbol, timeframe)

def fetch_smart_max_from_alpaca(symbol, timeframe):
    """
    Smart version of max bars that first detects what's available
    then fetches that amount using the reliable single request method.
    """
    logger.info(f"🔍 Detecting maximum available bars for {symbol} @ {timeframe}...")
    
    # Get both the count and the actual time range from detection
    detection_result = detect_max_available_bars_with_range(symbol, timeframe)
    
    if detection_result is None:
        logger.warning(f"No data available for {symbol} @ {timeframe}")
        return None
    
    max_available, start_time, end_time = detection_result
    
    logger.info(f"📊 Detected {max_available} bars available for {symbol} @ {timeframe}")
    logger.info(f"📅 Time range: {start_time.date()} to {end_time.date()}")
    logger.info(f"🚀 Fetching {max_available} bars using single request method...")
    
    # Use the reliable single request method with the detected amount and time range
    return fetch_single_alpaca_request_with_range(symbol, max_available, timeframe, start_time, end_time)

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
        return fetch_smart_max_from_alpaca(symbol, timeframe)
    else:
        logger.info(f"→ Fetching {bars} bars from Alpaca for {symbol} @ {timeframe}...")
        return fetch_single_alpaca_request(symbol, bars, timeframe)

def fetch_single_alpaca_request_with_range(symbol, bars, timeframe, start_time, end_time):
    """Fetch a single request from Alpaca using a specific time range."""
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

    logger.info(f"Requesting data from {start_time} to {end_time} (expecting ~{bars} bars)")

    req = StockBarsRequest(
        feed=DataFeed.IEX,
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe],
        start=format_date_for_alpaca(start_time),
        end=format_date_for_alpaca(end_time),
        limit=min(bars * 2, ALPACA_BAR_CAP),  # Request a bit more than needed
    )

    # Get raw data with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            raw_data = client.get_stock_bars(req)
            df = raw_data.df.reset_index()
            
            logger.info(f"Received {len(df)} bars from Alpaca (requested ~{bars})")
            if not df.empty:
                logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Take only the most recent 'bars' records
            if len(df) > bars:
                df = df.tail(bars).reset_index(drop=True)
                logger.info(f"Truncated to {len(df)} most recent bars")
            elif len(df) < bars:
                logger.warning(f"Only got {len(df)} bars, less than requested {bars}")
            
            # Transform to Nautilus format
            df = prepare_nautilus_dataframe(df, symbol, "ALPACA", timeframe)
            
            # Note: Database saving is now handled by the caller
            # save_to_timescaledb(df, symbol, "ALPACA", timeframe)
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

    # Calculate a reasonable time period to get the requested number of bars
    # For hourly data: account for ~6.5 hours per trading day, weekends, holidays
    # For daily data: account for weekends and holidays
    if timeframe == "1h":
        # Assume ~6.5 trading hours per day, ~5 trading days per week
        # So we need roughly (bars / 6.5) * 7/5 days to get enough bars
        days_needed = max(1, int((bars / 6.5) * 7 / 5))
        start_time = datetime.now(timezone.utc) - timedelta(days=days_needed)
    else:  # 1d
        # For daily data, assume ~5 trading days per week
        days_needed = max(1, int(bars * 7 / 5))
        start_time = datetime.now(timezone.utc) - timedelta(days=days_needed)
    
    end_time = datetime.now(timezone.utc)

    logger.info(f"Requesting data from {start_time} to {end_time} (expecting ~{bars} bars)")

    req = StockBarsRequest(
        feed=DataFeed.IEX,
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe],
        start=format_date_for_alpaca(start_time),
        end=format_date_for_alpaca(end_time),
        limit=min(bars * 5, ALPACA_BAR_CAP),  # Request much more than needed to ensure we get enough
    )

    # Get raw data with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            raw_data = client.get_stock_bars(req)
            df = raw_data.df.reset_index()
            
            logger.info(f"Received {len(df)} bars from Alpaca (requested ~{bars})")
            if not df.empty:
                logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Take only the most recent 'bars' records
            if len(df) > bars:
                df = df.tail(bars).reset_index(drop=True)
                logger.info(f"Truncated to {len(df)} most recent bars")
            elif len(df) < bars:
                logger.warning(f"Only got {len(df)} bars, less than requested {bars}")
            
            # Transform to Nautilus format
            df = prepare_nautilus_dataframe(df, symbol, "ALPACA", timeframe)
            
            # Note: Database saving is now handled by the caller
            # save_to_timescaledb(df, symbol, "ALPACA", timeframe)
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
            start=format_date_for_alpaca(start_time),
            end=format_date_for_alpaca(end_time),
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
    
    # Note: Database saving is now handled by the caller
    # save_to_timescaledb(df, symbol, "ALPACA", timeframe)
    
    return df

# ─────────────────────────────
# IBKR LOGIC
# ─────────────────────────────
def fetch_from_ib(symbol, bars, timeframe):
    """Fetch historical bars from IBKR."""
    from ib_insync import IB, Stock

    if timeframe not in ["1h", "1d"]:
        raise ValueError(f"Unsupported timeframe for IBKR: {timeframe}")

    if bars == "max":
        return fetch_max_from_ib(symbol, timeframe)
    
    # For fixed number of bars
    ib = get_ib_connection()
    
    try:
        contract = Stock(symbol, "SMART", "USD")
        bar_size = "1 hour" if timeframe == "1h" else "1 day"
        
        # Ensure proper spacing for IBKR duration format
        # IBKR only supports: S (seconds), D (days), W (weeks), M (months), Y (years)
        # For hourly data, convert to days: 1 hour = 1/24 day
        if timeframe == '1h':
            # Convert hours to days (1 hour = 1/24 day)
            hours_to_days = bars / 24
            dur_unit = f"{int(hours_to_days)} D"
        else:
            dur_unit = f"{bars} D"
        
        logger.info(f"IBKR duration string: '{dur_unit}' (length: {len(dur_unit)})")

        # Get data with retry logic
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
                
                if not bars_data:
                    logger.warning(f"No data returned from IBKR for {symbol}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame([b.__dict__ for b in bars_data])[['date', 'open', 'high', 'low', 'close', 'volume']]
                df.rename(columns={"date": "timestamp"}, inplace=True)
                
                # Handle timezone conversion properly
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d  %H:%M:%S")
                
                # Check if timestamp is already timezone-aware
                if df["timestamp"].dt.tz is None:
                    # Not timezone-aware, localize to US/Eastern then convert to UTC
                    df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
                else:
                    # Already timezone-aware, just convert to UTC
                    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

                # Transform to Nautilus format
                df = prepare_nautilus_dataframe(df, symbol, "IB", timeframe)
                
                # Note: Database saving is now handled by the caller
                # save_to_timescaledb(df, symbol, "IB", timeframe)
                
                return df
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "throttle" in str(e).lower():
                    logger.warning(f"Rate limited by IBKR. Waiting {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                elif attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to fetch data from IBKR after {MAX_RETRIES} attempts: {e}")
                    raise
    finally:
        # Don't disconnect - we want to reuse the connection
        pass

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
        ib = get_ib_connection()
        
        try:
            contract = Stock(symbol, "SMART", "USD")
            bar_size = "1 hour" if timeframe == "1h" else "1 day"
            # Ensure proper spacing for IBKR duration format
            # IBKR only supports: S (seconds), D (days), W (weeks), M (months), Y (years)
            # For hourly data, convert to days: 1 hour = 1/24 day
            if timeframe == '1h':
                # Convert hours to days (1 hour = 1/24 day)
                hours_to_days = IB_BAR_CAP / 24
                dur_unit = f"{int(hours_to_days)} D"
            else:
                dur_unit = f"{IB_BAR_CAP} D"
            
            logger.info(f"IBKR duration string: '{dur_unit}' (length: {len(dur_unit)})")

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
                    
                    # Handle timezone conversion properly
                    batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], format="%Y%m%d  %H:%M:%S")
                    
                    # Check if timestamp is already timezone-aware
                    if batch_df["timestamp"].dt.tz is None:
                        # Not timezone-aware, localize to US/Eastern then convert to UTC
                        batch_df["timestamp"] = batch_df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
                    else:
                        # Already timezone-aware, just convert to UTC
                        batch_df["timestamp"] = batch_df["timestamp"].dt.tz_convert("UTC")
                    
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
                        raise
        finally:
            # Don't disconnect - we want to reuse the connection
            pass
        
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
    
    # Note: Database saving is now handled by the caller
    # save_to_timescaledb(df, symbol, "IB", timeframe)
    
    return df

def get_unique_client_id():
    """Generate a unique client ID for IBKR connections"""
    # Use timestamp + random number to ensure uniqueness
    timestamp = int(time.time() * 1000) % 10000  # Last 4 digits of timestamp
    random_num = random.randint(100, 999)
    return timestamp + random_num

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
