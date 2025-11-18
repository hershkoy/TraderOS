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
import socket
from ib_insync import util
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
MAX_RETRIES = 5          # Increased from 3 to 5
RETRY_DELAY = 5
IB_REQUEST_DELAY = 1     # Delay between IBKR requests

# Enhanced retry configuration for timeout handling
TIMEOUT_RETRY_DELAYS = [2, 5, 10, 20, 30]  # Progressive delays for timeouts
RATE_LIMIT_RETRY_DELAYS = [5, 10, 20, 30, 60]  # Longer delays for rate limits
MAX_TIMEOUT_RETRIES = 3  # Specific retries for timeout errors
MAX_RATE_LIMIT_RETRIES = 3  # Specific retries for rate limit errors

# Set up logging (needed for connection functions)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _ensure_ib_loop():
    """Ensure the ib_insync background event loop is running."""
    global _loop_started
    if not _loop_started:
        util.startLoop()
        _loop_started = True


# Global IBKR connection manager
_ib_connection = None
_ib_lock = threading.Lock()
_loop_started = False

def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    """
    Check if a port is listening on the given host.
    
    Args:
        host: Hostname or IP address
        port: Port number to check
        timeout: Connection timeout in seconds
    
    Returns:
        True if port is listening, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def find_listening_port(host: str = "127.0.0.1", ports: list = None) -> int:
    """
    Find the first port that is listening from a list of ports.
    
    Args:
        host: Hostname or IP address (default: 127.0.0.1)
        ports: List of ports to check (default: [7497, 7496, 4002, 4001])
    
    Returns:
        Port number if found, None otherwise
    """
    if ports is None:
        ports = [7497, 7496, 4002, 4001]  # TWS paper, TWS live, Gateway paper, Gateway live
    
    for port in ports:
        if is_port_listening(host, port):
            logger.info(f"Found listening port: {host}:{port}")
            return port
    
    return None

def get_ib_connection():
    """Get or create a shared IBKR connection"""
    global _ib_connection
    
    with _ib_lock:
        if _ib_connection is None or not _ib_connection.isConnected():
            _ensure_ib_loop()
            from ib_insync import IB
            _ib_connection = IB()
            client_id = random.randint(1000, 9999)
            logger.info(f"Creating new IBKR connection with client ID {client_id}")
            
            # Find which port is listening
            HOST = "127.0.0.1"
            PORTS = [7497, 7496, 4002, 4001]  # TWS paper, TWS live, Gateway paper, Gateway live
            
            logger.info(f"Checking for listening ports on {HOST}...")
            port = find_listening_port(HOST, PORTS)
            
            if port is None:
                raise ConnectionError(
                    f"No IBKR service found listening on {HOST}. "
                    f"Checked ports: {PORTS}. "
                    f"Please ensure TWS or IB Gateway is running and API is enabled."
                )
            
            # Connect to the listening port with retry logic
            max_connection_attempts = 3
            connected = False
            
            logger.info(f"Connecting to {HOST}:{port}...")
            for attempt in range(max_connection_attempts):
                # Create a fresh IB instance for each attempt to avoid state issues
                try:
                    # Clean up previous connection if it exists
                    if _ib_connection is not None:
                        try:
                            if _ib_connection.isConnected():
                                _ib_connection.disconnect()
                        except:
                            pass
                        _ib_connection = None
                    
                    # Create fresh IB instance
                    _ib_connection = IB()
                    
                    # Use a unique client ID per attempt to avoid conflicts
                    attempt_client_id = client_id + attempt
                    logger.info(f"Connection attempt {attempt + 1}/{max_connection_attempts} with client ID {attempt_client_id}")
                    
                    # Connect with increased timeout (default is 4 seconds, increase to 15 for handshake)
                    _ib_connection.connect(HOST, port, clientId=attempt_client_id, timeout=15)
                    #HOST = '127.0.0.1'
                    #PORT = 7496   
                    #CLIENT_ID = 1
                    #_ib_connection.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
                    
                    # Wait a moment for API to fully initialize and complete handshake
                    time.sleep(2)
                    
                    # Verify connection is actually established
                    if _ib_connection.isConnected():
                        logger.info(f"IBKR connection established on port {port} with client ID {attempt_client_id}")
                        connected = True
                        break
                    else:
                        raise ConnectionError("Connection established but isConnected() returned False")
                        
                except TimeoutError as e:
                    # Clean up failed connection attempt
                    _ib_connection = None
                    
                    if attempt < max_connection_attempts - 1:
                        logger.warning(f"Connection attempt {attempt + 1} to port {port} timed out during API handshake. Retrying with longer delay...")
                        time.sleep(5)  # Longer delay for timeout errors to let API recover
                    else:
                        logger.error(f"Failed to connect to {HOST}:{port} after {max_connection_attempts} attempts: Connection timeout")
                        raise ConnectionError(
                            f"Connection to IBKR on {HOST}:{port} timed out after {max_connection_attempts} attempts. "
                            f"The socket connects but the API handshake times out. "
                            f"Please check:\n"
                            f"  1. TWS/Gateway API is enabled in settings\n"
                            f"  2. 'Enable ActiveX and Socket Clients' is checked\n"
                            f"  3. No firewall is blocking the connection\n"
                            f"  4. Try restarting TWS/Gateway"
                        )
                except Exception as e:
                    # Clean up failed connection attempt
                    _ib_connection = None
                        
                    if attempt < max_connection_attempts - 1:
                        logger.warning(f"Connection attempt {attempt + 1} to port {port} failed: {e}. Retrying...")
                        time.sleep(3)
                    else:
                        logger.error(f"Failed to connect to {HOST}:{port} after {max_connection_attempts} attempts: {e}")
                        raise ConnectionError(f"Failed to connect to IBKR on {HOST}:{port} after {max_connection_attempts} attempts: {e}")
            
            if not connected:
                raise ConnectionError(f"Failed to connect to IBKR on {HOST}:{port}")
        
        return _ib_connection

def create_ib_contract_with_primary_exchange(symbol):
    """Create an IB contract with proper primary exchange to avoid ambiguity"""
    from ib_insync import Stock, Index
    
    # Define index symbols that need special handling (actual indices, not ETFs)
    # SPX, RUT, NDX, VIX, DJX are indices
    # SPY, QQQ, IWM, DIA are ETFs and should be treated as stocks
    INDEX_SYMBOLS = {'SPX', 'RUT', 'NDX', 'VIX', 'DJX'}
    
    # Check if this is an index
    if symbol.upper() in INDEX_SYMBOLS:
        # For indices, use Index class with CBOE exchange
        logger.info(f"Detected index symbol {symbol}, using Index contract with CBOE exchange")
        return Index(symbol=symbol, exchange='CBOE', currency='USD')
    
    # Define primary exchanges for common symbols to avoid ambiguity
    # Major ETFs (SPY, QQQ, IWM, DIA) trade on ARCA - use SMART routing for these
    NASDAQ_SYMBOLS = {
        'STLD', 'STX', 'SWKS', 'TEAM', 'TECH', 'TER', 'ZS',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        'QQQ'  # QQQ is an ETF that trades on NASDAQ
    }
    
    NYSE_SYMBOLS = {
        'STE', 'STT', 'STZ', 'SWK', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY',
        'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'PFE', 'BAC', 'KO', 'PEP'
    }
    
    # Major ETFs that trade on ARCA - use SMART routing (no primaryExchange needed)
    ARCA_ETFS = {'SPY', 'DIA', 'IWM'}  # These ETFs trade on NYSE ARCA
    
    # Determine primary exchange
    if symbol in NASDAQ_SYMBOLS:
        primary_exchange = 'NASDAQ'
    elif symbol in NYSE_SYMBOLS:
        primary_exchange = 'NYSE'
    elif symbol in ARCA_ETFS:
        # For ARCA ETFs, use SMART routing without primaryExchange
        # IB will route to ARCA automatically
        primary_exchange = None
    else:
        # For unknown symbols, let IB resolve it
        primary_exchange = None
    
    if primary_exchange:
        logger.info(f"Creating contract for {symbol} with primary exchange {primary_exchange}")
        return Stock(symbol=symbol, exchange='SMART', currency='USD', primaryExchange=primary_exchange)
    else:
        if symbol in ARCA_ETFS:
            logger.info(f"Creating contract for ETF {symbol} using SMART routing (will route to ARCA)")
        else:
            logger.info(f"Creating contract for {symbol} without primary exchange (will be resolved by IB)")
        return Stock(symbol=symbol, exchange='SMART', currency='USD')

def create_ib_contract_from_cache(symbol, con_id, primary_exchange):
    """Create an IB contract directly from cached information (no qualification needed)"""
    from ib_insync import Stock, Index
    
    # Define index symbols that need special handling (actual indices, not ETFs)
    # SPX, RUT, NDX, VIX, DJX are indices
    # SPY, QQQ, IWM, DIA are ETFs and should be treated as stocks
    INDEX_SYMBOLS = {'SPX', 'RUT', 'NDX', 'VIX', 'DJX'}
    
    # Check if this is an index
    if symbol.upper() in INDEX_SYMBOLS:
        logger.info(f"Creating pre-qualified index contract for {symbol}: conId {con_id}, exchange {primary_exchange or 'CBOE'}")
        contract = Index(symbol=symbol, exchange=primary_exchange or 'CBOE', currency='USD')
        contract.conId = con_id
        return contract
    
    logger.info(f"Creating pre-qualified contract for {symbol}: conId {con_id}, exchange {primary_exchange}")
    contract = Stock(symbol=symbol, exchange='SMART', currency='USD', primaryExchange=primary_exchange)
    contract.conId = con_id
    return contract

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

def reset_ib_connection():
    """Reset the IB connection when we encounter persistent issues"""
    global _ib_connection
    
    with _ib_lock:
        if _ib_connection and _ib_connection.isConnected():
            try:
                logger.warning("[CONNECTION] Resetting IBKR connection due to persistent issues")
                _ib_connection.disconnect()
                logger.info("[CONNECTION] IBKR connection disconnected")
            except Exception as e:
                logger.warning(f"[CONNECTION] Error disconnecting IBKR: {e}")
            finally:
                _ib_connection = None
        
        # Force creation of new connection on next get_ib_connection() call
        logger.info("[CONNECTION] IBKR connection will be recreated on next request")

SAVE_DIR = Path("./data")
SAVE_DIR.mkdir(exist_ok=True)

# Logger is already defined above for connection functions

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
        # Ensure connection to the database before inserting data
        if not client.ensure_connection():
            logger.error(f"✗ Failed to connect to TimescaleDB")
            return False
        
        if client.insert_market_data(df, symbol, provider, timeframe):
            logger.info(f"✔ Saved {len(df)} records to TimescaleDB for {symbol} {timeframe}")
            return True
        else:
            logger.error(f"✗ Failed to save to TimescaleDB for {symbol} {timeframe}")
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
            if timeframe == "1m":
                start_time = end_time - timedelta(days=365)  # Try 1 year for minute data
            elif timeframe == "1h":
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

def fetch_smart_max_from_alpaca(symbol, timeframe, start_date=None):
    """
    Smart version of max bars that first detects what's available
    then fetches that amount using the reliable single request method.
    """
    logger.info(f"🔍 Detecting maximum available bars for {symbol} @ {timeframe}...")
    
    # If start_date is provided, use it; otherwise detect from available data
    if start_date:
        if isinstance(start_date, str):
            start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        else:
            start_time = start_date
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        end_time = datetime.now(timezone.utc)
        logger.info(f"Using provided start date: {start_time}, fetching until {end_time}")
        # Estimate max_available based on timeframe (will be adjusted by actual fetch)
        if timeframe == "1m":
            days = (end_time - start_time).days
            max_available = int(days * 390 * 5 / 7)  # Rough estimate
        elif timeframe == "1h":
            days = (end_time - start_time).days
            max_available = int(days * 6.5 * 5 / 7)  # Rough estimate
        else:  # 1d
            days = (end_time - start_time).days
            max_available = int(days * 5 / 7)  # Rough estimate
    else:
        # Get both the count and the actual time range from detection
        detection_result = detect_max_available_bars_with_range(symbol, timeframe)
        
        if detection_result is None:
            logger.warning(f"No data available for {symbol} @ {timeframe}")
            return None
        
        max_available, start_time, end_time = detection_result
    
    logger.info(f"📊 Fetching up to {max_available} bars for {symbol} @ {timeframe}")
    logger.info(f"📅 Time range: {start_time.date()} to {end_time.date()}")
    logger.info(f"🚀 Fetching bars using single request method...")
    
    # Use the reliable single request method with the detected amount and time range
    return fetch_single_alpaca_request_with_range(symbol, max_available, timeframe, start_time, end_time)

# ─────────────────────────────
# ALPACA LOGIC
# ─────────────────────────────
def fetch_from_alpaca(symbol, bars, timeframe, start_date=None):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    if bars == "max":
        logger.info(f"Fetching maximum available bars from Alpaca for {symbol} @ {timeframe}...")
        return fetch_smart_max_from_alpaca(symbol, timeframe, start_date=start_date)
    else:
        logger.info(f"Fetching {bars} bars from Alpaca for {symbol} @ {timeframe}...")
        return fetch_single_alpaca_request(symbol, bars, timeframe, start_date=start_date)

def fetch_single_alpaca_request_with_range(symbol, bars, timeframe, start_time, end_time):
    """Fetch a single request from Alpaca using a specific time range."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    tf_map = {
        "1m": TimeFrame.Minute,
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

def fetch_single_alpaca_request(symbol, bars, timeframe, start_date=None):
    """Fetch a single request from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    tf_map = {
        "1m": TimeFrame.Minute,
        "1h": TimeFrame.Hour,
        "1d": TimeFrame.Day,
    }

    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe for Alpaca: {timeframe}")

    client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
    )

    # Use provided start_date if available, otherwise calculate from bars
    if start_date:
        if isinstance(start_date, str):
            start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        else:
            start_time = start_date
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        logger.info(f"Using provided start date: {start_time}")
    else:
        # Calculate a reasonable time period to get the requested number of bars
        # For minute data: account for ~6.5 hours per trading day, ~390 minutes per day
        # For hourly data: account for ~6.5 hours per trading day, weekends, holidays
        # For daily data: account for weekends and holidays
        if timeframe == "1m":
            # Assume ~390 trading minutes per day (6.5 hours * 60), ~5 trading days per week
            # So we need roughly (bars / 390) * 7/5 days to get enough bars
            days_needed = max(1, int((bars / 390) * 7 / 5))
            start_time = datetime.now(timezone.utc) - timedelta(days=days_needed)
        elif timeframe == "1h":
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
        "1m": TimeFrame.Minute,
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
        if timeframe == "1m":
            # For 1-minute data, go back by the number of minutes (capped at ALPACA_BAR_CAP)
            start_time = end_time - timedelta(minutes=ALPACA_BAR_CAP)
        elif timeframe == "1h":
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
def fetch_from_ib(symbol, bars, timeframe, contract_info=None, start_date=None):
    """Fetch historical bars from IBKR."""
    from ib_insync import IB, Stock

    if timeframe not in ["1m", "1h", "1d"]:
        raise ValueError(f"Unsupported timeframe for IBKR: {timeframe}")

    if bars == "max":
        return fetch_max_from_ib(symbol, timeframe, contract_info, start_date=start_date)
    
    # For fixed number of bars
    ib = get_ib_connection()
    
    try:
        # Use pre-qualified contract info if available, otherwise qualify
        if contract_info and contract_info.get('con_id') and contract_info.get('primary_exchange'):
            logger.info(f"Using pre-qualified contract for {symbol}: conId {contract_info['con_id']}, exchange {contract_info['primary_exchange']}")
            contract = create_ib_contract_from_cache(symbol, contract_info['con_id'], contract_info['primary_exchange'])
        else:
            # Fallback: create and qualify contract
            logger.info(f"No pre-qualified contract info for {symbol}, qualifying contract...")
            contract = create_ib_contract_with_primary_exchange(symbol)
            
            # Qualify the contract to resolve ambiguity and get conId + primaryExchange
            logger.info(f"Qualifying contract for {symbol} to resolve ambiguity...")
            qualified_contracts = ib.qualifyContracts(contract)
            if qualified_contracts:
                contract = qualified_contracts[0]
                logger.info(f"Contract qualified: {symbol} -> conId: {contract.conId}, primaryExchange: {contract.primaryExchange}")
            else:
                logger.error(f"No qualified contracts found for {symbol}")
                logger.error(f"This usually means:")
                logger.error(f"1. Symbol {symbol} doesn't exist in IB")
                logger.error(f"2. No market data permissions for this symbol")
                logger.error(f"3. Symbol format is incorrect")
                return None
        
        # Map timeframe to IB bar size setting
        if timeframe == "1m":
            bar_size = "1 min"
        elif timeframe == "1h":
            bar_size = "1 hour"
        else:  # 1d
            bar_size = "1 day"
        
        # Handle start_date if provided
        if start_date:
            if isinstance(start_date, str):
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            else:
                start_dt = start_date
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            
            # Calculate duration from start_date to now
            end_dt = datetime.now(timezone.utc)
            duration_delta = end_dt - start_dt
            days_diff = duration_delta.days
            
            # IBKR duration format: use days, weeks, months, or years
            if days_diff >= 365:
                years = days_diff // 365
                dur_unit = f"{years} Y"
            elif days_diff >= 30:
                months = days_diff // 30
                dur_unit = f"{months} M"
            elif days_diff >= 7:
                weeks = days_diff // 7
                dur_unit = f"{weeks} W"
            else:
                dur_unit = f"{max(1, days_diff)} D"
            
            # Format endDateTime for IB (YYYYMMDD HH:MM:SS)
            end_date_str = end_dt.strftime("%Y%m%d %H:%M:%S")
            logger.info(f"Using provided start date: {start_dt}, duration: {dur_unit}, end: {end_date_str}")
        else:
            # Ensure proper spacing for IBKR duration format
            # IBKR only supports: S (seconds), D (days), W (weeks), M (months), Y (years)
            # For minute data, convert to days: 1 minute = 1/1440 day
            # For hourly data, convert to days: 1 hour = 1/24 day
            if timeframe == '1m':
                # Convert minutes to days (1 minute = 1/1440 day)
                minutes_to_days = bars / 1440
                if minutes_to_days < 1:
                    # For less than 1 day, use minutes (but IB doesn't support minutes in duration)
                    # So use days with fractional calculation, but IB needs integer days
                    # For very short periods, use 1 day minimum
                    dur_unit = "1 D"
                else:
                    dur_unit = f"{int(minutes_to_days)} D"
            elif timeframe == '1h':
                # Convert hours to days (1 hour = 1/24 day)
                hours_to_days = bars / 24
                if hours_to_days < 1:
                    # For less than 1 day, use hours instead
                    dur_unit = f"{bars} H"
                else:
                    dur_unit = f"{int(hours_to_days)} D"
            else:  # 1d
                dur_unit = f"{bars} D"
            end_date_str = ""  # Use current time
        
        logger.info(f"IBKR duration string: '{dur_unit}' (length: {len(dur_unit)})")

        # Get data with intelligent retry logic
        def fetch_bars_operation():
            return ib.reqHistoricalData(
                contract,
                endDateTime=end_date_str,
                durationStr=dur_unit,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        
        bars_data = intelligent_retry_with_backoff(
            fetch_bars_operation, 
            f"IBKR data fetch for {symbol} ({timeframe}) with qualified contract",
            reset_connection_on_failure=True
        )
        
        if bars_data is None:
            logger.error(f"Failed to fetch data from IBKR for {symbol} after all retries")
            return None
        
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
    finally:
        # Don't disconnect - we want to reuse the connection
        pass

def fetch_max_from_ib(symbol, timeframe, contract_info=None, start_date=None):
    """Fetch maximum available historical data from IBKR by looping through requests."""
    from ib_insync import IB, Stock

    if timeframe not in ["1m", "1h", "1d"]:
        raise ValueError(f"Unsupported timeframe for IBKR: {timeframe}")

    all_data = []
    total_bars_fetched = 0
    
    # Handle start_date if provided
    if start_date:
        if isinstance(start_date, str):
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_dt = start_date
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
        # Convert to IB format (YYYYMMDD HH:MM:SS) - this will be the initial end_date
        # We'll work backwards from now to start_date
        end_date = ""  # Start from most recent data
        target_start_date = start_dt
        logger.info(f"Using provided start date: {start_dt}, will fetch from {start_dt} to now")
    else:
        end_date = ""  # Start from most recent data
        target_start_date = None
    
    logger.info(f"Starting maximum data fetch for {symbol} @ {timeframe}")
    
    # Create contract once at the beginning if we have pre-qualified info
    if contract_info and contract_info.get('con_id') and contract_info.get('primary_exchange'):
        logger.info(f"Using pre-qualified contract for {symbol}: conId {contract_info['con_id']}, exchange {contract_info['primary_exchange']}")
        base_contract = create_ib_contract_from_cache(symbol, contract_info['con_id'], contract_info['primary_exchange'])
        logger.info(f"Pre-qualified contract ready: {symbol} -> conId: {base_contract.conId}, primaryExchange: {base_contract.primaryExchange}")
    else:
        logger.info(f"No pre-qualified contract info for {symbol}, will qualify in each batch")
        base_contract = None
    
    while True:
        ib = get_ib_connection()
        
        try:
            # Use pre-qualified contract or create/qualify new one
            if base_contract:
                contract = base_contract
                logger.info(f"Reusing pre-qualified contract for {symbol}")
            else:
                # Try to create contract with primary exchange first
                contract = create_ib_contract_with_primary_exchange(symbol)
                
                # Qualify the contract to resolve ambiguity and get conId + primaryExchange
                logger.info(f"Qualifying contract for {symbol} to resolve ambiguity...")
                qualified_contracts = ib.qualifyContracts(contract)
                if qualified_contracts:
                    contract = qualified_contracts[0]
                    logger.info(f"Contract qualified: {symbol} -> conId: {contract.conId}, primaryExchange: {contract.primaryExchange}")
                else:
                    logger.error(f"No qualified contracts found for {symbol}")
                    logger.error(f"This usually means:")
                    logger.error(f"1. Symbol {symbol} doesn't exist in IB")
                    logger.error(f"2. No market data permissions for this symbol")
                    logger.error(f"3. Symbol format is incorrect")
                    break
            
            # Map timeframe to IB bar size setting
            if timeframe == "1m":
                bar_size = "1 min"
            elif timeframe == "1h":
                bar_size = "1 hour"
            else:  # 1d
                bar_size = "1 day"
            
            # Ensure proper spacing for IBKR duration format
            # IBKR only supports: S (seconds), D (days), W (weeks), M (months), Y (years)
            # For minute data, convert to days: 1 minute = 1/1440 day
            # For hourly data, convert to days: 1 hour = 1/24 day
            if timeframe == '1m':
                # Convert minutes to days (1 minute = 1/1440 day)
                minutes_to_days = IB_BAR_CAP / 1440
                dur_unit = f"{max(1, int(minutes_to_days))} D"  # Minimum 1 day
            elif timeframe == '1h':
                # Convert hours to days (1 hour = 1/24 day)
                hours_to_days = IB_BAR_CAP / 24
                dur_unit = f"{int(hours_to_days)} D"
            else:  # 1d
                dur_unit = f"{IB_BAR_CAP} D"
            
            # Log what date range is being requested
            if end_date:
                logger.info(f"Requesting data ending at: {end_date}, duration: {dur_unit}")
            else:
                logger.info(f"Requesting most recent data, duration: {dur_unit}")
            
            if target_start_date:
                logger.info(f"Target start date: {target_start_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            logger.info(f"IBKR duration string: '{dur_unit}' (length: {len(dur_unit)})")

            # Get data with intelligent retry logic
            def fetch_batch_operation():
                return ib.reqHistoricalData(
                    contract,
                    endDateTime=end_date,
                    durationStr=dur_unit,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
            
            bars_data = intelligent_retry_with_backoff(
                fetch_batch_operation,
                f"IBKR batch fetch for {symbol} ({timeframe}) - batch {len(all_data) + 1} with qualified contract",
                reset_connection_on_failure=True
            )
            
            if bars_data is None:
                logger.error(f"Failed to fetch batch from IBKR for {symbol} after all retries")
                # Continue to next batch instead of failing completely
                break
            
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
        finally:
            # Don't disconnect - we want to reuse the connection
            pass
        
        if batch_data is None or batch_data.empty:
            break
        
        # Add to collection
        all_data.append(batch_data)
        total_bars_fetched += len(batch_data)
        
        # Get date range of this batch
        if len(batch_data) > 0:
            batch_min = batch_data['timestamp'].min()
            batch_max = batch_data['timestamp'].max()
            logger.info(f"Fetched batch of {len(batch_data)} bars. Total: {total_bars_fetched}")
            logger.info(f"  Batch date range: {batch_min.strftime('%Y-%m-%d %H:%M:%S %Z')} to {batch_max.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Check if we've reached the target start date
            if target_start_date:
                if batch_min <= target_start_date:
                    logger.info(f"Reached target start date ({target_start_date.strftime('%Y-%m-%d %H:%M:%S %Z')}). Stopping fetch.")
                    # Filter out data before target_start_date
                    batch_data = batch_data[batch_data['timestamp'] >= target_start_date]
                    if len(batch_data) < len(all_data[-1]):
                        all_data[-1] = batch_data
                        logger.info(f"Filtered batch to {len(batch_data)} bars after target date")
                    break
            
            # Update end_date for next batch (go further back in time)
            end_date = batch_min.strftime("%Y%m%d %H:%M:%S")
        else:
            logger.info(f"Fetched batch of {len(batch_data)} bars. Total: {total_bars_fetched}")
        
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
    
    # Filter to target start date if specified
    if target_start_date and len(combined_df) > 0:
        original_count = len(combined_df)
        combined_df = combined_df[combined_df['timestamp'] >= target_start_date]
        if len(combined_df) < original_count:
            logger.info(f"Filtered combined data from {original_count} to {len(combined_df)} bars after target start date")
    
    logger.info(f"Combined {len(combined_df)} unique bars from {len(all_data)} batches")
    
    if len(combined_df) > 0:
        final_min = combined_df['timestamp'].min()
        final_max = combined_df['timestamp'].max()
        logger.info(f"Final data date range: {final_min.strftime('%Y-%m-%d %H:%M:%S %Z')} to {final_max.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if target_start_date:
            logger.info(f"Requested start date: {target_start_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            if final_min > target_start_date:
                logger.warning(f"Earliest data ({final_min.strftime('%Y-%m-%d %H:%M:%S %Z')}) is after requested start date ({target_start_date.strftime('%Y-%m-%d %H:%M:%S %Z')})")
    
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

def intelligent_retry_with_backoff(operation_func, operation_name, max_retries=None, 
                                 timeout_retries=None, rate_limit_retries=None,
                                 reset_connection_on_failure=False):
    """
    Intelligent retry function with different strategies for different error types.
    
    Args:
        operation_func: Function to retry
        operation_name: Name of the operation for logging
        max_retries: Maximum total retries (default: MAX_RETRIES)
        timeout_retries: Maximum timeout-specific retries (default: MAX_TIMEOUT_RETRIES)
        rate_limit_retries: Maximum rate-limit-specific retries (default: MAX_RATE_LIMIT_RETRIES)
        reset_connection_on_failure: Whether to reset IB connection after all retries fail
    
    Returns:
        Result of operation_func or None if all retries exhausted
    """
    if max_retries is None:
        max_retries = MAX_RETRIES
    if timeout_retries is None:
        timeout_retries = MAX_TIMEOUT_RETRIES
    if rate_limit_retries is None:
        rate_limit_retries = MAX_RATE_LIMIT_RETRIES
    
    timeout_attempts = 0
    rate_limit_attempts = 0
    total_attempts = 0
    
    while total_attempts < max_retries:
        try:
            return operation_func()
        except Exception as e:
            total_attempts += 1
            error_str = str(e).lower()
            
            # Handle timeout errors specifically
            if "timeout" in error_str or "timed out" in error_str:
                if timeout_attempts < timeout_retries:
                    timeout_attempts += 1
                    delay = TIMEOUT_RETRY_DELAYS[min(timeout_attempts - 1, len(TIMEOUT_RETRY_DELAYS) - 1)]
                    logger.warning(f"[TIMEOUT] {operation_name} attempt {timeout_attempts}/{timeout_retries} failed. "
                                f"Retrying in {delay}s... (Error: {e})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"[TIMEOUT] {operation_name} failed after {timeout_attempts} timeout retries: {e}")
                    if reset_connection_on_failure:
                        reset_ib_connection()
                    return None
            
            # Handle rate limit errors specifically
            elif "rate limit" in error_str or "throttle" in error_str or "too many requests" in error_str:
                if rate_limit_attempts < rate_limit_retries:
                    rate_limit_attempts += 1
                    delay = RATE_LIMIT_RETRY_DELAYS[min(rate_limit_attempts - 1, len(RATE_LIMIT_RETRY_DELAYS) - 1)]
                    logger.warning(f"[RATE_LIMIT] {operation_name} attempt {rate_limit_attempts}/{rate_limit_retries} failed. "
                                f"Retrying in {delay}s... (Error: {e})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"[RATE_LIMIT] {operation_name} failed after {rate_limit_attempts} rate limit retries: {e}")
                    if reset_connection_on_failure:
                        reset_ib_connection()
                    return None
            
            # Handle other errors with standard retry logic
            elif total_attempts < max_retries:
                delay = RETRY_DELAY * total_attempts  # Progressive delay
                logger.warning(f"[RETRY] {operation_name} attempt {total_attempts}/{max_retries} failed: {e}. "
                            f"Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                logger.error(f"[FAILED] {operation_name} failed after {max_retries} total attempts: {e}")
                if reset_connection_on_failure:
                    reset_ib_connection()
                return None
    
    return None

# ─────────────────────────────
# MAIN ENTRY
# ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical bars from Alpaca or IBKR.")
    parser.add_argument("--symbol", required=True, help="Symbol to fetch (e.g. NFLX)")
    parser.add_argument("--provider", required=True, choices=["alpaca", "ib"], help="Data provider")
    parser.add_argument("--timeframe", required=True, choices=["1m", "1h", "1d"], help="Timeframe to fetch")
    parser.add_argument("--bars", default=1000, help="Number of bars to fetch or 'max' for maximum available")
    parser.add_argument("--since", type=str, help="Start date for fetching data (format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

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
        df = None
        if provider == "alpaca":
            if bars == "max":
                df = fetch_from_alpaca(symbol, bars, timeframe, start_date=args.since)
            else:
                bars = min(bars, ALPACA_BAR_CAP)
                df = fetch_from_alpaca(symbol, bars, timeframe, start_date=args.since)
        elif provider == "ib":
            if bars == "max":
                df = fetch_from_ib(symbol, bars, timeframe, start_date=args.since)
            else:
                bars = min(bars, IB_BAR_CAP)
                df = fetch_from_ib(symbol, bars, timeframe, start_date=args.since)
        
        # Save to database if data was fetched
        if df is not None and not df.empty:
            provider_upper = provider.upper()
            logger.info(f"Saving {len(df)} records to TimescaleDB for {symbol} {timeframe} from {provider_upper}...")
            if save_to_timescaledb(df, symbol, provider_upper, timeframe):
                logger.info(f"Successfully saved data to database")
            else:
                logger.error(f"Failed to save data to database")
        elif df is None or df.empty:
            logger.warning("No data was fetched, nothing to save")
            
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        exit(1)
