"""
TimescaleDB Data Loader for BackTrader Framework
Replaces Parquet file loading with database queries
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import logging
from utils.timescaledb_client import get_timescaledb_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_timescaledb_data(symbol: str, timeframe: str, provider: Optional[str] = None,
                         start_date: Optional[str] = None, end_date: Optional[str] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load market data from TimescaleDB for BackTrader
    
    Args:
        symbol: Stock symbol (e.g., 'NFLX')
        timeframe: Time interval ('1h', '1d')
        provider: Data provider ('ALPACA', 'IB') - optional filter
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        limit: Maximum number of records to return (optional)
    
    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    try:
        client = get_timescaledb_client()
        
        # Convert date strings to datetime objects
        start_time = None
        end_time = None
        
        if start_date:
            start_time = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        if end_date:
            end_time = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Get data from TimescaleDB
        df = client.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            provider=provider,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")
        
        # Convert to BackTrader format
        df_bt = pd.DataFrame({
            'datetime': df['timestamp'].dt.tz_localize(None),  # Remove timezone for BackTrader
            'open': df['open'].astype(float),
            'high': df['high'].astype(float),
            'low': df['low'].astype(float),
            'close': df['close'].astype(float),
            'volume': df['volume'].astype(float),
        })
        
        # Sort by datetime and set as index
        df_bt = df_bt.sort_values('datetime').reset_index(drop=True)
        df_bt.set_index('datetime', inplace=True)
        
        # Filter out future dates
        now = pd.Timestamp.now()
        df_bt = df_bt[df_bt.index <= now]
        
        if len(df_bt) < 10:
            raise ValueError(f"Not enough data after filtering: {len(df_bt)} rows")
        
        # Remove duplicate timestamps
        if df_bt.index.duplicated().any():
            logger.warning("Found duplicate timestamps, removing duplicates")
            df_bt = df_bt[~df_bt.index.duplicated(keep='first')]
        
        logger.info(f"Loaded {len(df_bt)} data points from {df_bt.index.min()} to {df_bt.index.max()}")
        return df_bt
        
    except Exception as e:
        logger.error(f"Failed to load data from TimescaleDB: {e}")
        raise

def load_timescaledb_1h(symbol: str, provider: Optional[str] = None,
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load 1-hour data from TimescaleDB (replaces load_parquet_1h)
    
    Args:
        symbol: Stock symbol
        provider: Data provider (optional)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        limit: Maximum number of records (optional)
    
    Returns:
        DataFrame with hourly OHLCV data
    """
    return load_timescaledb_data(symbol, '1h', provider, start_date, end_date, limit)

def load_timescaledb_daily(symbol: str, provider: Optional[str] = None,
                          start_date: Optional[str] = None, end_date: Optional[str] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load daily data from TimescaleDB (replaces load_daily_data)
    
    Args:
        symbol: Stock symbol
        provider: Data provider (optional)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        limit: Maximum number of records (optional)
    
    Returns:
        DataFrame with daily OHLCV data
    """
    return load_timescaledb_data(symbol, '1d', provider, start_date, end_date, limit)

def resample_hourly_to_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly data to daily (fallback if daily data not available)
    
    Args:
        df_1h: DataFrame with hourly data
    
    Returns:
        DataFrame with daily OHLCV data
    """
    df_daily = df_1h.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    logger.info(f"Resampled to {len(df_daily)} daily bars from {df_daily.index.min()} to {df_daily.index.max()}")
    return df_daily

def get_available_data() -> dict:
    """
    Get summary of available data in TimescaleDB
    
    Returns:
        Dictionary with data summary
    """
    try:
        client = get_timescaledb_client()
        summary = client.get_data_summary()
        
        # Get available symbols
        symbols = client.get_available_symbols()
        
        return {
            'summary': summary,
            'symbols': symbols
        }
    except Exception as e:
        logger.error(f"Failed to get available data: {e}")
        return {'summary': {}, 'symbols': []}

def list_available_symbols(provider: Optional[str] = None, timeframe: Optional[str] = None) -> list:
    """
    List available symbols in the database
    
    Args:
        provider: Filter by provider (optional)
        timeframe: Filter by timeframe (optional)
    
    Returns:
        List of available symbols
    """
    try:
        client = get_timescaledb_client()
        return client.get_available_symbols(provider, timeframe)
    except Exception as e:
        logger.error(f"Failed to list symbols: {e}")
        return []

# Backward compatibility functions
def load_parquet_1h(parquet_path: Path) -> pd.DataFrame:
    """
    Backward compatibility function - now loads from TimescaleDB
    Extracts symbol and timeframe from the old parquet path format
    """
    # Parse the old parquet path format: data/PROVIDER/SYMBOL/TIMEFRAME/symbol_timeframe.parquet
    path_parts = parquet_path.parts
    
    if len(path_parts) >= 4:
        provider = path_parts[-4]  # e.g., 'ALPACA'
        symbol = path_parts[-3]    # e.g., 'NFLX'
        timeframe = path_parts[-2] # e.g., '1h'
        
        logger.info(f"Converting parquet path to TimescaleDB query: {symbol} {timeframe} from {provider}")
        return load_timescaledb_1h(symbol, provider)
    else:
        raise ValueError(f"Invalid parquet path format: {parquet_path}")

def load_daily_data(parquet_path: Path) -> pd.DataFrame:
    """
    Backward compatibility function - now loads from TimescaleDB
    Extracts symbol and timeframe from the old parquet path format
    """
    # Parse the old parquet path format: data/PROVIDER/SYMBOL/TIMEFRAME/symbol_timeframe.parquet
    path_parts = parquet_path.parts
    
    if len(path_parts) >= 4:
        provider = path_parts[-4]  # e.g., 'ALPACA'
        symbol = path_parts[-3]    # e.g., 'NFLX'
        timeframe = path_parts[-2] # e.g., '1h'
        
        logger.info(f"Converting parquet path to TimescaleDB query: {symbol} {timeframe} from {provider}")
        
        # Try to get daily data directly first
        try:
            return load_timescaledb_daily(symbol, provider)
        except ValueError:
            # If daily data not available, resample from hourly
            logger.info(f"Daily data not available, resampling from hourly data")
            df_1h = load_timescaledb_1h(symbol, provider)
            return resample_hourly_to_daily(df_1h)
    else:
        raise ValueError(f"Invalid parquet path format: {parquet_path}")
