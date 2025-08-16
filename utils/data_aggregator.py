"""
Data aggregation utilities for different timeframes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAggregator:
    """Handles data aggregation for different timeframes"""
    
    TIMEFRAME_MAPPING = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': 'h',
        '4h': '4h',
        '1d': 'D',
        '1w': 'W',
        '1M': 'M'
    }
    
    @staticmethod
    def get_available_symbols() -> List[str]:
        """Get list of available symbols from data folder and TimescaleDB"""
        symbols = set()
        
        # Get symbols from local data folder
        data_path = "data/ALPACA"
        if os.path.exists(data_path):
            local_symbols = [d for d in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, d))]
            symbols.update(local_symbols)
        
        # Get symbols from TimescaleDB
        try:
            from utils.timescaledb_client import get_timescaledb_client
            client = get_timescaledb_client()
            if client.connect():
                db_symbols = client.get_available_symbols()
                symbols.update(db_symbols)
                client.disconnect()
        except Exception as e:
            logger.warning(f"Could not connect to TimescaleDB: {e}")
        
        return sorted(list(symbols))
    
    @staticmethod
    def get_available_timeframes(symbol: str) -> List[str]:
        """Get available timeframes for a symbol from data folder and TimescaleDB"""
        timeframes = set()
        
        # Get timeframes from local data folder
        symbol_path = f"data/ALPACA/{symbol}"
        if os.path.exists(symbol_path):
            local_timeframes = [d for d in os.listdir(symbol_path) 
                               if os.path.isdir(os.path.join(symbol_path, d))]
            timeframes.update(local_timeframes)
        
        # Get timeframes from TimescaleDB
        try:
            from utils.timescaledb_client import get_timescaledb_client
            client = get_timescaledb_client()
            if client.connect():
                # Get timeframes for this symbol from database
                query = "SELECT DISTINCT timeframe FROM market_data WHERE symbol = %s ORDER BY timeframe"
                cursor = client.connection.cursor()
                cursor.execute(query, [symbol.upper()])
                db_timeframes = [row[0] for row in cursor.fetchall()]
                timeframes.update(db_timeframes)
                cursor.close()
                client.disconnect()
        except Exception as e:
            logger.warning(f"Could not get timeframes from TimescaleDB: {e}")
        
        return sorted(list(timeframes))
    
    @staticmethod
    def load_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data for a symbol and timeframe from local files or TimescaleDB"""
        local_df = None
        db_df = None
        
        # Try to load from local parquet files
        file_pattern = f"data/ALPACA/{symbol}/{timeframe}/*.parquet"
        files = glob.glob(file_pattern)
        
        if files:
            # Load the first matching file
            local_df = pd.read_parquet(files[0])
            
            # Convert ts_event to datetime (nanoseconds since epoch)
            if 'ts_event' in local_df.columns:
                local_df['datetime'] = pd.to_datetime(local_df['ts_event'], unit='ns')
                local_df.set_index('datetime', inplace=True)
                local_df.drop('ts_event', axis=1, inplace=True)
            elif 'datetime' in local_df.columns:
                local_df['datetime'] = pd.to_datetime(local_df['datetime'])
                local_df.set_index('datetime', inplace=True)
            elif local_df.index.dtype != 'datetime64[ns]':
                local_df.index = pd.to_datetime(local_df.index)
            
            # Handle missing volume column
            if 'volume' not in local_df.columns:
                # Create a dummy volume column if missing
                local_df['volume'] = 1000.0
            
            logger.info(f"Found {len(local_df)} records in local file for {symbol} {timeframe}")
            logger.info(f"Local date range: {local_df.index.min()} to {local_df.index.max()}")
        
        # Try to load from TimescaleDB
        try:
            from utils.timescaledb_client import get_timescaledb_client
            client = get_timescaledb_client()
            if client.connect():
                db_df = client.get_market_data(symbol, timeframe)
                if not db_df.empty:
                    logger.info(f"Raw TimescaleDB data info:")
                    logger.info(f"  Shape: {db_df.shape}")
                    logger.info(f"  Columns: {db_df.columns.tolist()}")
                    logger.info(f"  Data types: {db_df.dtypes.to_dict()}")
                    logger.info(f"  Sample data (first 3 rows):")
                    logger.info(f"    {db_df.head(3).to_dict('records')}")
                    
                    # Convert to the expected format
                    # Check which timestamp column is available
                    if 'timestamp' in db_df.columns:
                        timestamp_col = 'timestamp'
                    elif 'ts' in db_df.columns:
                        timestamp_col = 'ts'
                    else:
                        logger.warning(f"No timestamp column found in TimescaleDB data. Columns: {db_df.columns.tolist()}")
                        db_df = None
                        client.disconnect()
                    
                    # Create a new DataFrame with the correct format
                    logger.info(f"Creating DataFrame with columns: open, high, low, close, volume")
                    logger.info(f"Using timestamp column: {timestamp_col}")
                    
                    # Check data types before conversion
                    logger.info(f"Column data types before conversion:")
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in db_df.columns:
                            logger.info(f"  {col}: {db_df[col].dtype}, sample values: {db_df[col].head(3).tolist()}")
                    
                    # Convert numeric columns to proper types
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        if col in db_df.columns:
                            # Convert to numeric, coercing errors to NaN
                            db_df[col] = pd.to_numeric(db_df[col], errors='coerce')
                            logger.info(f"Converted {col} to numeric: {db_df[col].dtype}")
                    
                    # Set the index to the timestamp column
                    db_df.set_index(timestamp_col, inplace=True)
                    
                    # Keep only the columns we need
                    db_df = db_df[['open', 'high', 'low', 'close', 'volume']]
                    
                    logger.info(f"Final DataFrame info:")
                    logger.info(f"  Shape: {db_df.shape}")
                    logger.info(f"  Data types: {db_df.dtypes.to_dict()}")
                    logger.info(f"  Sample data (first 3 rows):")
                    logger.info(f"    {db_df.head(3).to_dict('records')}")
                    logger.info(f"  NaN counts: {db_df.isna().sum().to_dict()}")
                    
                    logger.info(f"Found {len(db_df)} records in TimescaleDB for {symbol} {timeframe}")
                    logger.info(f"DB date range: {db_df.index.min()} to {db_df.index.max()}")
                
                client.disconnect()
        except Exception as e:
            logger.warning(f"Could not load data from TimescaleDB: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            db_df = None
        
        # Choose the data source with more historical data
        if local_df is not None and db_df is not None:
            # Compare date ranges and choose the one with earlier start date
            local_start = local_df.index.min()
            db_start = db_df.index.min()
            
            if db_start < local_start:
                logger.info(f"Using TimescaleDB data (starts from {db_start}) over local data (starts from {local_start})")
                return db_df
            else:
                logger.info(f"Using local data (starts from {local_start}) over TimescaleDB data (starts from {db_start})")
                return local_df
        elif db_df is not None:
            logger.info(f"Using TimescaleDB data for {symbol} {timeframe}")
            return db_df
        elif local_df is not None:
            logger.info(f"Using local data for {symbol} {timeframe}")
            return local_df
        else:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return None
    
    @staticmethod
    def aggregate_data(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Aggregate data to target timeframe"""
        if target_timeframe not in DataAggregator.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        # Resample the data
        resampled = df.resample(DataAggregator.TIMEFRAME_MAPPING[target_timeframe])
        
        # Aggregate OHLCV data
        agg_data = pd.DataFrame()
        
        if 'open' in df.columns:
            agg_data['open'] = resampled['open'].first()
        if 'high' in df.columns:
            agg_data['high'] = resampled['high'].max()
        if 'low' in df.columns:
            agg_data['low'] = resampled['low'].min()
        if 'close' in df.columns:
            agg_data['close'] = resampled['close'].last()
        if 'volume' in df.columns:
            agg_data['volume'] = resampled['volume'].sum()
        
        return agg_data.dropna()
    
    @staticmethod
    def get_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for symbol and timeframe, aggregating if necessary"""
        # First try to load exact timeframe
        df = DataAggregator.load_data(symbol, timeframe)
        if df is not None and not df.empty:
            logger.info(f"Returning data for {symbol} at {timeframe} (exact match)")
            return df
        
        # If not available, try to aggregate from smaller timeframe
        available_timeframes = DataAggregator.get_available_timeframes(symbol)
        
        # Find the smallest available timeframe
        if not available_timeframes:
            logger.warning(f"No timeframes available for {symbol}")
            return None
        
        smallest_timeframe = min(available_timeframes, 
                               key=lambda x: list(DataAggregator.TIMEFRAME_MAPPING.keys()).index(x))
        
        # Only aggregate if the smallest timeframe is different from the target
        if smallest_timeframe == timeframe:
            logger.warning(f"Smallest available timeframe ({smallest_timeframe}) matches target ({timeframe}), but no data found")
            return None
        
        # Load the smallest timeframe data
        df = DataAggregator.load_data(symbol, smallest_timeframe)
        if df is None or df.empty:
            logger.warning(f"Could not load data for {symbol} at {smallest_timeframe}")
            return None
        
        # Aggregate to target timeframe
        logger.info(f"Aggregating data from {smallest_timeframe} to {timeframe}")
        return DataAggregator.aggregate_data(df, timeframe)
