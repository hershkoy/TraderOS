"""
Data aggregation utilities for different timeframes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import glob


class DataAggregator:
    """Handles data aggregation for different timeframes"""
    
    TIMEFRAME_MAPPING = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W',
        '1M': '1M'
    }
    
    @staticmethod
    def get_available_symbols() -> List[str]:
        """Get list of available symbols from data folder"""
        symbols = []
        data_path = "data/ALPACA"
        if os.path.exists(data_path):
            symbols = [d for d in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, d))]
        return sorted(symbols)
    
    @staticmethod
    def get_available_timeframes(symbol: str) -> List[str]:
        """Get available timeframes for a symbol"""
        timeframes = []
        symbol_path = f"data/ALPACA/{symbol}"
        if os.path.exists(symbol_path):
            timeframes = [d for d in os.listdir(symbol_path) 
                         if os.path.isdir(os.path.join(symbol_path, d))]
        return sorted(timeframes)
    
    @staticmethod
    def load_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data for a symbol and timeframe"""
        file_pattern = f"data/ALPACA/{symbol}/{timeframe}/*.parquet"
        files = glob.glob(file_pattern)
        
        if not files:
            return None
        
        # Load the first matching file
        df = pd.read_parquet(files[0])
        
        # Convert ts_event to datetime (nanoseconds since epoch)
        if 'ts_event' in df.columns:
            df['datetime'] = pd.to_datetime(df['ts_event'], unit='ns')
            df.set_index('datetime', inplace=True)
            df.drop('ts_event', axis=1, inplace=True)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif df.index.dtype != 'datetime64[ns]':
            df.index = pd.to_datetime(df.index)
        
        # Handle missing volume column
        if 'volume' not in df.columns:
            # Create a dummy volume column if missing
            df['volume'] = 1000.0
        
        return df
    
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
        if df is not None:
            return df
        
        # If not available, try to aggregate from smaller timeframe
        available_timeframes = DataAggregator.get_available_timeframes(symbol)
        
        # Find the smallest available timeframe
        if not available_timeframes:
            return None
        
        smallest_timeframe = min(available_timeframes, 
                               key=lambda x: list(DataAggregator.TIMEFRAME_MAPPING.keys()).index(x))
        
        # Load the smallest timeframe data
        df = DataAggregator.load_data(symbol, smallest_timeframe)
        if df is None:
            return None
        
        # Aggregate to target timeframe
        return DataAggregator.aggregate_data(df, timeframe)
