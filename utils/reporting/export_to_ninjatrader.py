#!/usr/bin/env python3
"""
Export market data from TimescaleDB to NinjaTrader 8 CSV format

NinjaTrader 8 CSV Format Requirements:
- File name: Symbol.Last.txt (e.g., AAPL.Last.txt)
- Format: yyyyMMdd HHmmss,open,high,low,close,volume
- Comma-separated values
- No header row
- Timestamps in YYYYMMDD HHMMSS format
- Duplicate timestamps are automatically removed (first occurrence kept)
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import logging
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.db.timescaledb_client import get_timescaledb_client
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """
    Parse timeframe string to minutes
    
    Args:
        timeframe: Timeframe string (e.g., '1d', '1h', '15m', '4h')
    
    Returns:
        Number of minutes
    """
    # Match patterns like: 1d, 1h, 15m, 4h, 30m, etc.
    match = re.match(r'^(\d+)([dhm])$', timeframe.lower())
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'd':
        return value * 24 * 60  # days to minutes
    elif unit == 'h':
        return value * 60  # hours to minutes
    elif unit == 'm':
        return value  # minutes
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")


def find_best_lower_timeframe(requested_timeframe: str, available_timeframes: List[str]) -> Optional[str]:
    """
    Find the best lower timeframe to aggregate from
    
    Args:
        requested_timeframe: The requested timeframe (e.g., '1d')
        available_timeframes: List of available timeframes
    
    Returns:
        Best lower timeframe to use, or None if none found
    """
    try:
        requested_minutes = parse_timeframe_to_minutes(requested_timeframe)
    except ValueError:
        return None
    
    # Parse all available timeframes to minutes
    available_with_minutes = []
    for tf in available_timeframes:
        try:
            minutes = parse_timeframe_to_minutes(tf)
            available_with_minutes.append((tf, minutes))
        except ValueError:
            continue
    
    # Filter to only timeframes smaller than requested
    lower_timeframes = [(tf, mins) for tf, mins in available_with_minutes if mins < requested_minutes]
    
    if not lower_timeframes:
        return None
    
    # Sort by minutes descending (largest lower timeframe first)
    lower_timeframes.sort(key=lambda x: x[1], reverse=True)
    
    # Return the largest lower timeframe (closest to requested)
    return lower_timeframes[0][0]


def aggregate_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Aggregate OHLCV data to a higher timeframe
    
    Args:
        df: DataFrame with columns: ts (or timestamp), open, high, low, close, volume
        target_timeframe: Target timeframe to aggregate to (e.g., '1d', '1h')
    
    Returns:
        Aggregated DataFrame
    """
    # Ensure we have a timestamp column
    if 'ts' in df.columns:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['ts'], utc=True)
    elif 'timestamp' in df.columns:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    else:
        raise ValueError("No timestamp column found in data")
    
    # Ensure timestamp is timezone-aware and in UTC
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    
    # Parse target timeframe to pandas frequency
    target_minutes = parse_timeframe_to_minutes(target_timeframe)
    
    # Determine pandas frequency string
    if target_minutes >= 1440:  # 1 day or more
        days = target_minutes // 1440
        freq = f'{days}D'
    elif target_minutes >= 60:  # 1 hour or more
        hours = target_minutes // 60
        freq = f'{hours}H'
    else:  # minutes
        freq = f'{target_minutes}min'
    
    # Set timestamp as index for resampling
    df_indexed = df.set_index('timestamp')
    
    # Ensure numeric types
    df_indexed['open'] = pd.to_numeric(df_indexed['open'], errors='coerce')
    df_indexed['high'] = pd.to_numeric(df_indexed['high'], errors='coerce')
    df_indexed['low'] = pd.to_numeric(df_indexed['low'], errors='coerce')
    df_indexed['close'] = pd.to_numeric(df_indexed['close'], errors='coerce')
    df_indexed['volume'] = pd.to_numeric(df_indexed['volume'], errors='coerce')
    
    # Resample and aggregate OHLCV
    # Open: first value in period
    # High: maximum value in period
    # Low: minimum value in period
    # Close: last value in period
    # Volume: sum of volumes in period
    aggregated = pd.DataFrame()
    aggregated['open'] = df_indexed['open'].resample(freq).first()
    aggregated['high'] = df_indexed['high'].resample(freq).max()
    aggregated['low'] = df_indexed['low'].resample(freq).min()
    aggregated['close'] = df_indexed['close'].resample(freq).last()
    aggregated['volume'] = df_indexed['volume'].resample(freq).sum()
    
    # Remove any rows with NaN values (incomplete periods)
    aggregated = aggregated.dropna()
    
    # Reset index to get timestamp back as column
    aggregated = aggregated.reset_index()
    aggregated['ts'] = aggregated['timestamp']
    
    return aggregated


def export_to_ninjatrader(
    symbol: str,
    timeframe: str,
    output_dir: Optional[str] = None,
    provider: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: Optional[int] = None
) -> str:
    """
    Export market data from TimescaleDB to NinjaTrader 8 CSV format
    
    Args:
        symbol: Stock symbol to export
        timeframe: Time interval (1h, 1d, etc.)
        output_dir: Output directory for the CSV file (default: current directory)
        provider: Data provider filter (optional)
        start_time: Start timestamp filter (optional)
        end_time: End timestamp filter (optional)
        limit: Maximum number of records to export (optional)
    
    Returns:
        Path to the exported CSV file
    """
    # Get database client
    client = get_timescaledb_client()
    
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")
    
    try:
        # Retrieve data from database
        logger.info(f"Retrieving data for {symbol} {timeframe}...")
        df = client.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            provider=provider,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # If no data found, try to aggregate from lower timeframe
        if df is None or df.empty:
            logger.info(f"No data found for {symbol} {timeframe}, checking for lower timeframes to aggregate...")
            available_timeframes = client.get_available_timeframes(symbol, provider)
            
            if not available_timeframes:
                raise ValueError(f"No data found for {symbol} {timeframe} and no available timeframes to aggregate from")
            
            # Find best lower timeframe to aggregate from
            lower_timeframe = find_best_lower_timeframe(timeframe, available_timeframes)
            
            if not lower_timeframe:
                raise ValueError(
                    f"No data found for {symbol} {timeframe}. "
                    f"Available timeframes: {', '.join(available_timeframes)}, "
                    f"but none are suitable for aggregation to {timeframe}"
                )
            
            logger.info(f"Aggregating from {lower_timeframe} to {timeframe}...")
            
            # Get data from lower timeframe
            df = client.get_market_data(
                symbol=symbol,
                timeframe=lower_timeframe,
                provider=provider,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data found for {symbol} {lower_timeframe} to aggregate")
            
            logger.info(f"Retrieved {len(df)} records from {lower_timeframe}")
            
            # Aggregate to requested timeframe
            df = aggregate_ohlcv(df, timeframe)
            logger.info(f"Aggregated to {len(df)} records for {timeframe}")
        else:
            logger.info(f"Retrieved {len(df)} records")
        
        # Convert timestamp to NinjaTrader format (YYYYMMDD HHMMSS)
        # Ensure timestamps are timezone-aware, then convert to UTC and remove timezone
        if 'ts' in df.columns:
            timestamps = pd.to_datetime(df['ts'], utc=True)
        elif 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'], utc=True)
        else:
            raise ValueError("No timestamp column found in data")
        
        # Convert to UTC and remove timezone info for formatting
        # Handle both timezone-aware and naive timestamps
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            # If naive, assume UTC
            timestamps = timestamps
        
        # Add timestamps to dataframe for deduplication
        df['timestamp_parsed'] = timestamps
        
        # Remove duplicate rows based on timestamp (keep first occurrence)
        original_count = len(df)
        df = df.drop_duplicates(subset=['timestamp_parsed'], keep='first')
        df = df.sort_values('timestamp_parsed').reset_index(drop=True)
        
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate rows based on timestamp")
        
        logger.info(f"Exporting {len(df)} unique records")
        
        # Update timestamps after deduplication
        timestamps = df['timestamp_parsed']
        
        # Format timestamps as YYYYMMDD HHMMSS
        formatted_timestamps = timestamps.dt.strftime('%Y%m%d %H%M%S')
        
        # Prepare data columns - ensure numeric types (use reindexed dataframe)
        open_prices = pd.to_numeric(df['open'], errors='coerce')
        high_prices = pd.to_numeric(df['high'], errors='coerce')
        low_prices = pd.to_numeric(df['low'], errors='coerce')
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        volumes = pd.to_numeric(df['volume'], errors='coerce')
        
        # Create output DataFrame with comma-separated format
        output_data = []
        for i in range(len(df)):
            output_data.append(
                f"{formatted_timestamps.iloc[i]},"
                f"{open_prices.iloc[i]:.6f},"
                f"{high_prices.iloc[i]:.6f},"
                f"{low_prices.iloc[i]:.6f},"
                f"{close_prices.iloc[i]:.6f},"
                f"{int(volumes.iloc[i])}"
            )
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename: Symbol.Last.txt
        output_filename = f"{symbol.upper()}.Last.txt"
        output_path = output_dir / output_filename
        
        # Write to file (no header row)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_data))
        
        logger.info(f"Exported {len(output_data)} records to {output_path}")
        logger.info(f"Date range: {timestamps.min()} to {timestamps.max()}")
        
        return str(output_path)
        
    finally:
        client.disconnect()


def main():
    """Command-line interface for the export script"""
    parser = argparse.ArgumentParser(
        description='Export market data from TimescaleDB to NinjaTrader 8 CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all data for AAPL daily bars
  python export_to_ninjatrader.py AAPL 1d
  
  # Export to specific directory
  python export_to_ninjatrader.py AAPL 1d --output-dir ./exports
  
  # Export with date range
  python export_to_ninjatrader.py AAPL 1d --start-time "2024-01-01" --end-time "2024-12-31"
  
  # Export with provider filter
  python export_to_ninjatrader.py AAPL 1d --provider ALPACA
  
  # Export limited number of records
  python export_to_ninjatrader.py AAPL 1d --limit 1000
        """
    )
    
    parser.add_argument(
        'symbol',
        type=str,
        help='Stock symbol to export (e.g., AAPL)'
    )
    
    parser.add_argument(
        'timeframe',
        type=str,
        help='Time interval (e.g., 1h, 1d, 4h)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for the CSV file (default: current directory)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='Data provider filter (optional)'
    )
    
    parser.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='Start time filter in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS (optional)'
    )
    
    parser.add_argument(
        '--end-time',
        type=str,
        default=None,
        help='End time filter in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS (optional)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of records to export (optional)'
    )
    
    args = parser.parse_args()
    
    # Parse datetime strings if provided
    start_time = None
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                start_time = datetime.strptime(args.start_time, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid start-time format: {args.start_time}")
                sys.exit(1)
    
    end_time = None
    if args.end_time:
        try:
            end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                end_time = datetime.strptime(args.end_time, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid end-time format: {args.end_time}")
                sys.exit(1)
    
    try:
        output_path = export_to_ninjatrader(
            symbol=args.symbol,
            timeframe=args.timeframe,
            output_dir=args.output_dir,
            provider=args.provider,
            start_time=start_time,
            end_time=end_time,
            limit=args.limit
        )
        print(f"Successfully exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

