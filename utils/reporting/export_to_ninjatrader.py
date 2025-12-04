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
from datetime import datetime
from typing import Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ..db.timescaledb_client import get_timescaledb_client
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        
        if df is None or df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")
        
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

