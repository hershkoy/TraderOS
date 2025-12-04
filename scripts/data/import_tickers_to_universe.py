#!/usr/bin/env python3
"""
Import tickers from file into TimescaleDB universe
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data.ticker_universe import TickerUniverseManager

def load_tickers_from_file(file_path: str) -> List[str]:
    """Load tickers from a text file (one ticker per line)"""
    tickers = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                ticker = line.strip()
                if ticker and not ticker.startswith('#'):  # Skip empty lines and comments
                    tickers.append(ticker)
        
        print(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def import_tickers_to_universe(tickers: List[str], index_name: str = "custom") -> bool:
    """Import tickers into the TimescaleDB universe"""
    try:
        print(f"Importing {len(tickers)} tickers to universe as '{index_name}'...")
        
        # Initialize the ticker universe manager
        manager = TickerUniverseManager()
        
        # Prepare ticker data
        ticker_data = []
        for ticker in tickers:
            ticker_data.append({
                'symbol': ticker,
                'company_name': f'Custom Ticker {ticker}',
                'sector': 'Custom'
            })
        
        # Cache the tickers
        manager._cache_tickers(index_name, ticker_data)
        
        print(f"Successfully imported {len(tickers)} tickers to universe")
        return True
        
    except Exception as e:
        print(f"Error importing tickers: {e}")
        return False

def get_universe_stats():
    """Get statistics about the ticker universe"""
    try:
        manager = TickerUniverseManager()
        stats = manager.get_universe_stats()
        
        print("\nUniverse Statistics:")
        print("=" * 40)
        print(f"Total unique symbols: {stats.get('total_unique_symbols', 0)}")
        
        print("\nBy index:")
        for index, count in stats.get('index_counts', {}).items():
            print(f"  {index}: {count} symbols")
        
        print("\nCache status:")
        for index, cache_info in stats.get('cache_status', {}).items():
            last_fetched = cache_info['last_fetched'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {index}: {cache_info['ticker_count']} symbols, last updated {last_fetched}")
        
    except Exception as e:
        print(f"Error getting universe stats: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import tickers into TimescaleDB universe")
    parser.add_argument('--ticker-file', type=str, required=True,
                       help='Path to ticker file (one ticker per line)')
    parser.add_argument('--index-name', type=str, default='custom',
                       help='Name for the ticker index in database')
    parser.add_argument('--stats', action='store_true',
                       help='Show universe statistics after import')
    
    args = parser.parse_args()
    
    print("Ticker Universe Importer")
    print("=" * 30)
    
    # Step 1: Load tickers from file
    print(f"\nStep 1: Loading tickers from {args.ticker_file}")
    tickers = load_tickers_from_file(args.ticker_file)
    if not tickers:
        print("No tickers loaded. Exiting.")
        return
    
    # Step 2: Import tickers to universe
    print(f"\nStep 2: Importing tickers to universe '{args.index_name}'")
    success = import_tickers_to_universe(tickers, args.index_name)
    if not success:
        print("Failed to import tickers. Exiting.")
        return
    
    # Step 3: Show stats if requested
    if args.stats:
        get_universe_stats()
    
    print(f"\nImport completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nYou can now run scanners on these tickers using:")
    print(f"python scanner_runner.py --scanner hl_after_ll squeeze --symbols {' '.join(tickers[:10])}...")
    print(f"(or use the full universe with your existing scanner runner)")

if __name__ == "__main__":
    main()
