#!/usr/bin/env python3
"""
Example usage of the Ticker Universe Management System

This script demonstrates how to:
1. Initialize the ticker universe manager
2. Fetch and cache tickers from various indices
3. Get combined universe
4. Query ticker information
5. Get universe statistics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data.ticker_universe import TickerUniverseManager
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main example function"""
    print("🚀 Ticker Universe Management Example")
    print("=" * 50)
    
    try:
        # Initialize the ticker universe manager
        print("\n1. Initializing Ticker Universe Manager...")
        manager = TickerUniverseManager()
        print("✅ Manager initialized successfully")
        
        # Get S&P 500 tickers (will fetch from Wikipedia if not cached)
        print("\n2. Fetching S&P 500 tickers...")
        sp500_tickers = manager.get_sp500_tickers()
        print(f"✅ Retrieved {len(sp500_tickers)} S&P 500 tickers")
        print(f"   First 10: {sp500_tickers[:10]}")
        
        # Get NASDAQ-100 tickers
        print("\n3. Fetching NASDAQ-100 tickers...")
        nasdaq100_tickers = manager.get_nasdaq100_tickers()
        print(f"✅ Retrieved {len(nasdaq100_tickers)} NASDAQ-100 tickers")
        print(f"   First 10: {nasdaq100_tickers[:10]}")
        
        # Get combined universe
        print("\n4. Getting combined universe...")
        combined_universe = manager.get_combined_universe()
        print(f"✅ Combined universe contains {len(combined_universe)} unique symbols")
        print(f"   First 10: {combined_universe[:10]}")
        
        # Get detailed information about a specific ticker
        print("\n5. Getting ticker information...")
        sample_ticker = combined_universe[0] if combined_universe else "AAPL"
        ticker_info = manager.get_ticker_info(sample_ticker)
        if ticker_info:
            print(f"✅ Information for {sample_ticker}:")
            for key, value in ticker_info.items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ No information found for {sample_ticker}")
        
        # Get universe statistics
        print("\n6. Getting universe statistics...")
        stats = manager.get_universe_stats()
        print("✅ Universe Statistics:")
        print(f"   Total unique symbols: {stats.get('total_unique_symbols', 0)}")
        print("   Index counts:")
        for index, count in stats.get('index_counts', {}).items():
            print(f"     {index}: {count} tickers")
        print("   Cache status:")
        for index, cache_info in stats.get('cache_status', {}).items():
            print(f"     {index}: {cache_info['ticker_count']} tickers, last fetched: {cache_info['last_fetched']}")
        
        # Demonstrate cache behavior
        print("\n7. Demonstrating cache behavior...")
        print("   Getting S&P 500 tickers again (should use cache)...")
        cached_sp500 = manager.get_sp500_tickers()
        print(f"   ✅ Retrieved {len(cached_sp500)} tickers from cache")
        
        # Force refresh
        print("\n8. Force refreshing all indices...")
        refresh_results = manager.refresh_all_indices()
        print(f"✅ Refresh complete: {refresh_results}")
        
        # Get cached universe (no fetching)
        print("\n9. Getting cached universe only...")
        cached_universe = manager.get_cached_combined_universe()
        print(f"✅ Cached universe contains {len(cached_universe)} symbols")
        
        print("\n🎉 Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        logging.error(f"Example failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
