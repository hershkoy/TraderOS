#!/usr/bin/env python3
"""
Filter out OTC stocks that Alpaca doesn't support
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data.ticker_universe import TickerUniverseManager

def filter_otc_stocks():
    """Filter out OTC stocks from the universe"""
    try:
        # Get all tickers
        manager = TickerUniverseManager()
        all_tickers = manager._get_cached_tickers('custom_universe')
        
        if not all_tickers:
            print("No tickers found in universe")
            return
        
        print(f"Original ticker count: {len(all_tickers)}")
        
        # Filter out OTC stocks (common OTC suffixes)
        otc_suffixes = ['F', 'PK', 'OB', 'OTCMKTS']
        filtered_tickers = []
        otc_tickers = []
        
        for ticker in all_tickers:
            # Check if ticker ends with OTC suffixes
            is_otc = any(ticker.endswith(suffix) for suffix in otc_suffixes)
            
            if is_otc:
                otc_tickers.append(ticker)
            else:
                filtered_tickers.append(ticker)
        
        print(f"OTC tickers found: {len(otc_tickers)}")
        print(f"Filtered tickers: {len(filtered_tickers)}")
        
        if otc_tickers:
            print(f"\nFirst 10 OTC tickers that will be skipped:")
            for ticker in otc_tickers[:10]:
                print(f"  - {ticker}")
        
        # Cache the filtered tickers
        manager._cache_tickers('filtered_universe', [
            {'symbol': ticker, 'company_name': f'Filtered Ticker {ticker}', 'sector': 'Filtered'}
            for ticker in filtered_tickers
        ])
        
        print(f"\nFiltered universe saved as 'filtered_universe'")
        print(f"You can now run scanners on the filtered universe:")
        print(f"python scanner_runner.py --scanner hl_after_ll squeeze --skip-update")
        
    except Exception as e:
        print(f"Error filtering tickers: {e}")

if __name__ == "__main__":
    filter_otc_stocks()
