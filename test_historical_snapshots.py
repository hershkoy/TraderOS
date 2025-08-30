#!/usr/bin/env python3
"""
Test script for historical bid/ask snapshots via Polygon's upgraded API
"""

import os
import sys
from datetime import datetime, date

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.polygon_client import get_polygon_client

def test_historical_snapshot():
    """Test fetching a historical snapshot for a QQQ option"""
    
    print("Testing historical bid/ask snapshots via Polygon API...")
    
    try:
        # Get Polygon client
        client = get_polygon_client()
        print(f"✓ Polygon client initialized (rate limit: {client.requests_per_minute} req/min)")
        
        # Test with a QQQ Dec-19-2025 550 Call (example)
        ticker = "O:QQQ251219C00550000"
        test_date = "2024-07-09"  # Use a recent date for testing
        
        print(f"\nFetching snapshot for {ticker} as of {test_date}...")
        
        # Fetch historical snapshot
        snap = client.get_options_snapshot(
            option_ticker=ticker,
            as_of=test_date
        )
        
        print(f"✓ Snapshot response received")
        
        # Extract and display the data
        results = snap.get('results', {})
        if results:
            print(f"\nSnapshot Results:")
            print(f"  Ticker: {results.get('ticker', 'N/A')}")
            print(f"  Underlying: {results.get('underlying_asset', {}).get('ticker', 'N/A')}")
            
            # Quote data
            last_quote = results.get('last_quote', {})
            if last_quote:
                bid = last_quote.get('bid')
                ask = last_quote.get('ask')
                print(f"  Bid: {bid}")
                print(f"  Ask: {ask}")
                if bid is not None and ask is not None:
                    mid = (bid + ask) / 2.0
                    print(f"  Mid: {mid:.4f}")
            
            # Trade data
            last_trade = results.get('last_trade', {})
            if last_trade:
                print(f"  Last Trade: {last_trade.get('p')}")
                print(f"  Trade Size: {last_trade.get('s')}")
                print(f"  Trade Time: {last_trade.get('t')}")
            
            # Daily aggregates
            day_data = results.get('day', {})
            if day_data:
                print(f"  Volume: {day_data.get('v')}")
                print(f"  Open Interest: {day_data.get('o')}")
                print(f"  High: {day_data.get('h')}")
                print(f"  Low: {day_data.get('l')}")
                print(f"  Open: {day_data.get('o')}")
                print(f"  Close: {day_data.get('c')}")
            
            # Greeks (if available)
            greeks = results.get('greeks', {})
            if greeks:
                print(f"  Delta: {greeks.get('delta')}")
                print(f"  Gamma: {greeks.get('gamma')}")
                print(f"  Theta: {greeks.get('theta')}")
                print(f"  Vega: {greeks.get('vega')}")
                print(f"  Implied Volatility: {greeks.get('iv')}")
        else:
            print("  No results found in snapshot")
        
        # Test current snapshot for comparison
        print(f"\nFetching current snapshot for comparison...")
        current_snap = client.get_options_snapshot(ticker)
        current_results = current_snap.get('results', {})
        
        if current_results:
            current_quote = current_results.get('last_quote', {})
            current_bid = current_quote.get('bid')
            current_ask = current_quote.get('ask')
            print(f"  Current Bid: {current_bid}")
            print(f"  Current Ask: {current_ask}")
        
        print(f"\n✓ Historical snapshot test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error testing historical snapshots: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.close()

def test_rate_limit_config():
    """Test that rate limit configuration is working"""
    print("\n" + "="*50)
    print("Testing rate limit configuration...")
    
    # Test with different rate limits
    test_limits = [30, 60, 120]
    
    for limit in test_limits:
        os.environ["POLYGON_REQUESTS_PER_MINUTE"] = str(limit)
        client = get_polygon_client()
        print(f"  POLYGON_REQUESTS_PER_MINUTE={limit} → Client rate limit: {client.requests_per_minute}")
        client.close()
    
    # Reset to default
    if "POLYGON_REQUESTS_PER_MINUTE" in os.environ:
        del os.environ["POLYGON_REQUESTS_PER_MINUTE"]
    
    print("✓ Rate limit configuration test completed")

if __name__ == "__main__":
    print("Historical Bid/Ask Snapshot Test")
    print("=" * 50)
    
    # Test rate limit configuration
    test_rate_limit_config()
    
    # Test historical snapshots
    test_historical_snapshot()
    
    print("\n" + "="*50)
    print("All tests completed!")
