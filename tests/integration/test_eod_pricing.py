#!/usr/bin/env python3
"""
Test script for EOD pricing functionality via Polygon API
"""

import os
import sys
from datetime import datetime, date

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.api.polygon_client import get_polygon_client

def test_eod_pricing():
    """Test fetching EOD pricing data for QQQ options"""
    
    print("Testing EOD pricing via Polygon API...")
    
    try:
        # Get Polygon client
        client = get_polygon_client()
        print(f"✓ Polygon client initialized (rate limit: {client.requests_per_minute} req/min)")
        
        # Test with a QQQ Aug 1, 2025 500 Call
        ticker = "O:QQQ250801C00500000"
        test_date = "2024-07-09"  # Use a recent date for testing
        
        print(f"\nFetching EOD data for {ticker} on {test_date}...")
        
        # Fetch EOD data
        eod_data = client.get_option_eod(
            option_ticker=ticker,
            date=test_date
        )
        
        print(f"✓ EOD response received")
        print(f"Response structure: {type(eod_data)}")
        
        # Extract and display the data
        if isinstance(eod_data, dict):
            print(f"\nEOD Data Results:")
            print(f"  Status: {eod_data.get('status', 'N/A')}")
            print(f"  Symbol: {eod_data.get('symbol', 'N/A')}")
            print(f"  Date: {eod_data.get('date', 'N/A')}")
            
            # Pricing data
            open_price = eod_data.get('open')
            high_price = eod_data.get('high')
            low_price = eod_data.get('low')
            close_price = eod_data.get('close')
            volume = eod_data.get('volume')
            vwap = eod_data.get('vwap')
            transactions = eod_data.get('transactions')
            
            print(f"  Open: {open_price}")
            print(f"  High: {high_price}")
            print(f"  Low: {low_price}")
            print(f"  Close: {close_price}")
            print(f"  Volume: {volume}")
            print(f"  VWAP: {vwap}")
            print(f"  Transactions: {transactions}")
            
            # Check if we have valid pricing data
            if any(price is not None for price in [open_price, high_price, low_price, close_price]):
                print(f"\n✅ Valid pricing data found!")
                
                # Calculate some derived metrics
                if all(price is not None for price in [open_price, close_price]):
                    daily_return = ((close_price - open_price) / open_price) * 100
                    print(f"  Daily Return: {daily_return:.2f}%")
                
                if all(price is not None for price in [high_price, low_price]):
                    daily_range = high_price - low_price
                    print(f"  Daily Range: {daily_range:.4f}")
                
                if close_price is not None and vwap is not None:
                    vwap_deviation = ((close_price - vwap) / vwap) * 100
                    print(f"  VWAP Deviation: {vwap_deviation:.2f}%")
                    
            else:
                print(f"\n⚠️  No pricing data found")
                
        else:
            print(f"  Unexpected response type: {type(eod_data)}")
        
        # Test minute aggregates as well
        print(f"\n" + "="*50)
        print("Testing minute aggregates...")
        
        try:
            aggs_data = client.get_option_minute_aggs(
                option_ticker=ticker,
                date=test_date
            )
            
            if isinstance(aggs_data, dict):
                results = aggs_data.get('results', [])
                print(f"✓ Minute aggregates response received")
                print(f"  Number of minute bars: {len(results)}")
                
                if results:
                    # Show first few bars
                    print(f"  First few minute bars:")
                    for i, bar in enumerate(results[:3]):
                        print(f"    Bar {i+1}: O={bar.get('o')}, H={bar.get('h')}, L={bar.get('l')}, C={bar.get('c')}, V={bar.get('v')}")
                    
                    # Calculate some intraday metrics
                    if len(results) > 1:
                        first_bar = results[0]
                        last_bar = results[-1]
                        if first_bar.get('o') and last_bar.get('c'):
                            intraday_return = ((last_bar['c'] - first_bar['o']) / first_bar['o']) * 100
                            print(f"  Intraday Return: {intraday_return:.2f}%")
                else:
                    print(f"  ⚠️  No minute bars found")
            else:
                print(f"  ❌ Unexpected aggregates response type: {type(aggs_data)}")
                
        except Exception as e:
            print(f"  ❌ Error testing minute aggregates: {e}")
        
        print(f"\n✓ EOD pricing test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error testing EOD pricing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.close()

def test_multiple_options():
    """Test EOD pricing for multiple option contracts"""
    
    print("\n" + "="*50)
    print("Testing EOD pricing for multiple options...")
    
    try:
        client = get_polygon_client()
        
        # Test different QQQ options
        test_options = [
            "O:QQQ250801C00500000",  # QQQ Aug 1, 2025 500 Call
            "O:QQQ250801P00500000",  # QQQ Aug 1, 2025 500 Put
            "O:QQQ250801C00550000",  # QQQ Aug 1, 2025 550 Call
            "O:QQQ250801P00550000",  # QQQ Aug 1, 2025 550 Put
        ]
        
        test_date = "2024-07-09"
        
        successful = 0
        total = len(test_options)
        
        for ticker in test_options:
            try:
                print(f"\nTesting {ticker}...")
                
                eod_data = client.get_option_eod(
                    option_ticker=ticker,
                    date=test_date
                )
                
                if isinstance(eod_data, dict) and eod_data.get('status') == 'OK':
                    close_price = eod_data.get('close')
                    volume = eod_data.get('volume')
                    
                    if close_price is not None:
                        print(f"  ✅ Close: {close_price}, Volume: {volume}")
                        successful += 1
                    else:
                        print(f"  ⚠️  No close price")
                else:
                    print(f"  ❌ No data or error status")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        print(f"\n📊 Summary: {successful}/{total} options had valid EOD data")
        
        client.close()
        
    except Exception as e:
        print(f"✗ Error testing multiple options: {e}")

if __name__ == "__main__":
    print("EOD Pricing Test")
    print("=" * 50)
    
    # Test single option EOD pricing
    test_eod_pricing()
    
    # Test multiple options
    test_multiple_options()
    
    print("\n" + "="*50)
    print("All tests completed!")


