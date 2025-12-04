#!/usr/bin/env python3
"""
Test script for HL After LL Scanner
Tests the scanner with a single symbol to verify functionality
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.scanners.hl_after_ll_scanner import scan_symbol_for_setup

def create_sample_data():
    """Create sample weekly data for testing"""
    # Create sample weekly data with a clear LL → HH → HL pattern
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W-FRI')
    
    # Create a pattern: downtrend → LL → HH → HL
    n = len(dates)
    prices = []
    
    # Start with downtrend
    for i in range(n//3):
        prices.append(100 - i * 0.5)  # Downtrend
    
    # Create LL (Lower Low)
    ll_price = prices[-1] - 2
    prices.append(ll_price)
    
    # Create HH (Higher High) - break above previous high
    hh_price = ll_price + 8
    prices.append(hh_price)
    
    # Create HL (Higher Low) - pullback but higher than LL
    hl_price = ll_price + 3
    prices.append(hl_price)
    
    # Continue with uptrend
    for i in range(n - len(prices)):
        prices.append(hl_price + i * 0.3)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Simple OHLCV generation
        high = close * 1.02
        low = close * 0.98
        open_price = prices[i-1] if i > 0 else close
        volume = 1000000 + i * 1000
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_scanner():
    """Test the HL after LL scanner with sample data"""
    print("🧪 Testing HL After LL Scanner")
    print("=" * 40)
    
    # Create sample data
    print("Creating sample weekly data with LL → HH → HL pattern...")
    df = create_sample_data()
    print(f"Created {len(df)} weekly bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Test the scanner
    print("\nScanning for LL → HH → HL pattern...")
    match = scan_symbol_for_setup("TEST", df)
    
    if match:
        print("\n✅ Pattern found!")
        print(f"Symbol: {match.symbol}")
        print(f"LL: {match.ll_date.strftime('%Y-%m-%d')} at ${match.ll_price:.2f}")
        
        for i, (hh_date, hh_price) in enumerate(zip(match.hh_dates, match.hh_prices)):
            print(f"HH{i+1}: {hh_date.strftime('%Y-%m-%d')} at ${hh_price:.2f}")
        
        print(f"HL: {match.hl_date.strftime('%Y-%m-%d')} at ${match.hl_price:.2f}")
        print(f"Current Price: ${match.last_price:.2f}")
        
        # Monday check
        if match.last_price >= match.hl_price:
            print("✅ Monday Check: Current price >= HL price")
        else:
            print("❌ Monday Check: Current price < HL price")
    else:
        print("\n❌ No LL → HH → HL pattern found")
    
    print(f"\n📊 Sample data preview:")
    print(df.tail(10))

if __name__ == "__main__":
    test_scanner()
