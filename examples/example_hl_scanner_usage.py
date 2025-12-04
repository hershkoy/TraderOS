#!/usr/bin/env python3
"""
Example usage of HL After LL Scanner
Shows different ways to use the scanner
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.scanning.hl_after_ll import scan_symbol_for_setup, scan_universe
from hl_after_ll_scanner_runner import HLAfterLLScanner

def example_1_single_symbol():
    """Example 1: Scan a single symbol"""
    print("=" * 60)
    print("Example 1: Scanning a single symbol")
    print("=" * 60)
    
    # Create sample data with LL → HH → HL pattern
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W-FRI')
    n = len(dates)
    
    # Create pattern: downtrend → LL → HH → HL
    prices = []
    
    # Downtrend phase
    for i in range(n//3):
        prices.append(100 - i * 0.5)
    
    # LL (Lower Low)
    ll_price = prices[-1] - 2
    prices.append(ll_price)
    
    # HH (Higher High)
    hh_price = ll_price + 8
    prices.append(hh_price)
    
    # HL (Higher Low)
    hl_price = ll_price + 3
    prices.append(hl_price)
    
    # Continue uptrend
    for i in range(n - len(prices)):
        prices.append(hl_price + i * 0.3)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
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
    
    # Scan for pattern
    match = scan_symbol_for_setup("EXAMPLE", df)
    
    if match:
        print(f"✅ Pattern found in {match.symbol}!")
        print(f"   LL: {match.ll_date.strftime('%Y-%m-%d')} at ${match.ll_price:.2f}")
        print(f"   HH: {match.hh_dates[0].strftime('%Y-%m-%d')} at ${match.hh_prices[0]:.2f}")
        print(f"   HL: {match.hl_date.strftime('%Y-%m-%d')} at ${match.hl_price:.2f}")
        print(f"   Current: ${match.last_price:.2f}")
        
        if match.last_price >= match.hl_price:
            print("   ✅ Monday Check: Pattern still valid")
        else:
            print("   ❌ Monday Check: Pattern broken")
    else:
        print("❌ No pattern found")

def example_2_multiple_symbols():
    """Example 2: Scan multiple symbols"""
    print("\n" + "=" * 60)
    print("Example 2: Scanning multiple symbols")
    print("=" * 60)
    
    # Create sample data for multiple symbols
    symbols_data = {}
    
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W-FRI')
        n = len(dates)
        
        # Create different patterns for each symbol
        if symbol == "AAPL":
            # Clear LL → HH → HL pattern
            prices = [100 - i * 0.3 for i in range(n//2)]  # Downtrend
            prices.extend([prices[-1] - 2, prices[-1] + 8, prices[-1] + 3])  # LL → HH → HL
            prices.extend([prices[-1] + i * 0.2 for i in range(n - len(prices))])  # Uptrend
        elif symbol == "MSFT":
            # Extended LL → HH → HH → HL pattern
            prices = [200 - i * 0.4 for i in range(n//3)]  # Downtrend
            prices.extend([prices[-1] - 3, prices[-1] + 6, prices[-1] + 4, prices[-1] + 2])  # LL → HH → HH → HL
            prices.extend([prices[-1] + i * 0.15 for i in range(n - len(prices))])  # Uptrend
        else:  # GOOGL
            # No clear pattern (should not match)
            prices = [150 + i * 0.1 for i in range(n)]  # Simple uptrend
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * 1.01
            low = close * 0.99
            open_price = prices[i-1] if i > 0 else close
            volume = 2000000 + i * 2000
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        symbols_data[symbol] = pd.DataFrame(data, index=dates)
    
    # Scan all symbols
    matches = scan_universe(symbols_data)
    
    print(f"Scanned {len(symbols_data)} symbols, found {len(matches)} patterns:")
    
    for match in matches:
        print(f"\n✅ {match.symbol}:")
        print(f"   LL: {match.ll_date.strftime('%Y-%m-%d')} at ${match.ll_price:.2f}")
        for i, (hh_date, hh_price) in enumerate(zip(match.hh_dates, match.hh_prices)):
            print(f"   HH{i+1}: {hh_date.strftime('%Y-%m-%d')} at ${hh_price:.2f}")
        print(f"   HL: {match.hl_date.strftime('%Y-%m-%d')} at ${match.hl_price:.2f}")
        print(f"   Current: ${match.last_price:.2f}")

def example_3_configuration():
    """Example 3: Using configuration file"""
    print("\n" + "=" * 60)
    print("Example 3: Using configuration file")
    print("=" * 60)
    
    # Create a custom configuration
    config = {
        'scanner': {
            'left_bars': 1,
            'right_bars': 2,
            'patterns': ['LL→HH→HL', 'LL→HH→HH→HL'],
            'monday_check': {'enabled': True},
            'data': {'timeframe': '1d', 'min_weeks': 52}
        },
        'output': {
            'console': {'show_summary': True, 'show_details': True},
            'files': {'save_csv': False, 'save_log': False}
        },
        'universe': {'source': 'manual', 'symbols': ['AAPL', 'MSFT']},
        'logging': {'level': 'INFO'}
    }
    
    # Save config to file
    import yaml
    with open('example_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("Created example_config.yaml")
    print("You can now run: python hl_after_ll_scanner_runner.py --config example_config.yaml")

def example_4_monday_check():
    """Example 4: Demonstrating Monday check logic"""
    print("\n" + "=" * 60)
    print("Example 4: Monday check logic")
    print("=" * 60)
    
    # Create data with pattern but current price below HL
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W-FRI')
    n = len(dates)
    
    # Create LL → HH → HL pattern
    prices = [100 - i * 0.5 for i in range(n//2)]  # Downtrend
    prices.extend([prices[-1] - 2, prices[-1] + 8, prices[-1] + 3])  # LL → HH → HL
    prices.extend([prices[-1] + i * 0.1 for i in range(n - len(prices))])  # Continue
    
    # But make current price below HL (pattern broken)
    prices[-1] = prices[-3] - 1  # Current price below HL
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
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
    
    # Scan for pattern
    match = scan_symbol_for_setup("BROKEN_PATTERN", df)
    
    if match:
        print(f"Pattern found but Monday check failed:")
        print(f"   HL: ${match.hl_price:.2f}")
        print(f"   Current: ${match.last_price:.2f}")
        print(f"   ❌ Monday Check: Current price below HL - Pattern broken!")
    else:
        print("No pattern found (likely due to Monday check failure)")

def main():
    """Run all examples"""
    print("HL After LL Scanner - Usage Examples")
    print("=" * 60)
    
    try:
        example_1_single_symbol()
        example_2_multiple_symbols()
        example_3_configuration()
        example_4_monday_check()
        
        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)
        print("\nTo run the actual scanner:")
        print("1. python test_hl_after_ll_scanner.py          # Test with sample data")
        print("2. python hl_after_ll_scanner_runner.py         # Run on your universe")
        print("3. python hl_after_ll_scanner_runner.py --test  # Test mode")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
