#!/usr/bin/env python3
"""
Update Data and Scan Script
Fetches latest data from providers before running scanners
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ticker_universe import get_ticker_universe
from scanner_runner import ScannerRunner

def fetch_latest_data(symbols, provider="ALPACA", timeframe="1d", days_back=30):
    """
    Fetch latest data for symbols from provider
    
    Args:
        symbols: List of symbols to fetch data for
        provider: Data provider (ALPACA, IB)
        timeframe: Data timeframe (1d, 1h)
        days_back: How many days back to fetch (to ensure we have recent data)
    """
    print(f"📡 Fetching latest data for {len(symbols)} symbols from {provider}")
    print(f"   Timeframe: {timeframe}, Days back: {days_back}")
    
    successful_fetches = 0
    failed_fetches = 0
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"   [{i}/{len(symbols)}] Fetching {symbol}...")
            
            # Calculate bars needed (approximate)
            if timeframe == "1d":
                bars_needed = days_back + 10  # Extra buffer
            else:  # 1h
                bars_needed = days_back * 7 + 50  # Extra buffer for hourly
            
            # Run fetch_data.py for this symbol
            cmd = [
                "python", "utils/fetch_data.py",
                "--symbol", symbol,
                "--provider", provider,
                "--timeframe", timeframe,
                "--bars", str(bars_needed)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                successful_fetches += 1
                print(f"   ✅ {symbol} - Data fetched successfully")
            else:
                failed_fetches += 1
                print(f"   ❌ {symbol} - Failed: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            failed_fetches += 1
            print(f"   ⏰ {symbol} - Timeout")
        except Exception as e:
            failed_fetches += 1
            print(f"   ❌ {symbol} - Error: {e}")
    
    print(f"\n📊 Data Fetch Summary:")
    print(f"   ✅ Successful: {successful_fetches}")
    print(f"   ❌ Failed: {failed_fetches}")
    print(f"   📈 Success Rate: {successful_fetches/(successful_fetches+failed_fetches)*100:.1f}%")
    
    return successful_fetches > 0

def update_and_scan(scanner_type="hl_after_ll", symbols=None, provider="ALPACA", 
                    timeframe="1d", days_back=30, config_file="scanner_config.yaml"):
    """
    Update data and run scanner
    
    Args:
        scanner_type: Type of scanner to run
        symbols: List of symbols to update and scan (None = use universe)
        provider: Data provider to fetch from
        timeframe: Data timeframe
        days_back: Days back to fetch
        config_file: Scanner configuration file
    """
    print("🔄 Update and Scan Workflow")
    print("=" * 50)
    
    # Get symbols to process
    if symbols is None:
        print("Loading ticker universe...")
        universe = get_ticker_universe()
        symbols = list(universe.keys())
        print(f"Loaded {len(symbols)} symbols from universe")
    else:
        print(f"Using provided {len(symbols)} symbols")
    
    # Limit symbols for testing (optional)
    if len(symbols) > 50:
        print(f"⚠️  Large universe ({len(symbols)} symbols). Consider using --symbols for specific symbols.")
        response = input("Continue with all symbols? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Step 1: Fetch latest data
    print(f"\n📡 Step 1: Fetching latest data from {provider}")
    print("-" * 50)
    
    data_success = fetch_latest_data(symbols, provider, timeframe, days_back)
    
    if not data_success:
        print("❌ Data fetching failed. Aborting scan.")
        return
    
    # Step 2: Run scanner
    print(f"\n🔍 Step 2: Running {scanner_type.upper()} scanner")
    print("-" * 50)
    
    try:
        # Initialize scanner runner
        scanner_runner = ScannerRunner(config_file)
        
        # Override symbols if provided
        if symbols:
            scanner_runner.config['universe'] = {'source': 'manual', 'symbols': symbols}
        
        # Run the scanner
        scanner_runner.run(scanner_type, symbols)
        
    except Exception as e:
        print(f"❌ Scanner failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Update Data and Run Scanner")
    parser.add_argument('--scanner', type=str, default='hl_after_ll',
                       choices=['hl_after_ll', 'vcp', 'liquidity_sweep'],
                       help='Scanner type to run')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to update and scan')
    parser.add_argument('--provider', type=str, choices=['ALPACA', 'IB'], default='ALPACA',
                       help='Data provider to fetch from')
    parser.add_argument('--timeframe', type=str, choices=['1d', '1h'], default='1d',
                       help='Data timeframe')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days back to fetch data')
    parser.add_argument('--config', type=str, default='scanner_config.yaml',
                       help='Scanner configuration file')
    parser.add_argument('--skip-update', action='store_true',
                       help='Skip data update, just run scanner')
    
    args = parser.parse_args()
    
    if args.skip_update:
        print("⏭️  Skipping data update, running scanner only...")
        scanner_runner = ScannerRunner(args.config)
        scanner_runner.run(args.scanner, args.symbols)
    else:
        update_and_scan(
            scanner_type=args.scanner,
            symbols=args.symbols,
            provider=args.provider,
            timeframe=args.timeframe,
            days_back=args.days_back,
            config_file=args.config
        )

if __name__ == "__main__":
    main()
