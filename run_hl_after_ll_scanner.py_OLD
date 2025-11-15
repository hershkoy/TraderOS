#!/usr/bin/env python3
"""
HL After LL Scanner Runner
Scans for LL → HH → HL patterns in weekly data
Runs on Monday after market open with 1 left bar, 2 right bar confirmation
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hl_after_ll_scanner import scan_symbol_for_setup, scan_universe, load_from_timescaledb
from utils.ticker_universe import get_ticker_universe

def setup_logging():
    """Setup logging for the scanner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/hl_after_ll_scanner.log')
        ]
    )
    return logging.getLogger(__name__)

def scan_from_timescaledb(symbols=None, timeframe="1d"):
    """
    Scan symbols from TimescaleDB for HL after LL patterns
    
    Args:
        symbols: List of symbols to scan (None = scan all available)
        timeframe: Data timeframe to use (default: "1d" for daily data)
    
    Returns:
        List of Match objects for symbols with LL → HH → HL patterns
    """
    logger = setup_logging()
    
    try:
        # Get symbols to scan
        if symbols is None:
            logger.info("Loading ticker universe...")
            universe = get_ticker_universe()
            symbols = list(universe.keys())
            logger.info(f"Loaded {len(symbols)} symbols from universe")
        else:
            logger.info(f"Using provided {len(symbols)} symbols")
        
        # Load data from TimescaleDB
        logger.info(f"Loading {timeframe} data from TimescaleDB for {len(symbols)} symbols...")
        ohlcv_data = load_from_timescaledb(symbols, timeframe)
        logger.info(f"Successfully loaded data for {len(ohlcv_data)} symbols")
        
        if not ohlcv_data:
            logger.warning("No data loaded from TimescaleDB")
            return []
        
        # Scan for patterns
        logger.info("Scanning for LL → HH → HL patterns...")
        matches = scan_universe(ohlcv_data)
        
        logger.info(f"Found {len(matches)} symbols with LL → HH → HL patterns")
        return matches
        
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def print_matches(matches):
    """Print scan results in a formatted way"""
    if not matches:
        print("\n❌ No LL → HH → HL patterns found")
        return
    
    print(f"\n✅ Found {len(matches)} symbols with LL → HH → HL patterns:")
    print("=" * 80)
    
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match.symbol}")
        print(f"   LL: {match.ll_date.strftime('%Y-%m-%d')} at ${match.ll_price:.2f}")
        
        # Print HH details
        for j, (hh_date, hh_price) in enumerate(zip(match.hh_dates, match.hh_prices)):
            print(f"   HH{j+1}: {hh_date.strftime('%Y-%m-%d')} at ${hh_price:.2f}")
        
        print(f"   HL: {match.hl_date.strftime('%Y-%m-%d')} at ${match.hl_price:.2f}")
        print(f"   Current Price: ${match.last_price:.2f}")
        
        # Check if current price is above HL (Monday check)
        if match.last_price >= match.hl_price:
            print(f"   ✅ Monday Check: Current price (${match.last_price:.2f}) >= HL (${match.hl_price:.2f})")
        else:
            print(f"   ❌ Monday Check: Current price (${match.last_price:.2f}) < HL (${match.hl_price:.2f}) - Pattern broken")

def save_results_to_csv(matches, filename=None):
    """Save scan results to CSV file"""
    if not matches:
        print("No matches to save")
        return
    
    import pandas as pd
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/hl_after_ll_scan_{timestamp}.csv"
    
    # Ensure reports directory exists
    Path("reports").mkdir(exist_ok=True)
    
    # Prepare data for CSV
    data = []
    for match in matches:
        # Create one row per HH
        for i, (hh_date, hh_price) in enumerate(zip(match.hh_dates, match.hh_prices)):
            data.append({
                'symbol': match.symbol,
                'll_date': match.ll_date.strftime('%Y-%m-%d'),
                'll_price': match.ll_price,
                'hh_number': i + 1,
                'hh_date': hh_date.strftime('%Y-%m-%d'),
                'hh_price': hh_price,
                'hl_date': match.hl_date.strftime('%Y-%m-%d'),
                'hl_price': match.hl_price,
                'current_price': match.last_price,
                'monday_check_pass': match.last_price >= match.hl_price,
                'pattern_type': f"LL→HH{'→HH' * (i)}→HL" if i > 0 else "LL→HH→HL"
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n📊 Results saved to: {filename}")

def main():
    """Main function to run the HL after LL scanner"""
    print("🔍 HL After LL Scanner")
    print("=" * 50)
    print("Scanning for LL → HH → HL patterns in weekly data")
    print("Configuration: 1 left bar, 2 right bar confirmation")
    print("Monday check: Current price must be >= HL price")
    print()
    
    # Check if it's Monday (optional)
    today = datetime.now()
    if today.weekday() == 0:  # Monday
        print("✅ Running on Monday - perfect timing!")
    else:
        print(f"⚠️  Running on {today.strftime('%A')} - consider running on Monday for best results")
    
    print()
    
    # Run the scan
    matches = scan_from_timescaledb()
    
    # Display results
    print_matches(matches)
    
    # Save to CSV
    if matches:
        save_results_to_csv(matches)
    
    print(f"\n🎯 Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
