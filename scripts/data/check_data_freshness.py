#!/usr/bin/env python3
"""
Data Freshness Checker
Checks how recent the data in TimescaleDB is
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.db.timescaledb_client import get_timescaledb_client

def check_data_freshness(symbols=None, provider="ALPACA", timeframe="1d"):
    """
    Check how fresh the data is in TimescaleDB
    
    Args:
        symbols: List of symbols to check (None = check all)
        provider: Data provider to check
        timeframe: Data timeframe to check
    """
    print("📊 Data Freshness Check")
    print("=" * 50)
    
    try:
        # Connect to database
        client = get_timescaledb_client()
        if not client.ensure_connection():
            print("❌ Cannot connect to TimescaleDB")
            return
        
        # Get data freshness for symbols
        if symbols is None:
            # Get all symbols from database
            query = """
            SELECT DISTINCT symbol 
            FROM market_data 
            WHERE provider = %s AND timeframe = %s
            ORDER BY symbol
            """
            result = client.execute_query(query, (provider, timeframe))
            symbols = [row['symbol'] for row in result] if result else []
        
        print(f"Checking {len(symbols)} symbols for {provider} {timeframe} data...")
        
        # Check each symbol
        fresh_data = []
        stale_data = []
        no_data = []
        
        for symbol in symbols:
            try:
                # Get latest data for this symbol
                query = """
                SELECT MAX(timestamp) as latest_date, COUNT(*) as record_count
                FROM market_data 
                WHERE symbol = %s AND provider = %s AND timeframe = %s
                """
                result = client.execute_query(query, (symbol, provider, timeframe))
                
                if result and result[0]['latest_date']:
                    latest_date = result[0]['latest_date']
                    record_count = result[0]['record_count']
                    
                    # Calculate days since latest data
                    if isinstance(latest_date, str):
                        latest_date = datetime.fromisoformat(latest_date.replace('Z', '+00:00'))
                    
                    days_old = (datetime.now() - latest_date.replace(tzinfo=None)).days
                    
                    if days_old <= 1:
                        fresh_data.append((symbol, latest_date, days_old, record_count))
                    elif days_old <= 7:
                        stale_data.append((symbol, latest_date, days_old, record_count))
                    else:
                        stale_data.append((symbol, latest_date, days_old, record_count))
                else:
                    no_data.append(symbol)
                    
            except Exception as e:
                print(f"   ❌ Error checking {symbol}: {e}")
                no_data.append(symbol)
        
        # Print results
        print(f"\n📈 Data Freshness Summary:")
        print(f"   ✅ Fresh data (≤1 day old): {len(fresh_data)}")
        print(f"   ⚠️  Stale data (>1 day old): {len(stale_data)}")
        print(f"   ❌ No data: {len(no_data)}")
        
        if fresh_data:
            print(f"\n✅ Fresh Data ({len(fresh_data)} symbols):")
            for symbol, date, days, count in fresh_data[:10]:  # Show first 10
                print(f"   {symbol}: {date.strftime('%Y-%m-%d')} ({days} days old, {count} records)")
            if len(fresh_data) > 10:
                print(f"   ... and {len(fresh_data) - 10} more")
        
        if stale_data:
            print(f"\n⚠️  Stale Data ({len(stale_data)} symbols):")
            for symbol, date, days, count in stale_data[:10]:  # Show first 10
                print(f"   {symbol}: {date.strftime('%Y-%m-%d')} ({days} days old, {count} records)")
            if len(stale_data) > 10:
                print(f"   ... and {len(stale_data) - 10} more")
        
        if no_data:
            print(f"\n❌ No Data ({len(no_data)} symbols):")
            for symbol in no_data[:10]:  # Show first 10
                print(f"   {symbol}")
            if len(no_data) > 10:
                print(f"   ... and {len(no_data) - 10} more")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if stale_data:
            print(f"   • Run data update for stale symbols:")
            print(f"     python update_and_scan.py --symbols {' '.join([s[0] for s in stale_data[:5]])}")
        if no_data:
            print(f"   • Fetch data for symbols with no data:")
            print(f"     python utils/fetch_data.py --symbol {no_data[0]} --provider {provider} --timeframe {timeframe} --bars max")
        
        if not stale_data and not no_data:
            print(f"   • All data is fresh! You can run scanners directly.")
            print(f"     python scanner_runner.py --scanner hl_after_ll")
        
        client.disconnect()
        
    except Exception as e:
        print(f"❌ Error checking data freshness: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Data Freshness")
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to check')
    parser.add_argument('--provider', type=str, choices=['ALPACA', 'IB'], default='ALPACA',
                       help='Data provider to check')
    parser.add_argument('--timeframe', type=str, choices=['1d', '1h'], default='1d',
                       help='Data timeframe to check')
    
    args = parser.parse_args()
    
    check_data_freshness(args.symbols, args.provider, args.timeframe)

if __name__ == "__main__":
    main()
