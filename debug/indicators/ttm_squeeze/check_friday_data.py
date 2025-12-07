#!/usr/bin/env python3
"""
Check if Friday June 14, 2019 data exists in the database.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from utils.db.timescaledb_client import get_timescaledb_client
except ImportError:
    from utils.db.timescaledb_client import get_timescaledb_client

def check_friday_data(symbol, target_date):
    """Check if Friday data exists for the target week."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        print("ERROR: Cannot connect to TimescaleDB")
        return
    
    # Check for Friday June 14, 2019
    friday_date = datetime(2019, 6, 14)
    
    # Use a wider range to ensure we capture all data for that day
    # Market hours in ET are 9:30 AM - 4:00 PM, which is 13:30-20:00 UTC
    # But we need to account for timezone differences and ensure we get the full day
    start_time = pd.Timestamp(friday_date).replace(hour=0, minute=0, second=0) - pd.Timedelta(hours=5)  # Start of day in ET (UTC-5)
    end_time = pd.Timestamp(friday_date).replace(hour=23, minute=59, second=59) + pd.Timedelta(hours=5)  # End of day in ET (UTC+5)
    
    print("=" * 100)
    print(f"Checking for Friday {friday_date.date()} data for {symbol}")
    print("=" * 100)
    print()
    
    # Check both 15m and daily data
    print("Checking 15m data...")
    df_15m = client.get_market_data(symbol, "15m", start_time=start_time, end_time=end_time)
    
    print("Checking daily data...")
    df_daily = client.get_market_data(symbol, "1d", start_time=start_time, end_time=end_time)
    
    # Check daily data first
    friday_daily = None
    if df_daily is not None and not df_daily.empty:
        df_daily['ts'] = pd.to_datetime(df_daily['ts'])
        if df_daily['ts'].dt.tz is not None:
            df_daily['ts'] = df_daily['ts'].dt.tz_convert(None)
        df_daily = df_daily.set_index('ts')
        friday_daily = df_daily[df_daily.index.date == friday_date.date()]
        if len(friday_daily) > 0:
            print(f"OK: Found daily data for Friday {friday_date.date()}")
            print(f"  Daily bar: O={friday_daily['open'].iloc[0]:.2f}, H={friday_daily['high'].iloc[0]:.2f}, "
                  f"L={friday_daily['low'].iloc[0]:.2f}, C={friday_daily['close'].iloc[0]:.2f}, "
                  f"V={friday_daily['volume'].iloc[0]:,.0f}")
        else:
            print(f"WARNING: No daily data for Friday {friday_date.date()}")
    else:
        print(f"WARNING: No daily data found in range")
    
    print()
    
    # Check 15m data
    friday_15m = None
    if df_15m is None or df_15m.empty:
        print(f"✗ No 15m data found for {friday_date.date()}")
        print()
        print("=" * 100)
        print("ROOT CAUSE IDENTIFIED:")
        print("=" * 100)
        if friday_daily is not None and len(friday_daily) > 0:
            print("Friday June 14, 2019 exists in DAILY data but NOT in 15m data!")
            print()
            print("This is why the weekly aggregation is incomplete:")
            print("- Weekly bars are aggregated from 15m data")
            print("- Friday 15m data is missing")
            print("- So the weekly bar only includes Mon-Thu")
            print("- This causes incorrect TTM Squeeze calculations")
            print()
            print("SOLUTION: Fetch 15m data for missing Fridays, or use daily data")
            print("to aggregate weekly bars instead of 15m data.")
        else:
            print("Friday June 14, 2019 data is missing from both 15m and daily!")
            print("This explains why the weekly bar is incomplete.")
    else:
        # Convert to DataFrame
        df_15m['ts'] = pd.to_datetime(df_15m['ts'])
        if df_15m['ts'].dt.tz is not None:
            df_15m['ts'] = df_15m['ts'].dt.tz_convert(None)
        
        df_15m = df_15m.set_index('ts')
        df_15m = df_15m[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Filter for Friday June 14
        friday_15m = df_15m[df_15m.index.date == friday_date.date()]
        
        if len(friday_15m) > 0:
            print(f"OK: Found {len(friday_15m)} 15m bars for Friday {friday_date.date()}")
        else:
            print(f"WARNING: No 15m bars found for Friday {friday_date.date()}")
            if friday_daily is not None and len(friday_daily) > 0:
                print("  (But daily data exists - see above)")
    
    print()
    
    # Initialize friday_data
    friday_data = None
    if friday_15m is not None and len(friday_15m) > 0:
        friday_data = friday_15m
    
    if friday_data is not None and len(friday_data) > 0:
        print("Friday June 14, 2019 15m bars:")
        print("-" * 100)
        print(f"{'Timestamp':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
        print("-" * 100)
        
        for idx, row in friday_data.iterrows():
            print(f"{idx} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} "
                  f"{row['close']:>10.2f} {row['volume']:>12,.0f}")
        
        print()
        print("=" * 100)
        print("Analysis:")
        print("=" * 100)
        print(f"Friday close: {friday_data['close'].iloc[-1]:.2f}")
        print(f"Friday high: {friday_data['high'].max():.2f}")
        print(f"Friday low: {friday_data['low'].min():.2f}")
        print(f"Friday volume: {friday_data['volume'].sum():,.0f}")
        print()
        print("If this data exists but wasn't included in the weekly bar, there's a timezone")
        print("or aggregation boundary issue.")
    else:
        print("=" * 100)
        print("CRITICAL: No Friday June 14, 2019 data found!")
        print("=" * 100)
        print()
        print("This is why the weekly bar is incomplete.")
        print("The weekly bar for 'week ending 2019-06-14' doesn't include Friday because")
        print("the Friday data is missing from the database.")
        print()
        print("This explains the incorrect TTM cross detection - the weekly bar is based")
        print("on incomplete data (only Mon-Thu), which causes different momentum values.")
    
    # Also check the surrounding days to see the pattern
    print()
    print("=" * 100)
    print("Surrounding days (June 10-17, 2019):")
    print("=" * 100)
    
    week_start = datetime(2019, 6, 10)
    week_end = datetime(2019, 6, 17)
    
    week_data = df_15m[(df_15m.index.date >= week_start.date()) & (df_15m.index.date < week_end.date())]
    
    days = pd.Series(week_data.index.date).unique()
    print(f"Days with data: {sorted(days)}")
    
    for day in sorted(days):
        day_data = week_data[week_data.index.date == day]
        print(f"  {day}: {len(day_data)} bars")
    
    client.disconnect()

def main():
    check_friday_data("HUBB", datetime(2019, 6, 14))

if __name__ == '__main__':
    main()

