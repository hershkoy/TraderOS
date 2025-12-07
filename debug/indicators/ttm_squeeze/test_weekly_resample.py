#!/usr/bin/env python3
"""
Test weekly resampling to see if Friday June 14, 2019 is included correctly.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.db.timescaledb_loader import load_timescaledb_15m
from backtrader_runner_yaml import resample_to_weekly_pandas

def test_weekly_resample():
    symbol = "HUBB"
    provider = "IB"
    
    print("=" * 100)
    print(f"Testing weekly resampling for {symbol} (June 2019)")
    print("=" * 100)
    print()
    
    # Load 15m data for June 2019 (same range as the backtest)
    print("Loading 15m data from TimescaleDB...")
    df_15m = load_timescaledb_15m(
        symbol, 
        provider, 
        start_date="2019-06-01", 
        end_date="2019-06-30"
    )
    
    print(f"Loaded {len(df_15m)} 15m bars")
    print(f"Date range: {df_15m.index.min()} to {df_15m.index.max()}")
    print()
    
    # Check for Friday June 14 data
    friday_date = datetime(2019, 6, 14).date()
    friday_data = df_15m[df_15m.index.date == friday_date]
    print(f"Friday June 14, 2019: {len(friday_data)} bars")
    if len(friday_data) > 0:
        print(f"  Time range: {friday_data.index.min()} to {friday_data.index.max()}")
        print(f"  First bar timestamp: {friday_data.index[0]} (day: {friday_data.index[0].date()}, weekday: {friday_data.index[0].strftime('%A')})")
        print(f"  Last bar timestamp: {friday_data.index[-1]} (day: {friday_data.index[-1].date()}, weekday: {friday_data.index[-1].strftime('%A')})")
        print(f"  Close: {friday_data['close'].iloc[-1]:.2f}")
        print(f"  High: {friday_data['high'].max():.2f}")
        print(f"  Low: {friday_data['low'].min():.2f}")
        print(f"  Volume: {friday_data['volume'].sum():,.0f}")
    print()
    
    # Check what pandas resample thinks the week boundaries are
    print("Checking pandas resample week boundaries...")
    # Create a test with just Friday data
    test_df = df_15m.copy()
    # Check which week Friday June 14 belongs to according to pandas
    friday_timestamp = pd.Timestamp('2019-06-14 19:45:00')
    week_end = friday_timestamp.to_period('W-FRI').end_time
    print(f"Friday June 14, 19:45 -> Week ending: {week_end.date()}")
    print()
    
    # Resample to weekly
    print("Resampling to weekly (W-FRI)...")
    df_weekly = resample_to_weekly_pandas(df_15m)
    
    print(f"Created {len(df_weekly)} weekly bars")
    print()
    
    # Find the weekly bar that should include June 14
    print("Weekly bars in June 2019:")
    print("-" * 100)
    print(f"{'Week End Date':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>15}")
    print("-" * 100)
    
    june_weeks = df_weekly[df_weekly.index.month == 6]
    for week_end in june_weeks.index:
        week_start = week_end - pd.Timedelta(days=6)
        row = june_weeks.loc[week_end]
        includes_friday = (week_start.date() <= friday_date <= week_end.date())
        marker = " <-- JUNE 14" if includes_friday else ""
        print(f"{week_end.date()}        {row['open']:>10.2f} {row['high']:>10.2f} "
              f"{row['low']:>10.2f} {row['close']:>10.2f} {row['volume']:>15,.0f}{marker}")
    
    print()
    
    # Check the specific week ending June 14
    week_ending_june14 = df_weekly[df_weekly.index.date == friday_date]
    if len(week_ending_june14) > 0:
        print("=" * 100)
        print("Week ending June 14, 2019:")
        print("=" * 100)
        week_bar = week_ending_june14.iloc[0]
        print(f"Open: {week_bar['open']:.2f}")
        print(f"High: {week_bar['high']:.2f}")
        print(f"Low: {week_bar['low']:.2f}")
        print(f"Close: {week_bar['close']:.2f}")
        print(f"Volume: {week_bar['volume']:,.0f}")
        print()
        
        # Check what 15m bars are included in this week
        week_end_timestamp = week_ending_june14.index[0]
        print(f"Week ending timestamp: {week_end_timestamp}")
        print(f"Week ending timestamp type: {type(week_end_timestamp)}")
        print()
        
        # For W-FRI, the week should be Monday to Friday
        # Monday = Friday - 4 days
        week_start = week_end_timestamp - pd.Timedelta(days=4)
        print(f"Week start (Monday): {week_start}")
        print(f"Week end (Friday): {week_end_timestamp}")
        print()
        
        # The weekly bar timestamp is at 00:00:00 (start of Friday)
        # But we need to include all of Friday, so check up to end of Friday
        week_end_inclusive = week_end_timestamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        print(f"Week end inclusive (end of Friday): {week_end_inclusive}")
        print()
        
        week_data = df_15m[(df_15m.index >= week_start) & (df_15m.index <= week_end_inclusive)]
        unique_days = pd.Series(week_data.index.date).nunique()
        unique_dates = sorted(pd.Series(week_data.index.date).unique())
        print(f"15m bars included: {len(week_data)}")
        print(f"Unique trading days: {unique_days}")
        print(f"Trading days in week: {[str(d) for d in unique_dates]}")
        print()
        
        # Check which day of the week each date is
        expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        actual_days = [pd.Timestamp(d).strftime('%A') for d in unique_dates]
        print(f"Days of week: {actual_days}")
        print()
        
        if unique_days < 5:
            print("WARNING: Incomplete week! Missing trading days.")
            # Find which day is missing
            week_start_date = week_start.date()
            week_end_date = week_ending_june14.index[0].date()
            all_weekdays = pd.date_range(week_start_date, week_end_date, freq='B')  # Business days only
            missing_days = [d.date() for d in all_weekdays if d.date() not in unique_dates]
            if missing_days:
                print(f"Missing days: {[str(d) for d in missing_days]}")
        else:
            print("OK: Complete week with all 5 trading days")
    else:
        print("=" * 100)
        print("ERROR: No weekly bar found ending on June 14, 2019!")
        print("=" * 100)
        print()
        print("This means the weekly resampling did not create a bar for that week.")
        print("Possible reasons:")
        print("1. The week boundary calculation is off")
        print("2. There's not enough data for that week")
        print("3. The resample logic has a bug")

if __name__ == "__main__":
    test_weekly_resample()

