#!/usr/bin/env python3
"""
Check weekly aggregation from 15m data to identify potential aggregation issues.
This will help diagnose if the weekly bars differ from TradingView due to aggregation.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from utils.db.timescaledb_client import get_timescaledb_client
except ImportError:
    from utils.db.timescaledb_client import get_timescaledb_client

from backtrader_runner_yaml import resample_to_weekly_pandas

def analyze_weekly_aggregation(symbol, target_date):
    """Analyze how weekly bars are aggregated from 15m data."""
    print("=" * 100)
    print(f"Weekly Aggregation Analysis for {symbol} - Week containing {target_date}")
    print("=" * 100)
    print()
    
    # Load 15m data from database
    client = get_timescaledb_client()
    if not client.ensure_connection():
        print("ERROR: Cannot connect to TimescaleDB")
        return
    
    # Get 15m data for a range around the target date
    start_time = pd.Timestamp(target_date) - pd.Timedelta(days=14)
    end_time = pd.Timestamp(target_date) + pd.Timedelta(days=14)
    
    print(f"Loading 15m data from {start_time.date()} to {end_time.date()}...")
    
    # Load 15m data
    df_15m = client.get_market_data(symbol, "15m", start_time=start_time, end_time=end_time)
    
    if df_15m is None or df_15m.empty:
        print(f"ERROR: No 15m data found for {symbol}")
        client.disconnect()
        return
    
    print(f"Loaded {len(df_15m)} 15m bars")
    print(f"Date range: {df_15m['ts'].min()} to {df_15m['ts'].max()}")
    print()
    
    # Convert to DataFrame with datetime index
    df_15m['ts'] = pd.to_datetime(df_15m['ts'])
    if df_15m['ts'].dt.tz is not None:
        df_15m['ts'] = df_15m['ts'].dt.tz_convert(None)
    
    df_15m = df_15m.set_index('ts')
    df_15m = df_15m[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # Aggregate to weekly
    df_weekly = resample_to_weekly_pandas(df_15m)
    
    print(f"Aggregated to {len(df_weekly)} weekly bars")
    print()
    
    # Find the week containing target_date
    target_ts = pd.Timestamp(target_date)
    closest_idx = df_weekly.index.get_indexer([target_ts], method='nearest')[0]
    closest_week = df_weekly.index[closest_idx]
    week_row = df_weekly.iloc[closest_idx]
    
    print("=" * 100)
    print(f"Weekly Bar for week ending {closest_week.date()}:")
    print("=" * 100)
    print(f"Open:  {week_row['open']:.2f}")
    print(f"High:  {week_row['high']:.2f}")
    print(f"Low:   {week_row['low']:.2f}")
    print(f"Close: {week_row['close']:.2f}")
    print(f"Volume: {week_row['volume']:,.0f}")
    print()
    
    # Show which 15m bars contributed to this weekly bar
    # W-FRI means week ending on Friday, so the week runs from the previous Saturday to this Friday
    # But pandas W-FRI actually groups from Monday to Friday
    week_start = closest_week - pd.Timedelta(days=6)  # Monday of the week
    week_end = closest_week  # Friday of the week
    
    week_15m = df_15m[(df_15m.index >= week_start) & (df_15m.index <= week_end)]
    
    print("=" * 100)
    print(f"15m Bars contributing to this weekly bar ({len(week_15m)} bars):")
    print("=" * 100)
    print(f"Date range: {week_15m.index.min()} to {week_15m.index.max()}")
    print()
    
    # Check for potential issues
    print("=" * 100)
    print("Potential Aggregation Issues:")
    print("=" * 100)
    
    # 1. Check timezone
    print(f"1. Timezone: Data is timezone-naive (converted from UTC)")
    
    # 2. Check for gaps
    expected_bars_per_day = 26  # 6.5 hours * 4 (15m bars per hour) = 26 bars per trading day
    expected_bars_per_week = expected_bars_per_day * 5  # 5 trading days
    actual_bars = len(week_15m)
    
    print(f"2. Data Completeness:")
    print(f"   Expected ~{expected_bars_per_week} 15m bars for a full week")
    print(f"   Actual: {actual_bars} bars")
    if actual_bars < expected_bars_per_week * 0.9:
        print(f"   WARNING: Missing {expected_bars_per_week - actual_bars} bars ({(1 - actual_bars/expected_bars_per_week)*100:.1f}% missing)")
    else:
        print(f"   OK: Data appears complete")
    
    # Check what days are included
    week_days = pd.Series(week_15m.index.date).unique()
    print(f"   Days included: {sorted(week_days)}")
    if len(week_days) < 5:
        print(f"   WARNING: Only {len(week_days)} trading days in this week (expected 5)")
    
    # 3. Check for pre/post-market data
    # Regular trading hours: 9:30 AM - 4:00 PM ET
    # Note: Data is in UTC, so we need to account for timezone
    # ET is UTC-5 (or UTC-4 during DST)
    # 9:30 AM ET = 13:30 UTC (or 14:30 during DST)
    # 4:00 PM ET = 20:00 UTC (or 21:00 during DST)
    week_15m_hour = week_15m.index.hour
    # Check for bars outside 13:00-21:00 UTC (approximate RTH window accounting for DST)
    outside_rth = week_15m[(week_15m_hour < 13) | (week_15m_hour >= 21)]
    
    print(f"3. Trading Hours:")
    print(f"   Regular trading hours: 9:30 AM - 4:00 PM ET (approx 13:30-20:00 UTC)")
    print(f"   Bars outside RTH window: {len(outside_rth)}")
    if len(outside_rth) > 0:
        print(f"   WARNING: {len(outside_rth)} bars are outside regular trading hours window")
        print(f"   This could cause weekly aggregation to differ from TradingView")
    else:
        print(f"   OK: All bars are within regular trading hours window")
    
    # 4. Check Friday close time
    friday_bars = week_15m[week_15m.index.weekday == 4]  # Friday = 4
    if len(friday_bars) > 0:
        last_friday_bar = friday_bars.index.max()
        last_friday_time = last_friday_bar.time()
        print(f"4. Friday Close Time:")
        print(f"   Last Friday bar: {last_friday_bar}")
        print(f"   Time (UTC): {last_friday_time}")
        # 4:00 PM ET = 20:00 UTC (or 21:00 during DST)
        if last_friday_time.hour < 19 or last_friday_time.hour > 21:
            print(f"   WARNING: Friday close time ({last_friday_time} UTC) might not match TradingView")
        else:
            print(f"   OK: Friday close time appears correct")
    else:
        print(f"4. Friday Close Time:")
        print(f"   WARNING: No Friday bars found in this week!")
    
    # 5. Show sample of 15m bars
    print()
    print("=" * 100)
    print("Sample 15m bars (first 10 and last 10 of the week):")
    print("=" * 100)
    print(f"{'Timestamp':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    print("-" * 100)
    
    for idx in list(week_15m.head(10).index) + list(week_15m.tail(10).index):
        row = week_15m.loc[idx]
        print(f"{idx} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} "
              f"{row['close']:>10.2f} {row['volume']:>12,.0f}")
    
    client.disconnect()
    
    print()
    print("=" * 100)
    print("Summary:")
    print("=" * 100)
    print("If weekly bars differ from TradingView, check:")
    print("1. Missing 15m bars (data gaps)")
    print("2. Pre/post-market data inclusion")
    print("3. Friday close time alignment")
    print("4. Timezone handling")
    print("5. Holiday week handling")

def main():
    symbol = "HUBB"
    target_date = datetime(2019, 6, 14)
    
    analyze_weekly_aggregation(symbol, target_date)

if __name__ == '__main__':
    main()

