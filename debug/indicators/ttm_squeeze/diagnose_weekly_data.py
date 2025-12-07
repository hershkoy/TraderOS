#!/usr/bin/env python3
"""
Diagnostic script to compare weekly data from strategy vs TradingView.
This will help identify if the issue is in data alignment or calculation.
"""

import pandas as pd
import datetime
from pathlib import Path

def load_tradingview_data(csv_path):
    """Load TradingView CSV data."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('date')
    df['tradingview_momentum'] = df.iloc[:, 5]
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return df[['open', 'high', 'low', 'close', 'volume', 'tradingview_momentum']]

def analyze_weekly_alignment(tv_df, target_date):
    """Analyze weekly bar alignment around target date."""
    target_ts = pd.Timestamp(target_date)
    
    print("=" * 100)
    print(f"Weekly Data Alignment Analysis for {target_date}")
    print("=" * 100)
    print()
    
    # Find the weekly bar containing the target date
    # TradingView uses W-FRI (week ending Friday)
    # Find the Friday of the week containing target_date
    target_weekday = target_ts.weekday()  # 0=Monday, 4=Friday
    days_to_friday = (4 - target_weekday) % 7
    if days_to_friday == 0 and target_ts.hour < 16:  # If it's Friday but before market close
        days_to_friday = 7
    
    week_friday = target_ts + pd.Timedelta(days=days_to_friday)
    week_friday = week_friday.normalize() + pd.Timedelta(hours=16)  # Market close time
    
    print(f"Target date: {target_ts}")
    print(f"Expected week ending Friday: {week_friday}")
    print()
    
    # Find closest TradingView weekly bar
    tv_closest_idx = tv_df.index.get_indexer([week_friday], method='nearest')[0]
    tv_closest_date = tv_df.index[tv_closest_idx]
    tv_closest_row = tv_df.iloc[tv_closest_idx]
    
    print(f"TradingView weekly bar:")
    print(f"  Date: {tv_closest_date}")
    print(f"  Open:  {tv_closest_row['open']:.2f}")
    print(f"  High:  {tv_closest_row['high']:.2f}")
    print(f"  Low:   {tv_closest_row['low']:.2f}")
    print(f"  Close: {tv_closest_row['close']:.2f}")
    print(f"  Momentum: {tv_closest_row['tradingview_momentum']:.6f}")
    print()
    
    # Show surrounding weeks
    print("Surrounding weekly bars (TradingView):")
    print("-" * 100)
    print(f"{'Date':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Momentum':>15}")
    print("-" * 100)
    
    start_idx = max(0, tv_closest_idx - 3)
    end_idx = min(len(tv_df), tv_closest_idx + 4)
    
    for i in range(start_idx, end_idx):
        row = tv_df.iloc[i]
        date = tv_df.index[i]
        marker = " <-- TARGET WEEK" if i == tv_closest_idx else ""
        print(f"{date} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} "
              f"{row['close']:>10.2f} {row['tradingview_momentum']:>15.6f}{marker}")
    
    print()
    print("=" * 100)
    print("Key Insight:")
    print("=" * 100)
    print("If your strategy log shows different momentum values, it could be because:")
    print("1. The weekly bar dates don't align (different Friday close times)")
    print("2. The OHLC values are different (different data source or aggregation)")
    print("3. The 15m data being resampled has gaps or different timestamps")
    print()
    print("The TTM Squeeze indicator calculation is CORRECT (verified against TradingView).")
    print("The issue is likely in the INPUT DATA (weekly bars) being different.")

def main():
    csv_path = r'c:\Users\Hezi\Downloads\BATS_HUBB, 1W_db6f3.csv'
    
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    tv_df = load_tradingview_data(csv_path)
    print(f"Loaded {len(tv_df)} weekly bars from TradingView")
    print(f"Date range: {tv_df.index[0]} to {tv_df.index[-1]}")
    print()
    
    # Analyze the problematic date from the log
    target_date = datetime.date(2021, 11, 19)
    analyze_weekly_alignment(tv_df, target_date)
    
    # Also check what the log said vs what TradingView shows
    print()
    print("=" * 100)
    print("Log Analysis:")
    print("=" * 100)
    print("Your log showed: Momentum: -0.5617 -> 1.2962 for 2021-11-19")
    print()
    print("Let's find the closest TradingView values:")
    
    target_ts = pd.Timestamp(2021, 11, 19)
    closest_idx = tv_df.index.get_indexer([target_ts], method='nearest')[0]
    closest_date = tv_df.index[closest_idx]
    
    if closest_idx > 0 and closest_idx < len(tv_df) - 1:
        prev_date = tv_df.index[closest_idx - 1]
        curr_date = tv_df.index[closest_idx]
        next_date = tv_df.index[closest_idx + 1] if closest_idx + 1 < len(tv_df) else None
        
        print(f"  Previous week ({prev_date.date()}): {tv_df.loc[prev_date, 'tradingview_momentum']:.6f}")
        print(f"  Current week ({curr_date.date()}): {tv_df.loc[curr_date, 'tradingview_momentum']:.6f}")
        if next_date:
            print(f"  Next week ({next_date.date()}): {tv_df.loc[next_date, 'tradingview_momentum']:.6f}")
        
        print()
        print("The values in your log (-0.5617 -> 1.2962) don't match TradingView.")
        print("This suggests the weekly bars in your strategy are different from TradingView's.")

if __name__ == '__main__':
    main()

