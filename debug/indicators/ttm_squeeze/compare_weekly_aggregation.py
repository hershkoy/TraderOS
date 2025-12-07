#!/usr/bin/env python3
"""
Compare our weekly aggregation with TradingView to identify discrepancies.
This will help identify if the issue is in how we aggregate weekly bars.
"""

import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtrader_runner_yaml import resample_to_weekly_pandas

def load_tradingview_data(csv_path):
    """Load TradingView CSV data."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('date')
    df['tradingview_momentum'] = df.iloc[:, 5]
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return df[['open', 'high', 'low', 'close', 'volume', 'tradingview_momentum']]

def simulate_15m_to_weekly(weekly_tv_data):
    """
    Simulate what would happen if we had 15m data and aggregated it to weekly.
    Since we don't have the original 15m data, we'll create synthetic 15m bars
    from the weekly data to test the aggregation logic.
    """
    # This is a simplified simulation - in reality we'd need actual 15m data
    # But we can at least verify the aggregation logic is correct
    pass

def analyze_weekly_bar_differences(tv_df, target_date, target_week_idx=None):
    """Analyze differences in weekly bar aggregation."""
    target_ts = pd.Timestamp(target_date)
    
    print("=" * 100)
    print(f"Weekly Aggregation Analysis for {target_date}")
    print("=" * 100)
    print()
    
    # Find TradingView weekly bar for this date
    tv_closest_idx = tv_df.index.get_indexer([target_ts], method='nearest')[0]
    tv_closest_date = tv_df.index[tv_closest_idx]
    tv_row = tv_df.iloc[tv_closest_idx]
    
    print("TradingView Weekly Bar:")
    print(f"  Date: {tv_closest_date}")
    print(f"  Open:  {tv_row['open']:.2f}")
    print(f"  High:  {tv_row['high']:.2f}")
    print(f"  Low:   {tv_row['low']:.2f}")
    print(f"  Close: {tv_row['close']:.2f}")
    print(f"  Volume: {tv_row['volume']:,.0f}")
    print(f"  Momentum: {tv_row['tradingview_momentum']:.6f}")
    print()
    
    # Show surrounding weeks to understand the pattern
    print("Surrounding TradingView Weekly Bars:")
    print("-" * 100)
    print(f"{'Date':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>15} {'Momentum':>15}")
    print("-" * 100)
    
    start_idx = max(0, tv_closest_idx - 5)
    end_idx = min(len(tv_df), tv_closest_idx + 6)
    
    for i in range(start_idx, end_idx):
        row = tv_df.iloc[i]
        date = tv_df.index[i]
        marker = " <-- TARGET" if i == tv_closest_idx else ""
        print(f"{date} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} "
              f"{row['close']:>10.2f} {row['volume']:>15,.0f} {row['tradingview_momentum']:>15.6f}{marker}")
    
    print()
    print("=" * 100)
    print("Potential Aggregation Issues:")
    print("=" * 100)
    print("1. **Week Boundary Alignment**:")
    print("   - Our code uses 'W-FRI' (week ending Friday)")
    print("   - TradingView also uses W-FRI")
    print("   - But the exact Friday close time might differ (16:00 ET vs other times)")
    print()
    print("2. **Data Source Differences**:")
    print("   - We aggregate from 15m IB data")
    print("   - TradingView might use daily bars or different data source")
    print("   - Missing 15m bars could cause incorrect weekly aggregation")
    print()
    print("3. **Timezone Issues**:")
    print("   - IB data might be in UTC or ET")
    print("   - TradingView might use a different timezone")
    print("   - This could shift which bars belong to which week")
    print()
    print("4. **Market Hours**:")
    print("   - If 15m data includes pre/post-market, weekly bars will differ")
    print("   - TradingView typically uses regular trading hours only")
    print()
    print("5. **Holiday Handling**:")
    print("   - Short weeks (holidays) might be aggregated differently")
    print("   - TradingView might skip or adjust for holidays")
    print()
    print("=" * 100)
    print("Recommendation:")
    print("=" * 100)
    print("To verify if aggregation is the issue:")
    print("1. Check the actual 15m data for the week containing the target date")
    print("2. Manually aggregate it to weekly and compare OHLC values")
    print("3. Verify timezone handling - ensure all times are in the same timezone")
    print("4. Check if pre/post-market data is included in the 15m feed")
    print("5. Compare the exact Friday close times between our data and TradingView")

def main():
    csv_path = r'c:\Users\Hezi\Downloads\BATS_HUBB, 1W_db6f3.csv'
    
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    tv_df = load_tradingview_data(csv_path)
    print(f"Loaded {len(tv_df)} weekly bars from TradingView")
    print(f"Date range: {tv_df.index[0]} to {tv_df.index[-1]}")
    print()
    
    # Analyze the problematic date from the log (June 2019)
    # The log shows Week 75, which should be around June 2019
    target_date = datetime.date(2019, 6, 14)
    analyze_weekly_bar_differences(tv_df, target_date)
    
    print()
    print("=" * 100)
    print("To diagnose further, you could:")
    print("=" * 100)
    print("1. Export the 15m data for the week containing 2019-06-14")
    print("2. Manually check which 15m bars are included in that week")
    print("3. Verify the Friday close time matches TradingView")
    print("4. Check if there are any missing 15m bars that could affect aggregation")

if __name__ == '__main__':
    main()

