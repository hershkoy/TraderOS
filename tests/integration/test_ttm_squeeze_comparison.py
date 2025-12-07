#!/usr/bin/env python3
"""
Comprehensive test to compare Backtrader TTMSqueezeMomentum with pandas calculation.
This will help us identify where the discrepancy occurs.
"""

import pandas as pd
import numpy as np
import datetime
import backtrader as bt
from pathlib import Path
import sys

# Import our implementations
from indicators.ttm_squeeze import TTMSqueezeMomentum, calculate_squeeze_momentum

class TestStrategy(bt.Strategy):
    """Simple strategy to capture indicator values."""
    
    def __init__(self):
        self.mom = TTMSqueezeMomentum(self.data, lengthKC=20)
        self.values = []
        self.dates = []
    
    def next(self):
        if len(self.mom) > 0:
            self.values.append(float(self.mom.momentum[0]))
            self.dates.append(self.data.datetime.date(0))

def load_tradingview_data(csv_path):
    """Load TradingView CSV data."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('date')
    df['tradingview_momentum'] = df.iloc[:, 5]  # 6th column
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return df[['open', 'high', 'low', 'close', 'volume', 'tradingview_momentum']]

def test_backtrader_indicator(df, lengthKC=20):
    """Test Backtrader indicator on the data."""
    cerebro = bt.Cerebro()
    
    # Convert DataFrame to Backtrader format
    # Backtrader expects datetime as a column, not index
    bt_df = df.reset_index()
    bt_df.rename(columns={'date': 'datetime'}, inplace=True)
    
    data = bt.feeds.PandasData(
        dataname=bt_df,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    
    # Run and get strategy instance
    results = cerebro.run()
    strategy = results[0]
    
    # Create results DataFrame
    if len(strategy.dates) > 0:
        results = pd.DataFrame({
            'date': strategy.dates,
            'momentum': strategy.values
        })
        results['date'] = pd.to_datetime(results['date'])
        results = results.set_index('date')
        return results
    return pd.DataFrame()

def compare_calculations(df, lengthKC=20, target_date=None):
    """Compare pandas vs Backtrader vs TradingView."""
    print("=" * 100)
    print("TTM Squeeze Calculation Comparison")
    print("=" * 100)
    print()
    
    # Calculate using pandas
    ohlcv = df[['open', 'high', 'low', 'close', 'volume']].copy()
    pandas_momentum = calculate_squeeze_momentum(ohlcv, lengthKC=lengthKC, use_logging=False)
    
    # Calculate using Backtrader
    print("Running Backtrader indicator...")
    backtrader_results = test_backtrader_indicator(ohlcv, lengthKC=lengthKC)
    
    print(f"Pandas calculated {len(pandas_momentum)} values")
    print(f"Backtrader calculated {len(backtrader_results)} values")
    print()
    
    # Align dates - normalize to date only for comparison
    pandas_dates = pd.to_datetime(pandas_momentum.index).normalize()
    bt_dates = pd.to_datetime(backtrader_results.index).normalize()
    
    # Create aligned series
    pandas_aligned = pd.Series(pandas_momentum.values, index=pandas_dates)
    bt_aligned = pd.Series(backtrader_results['momentum'].values, index=bt_dates)
    
    # Find overlapping dates
    common_dates = pandas_aligned.index.intersection(bt_aligned.index)
    if len(common_dates) == 0:
        print("ERROR: No overlapping dates between pandas and Backtrader results!")
        print(f"Pandas dates: {pandas_aligned.index[0]} to {pandas_aligned.index[-1]}")
        if len(bt_aligned) > 0:
            print(f"Backtrader dates: {bt_aligned.index[0]} to {bt_aligned.index[-1]}")
        return
    
    print(f"Found {len(common_dates)} overlapping dates")
    print()
    
    # Compare around target date or show last 20
    if target_date:
        start_date = pd.Timestamp(target_date) - pd.Timedelta(days=14)
        end_date = pd.Timestamp(target_date) + pd.Timedelta(days=14)
        mask = (common_dates >= start_date) & (common_dates <= end_date)
        compare_dates = common_dates[mask]
    else:
        compare_dates = common_dates[-20:]
    
    print("=" * 100)
    print(f"Comparison (showing {len(compare_dates)} dates):")
    print("=" * 100)
    print(f"{'Date':<12} {'TradingView':>15} {'Pandas':>15} {'Backtrader':>15} {'Pandas Diff':>15} {'BT Diff':>15}")
    print("-" * 100)
    
    max_pandas_diff = 0
    max_bt_diff = 0
    
    for date in compare_dates:
        # Find closest TradingView date
        tv_date = df.index[df.index.get_indexer([date], method='nearest')[0]] if len(df) > 0 else None
        tv_val = df.loc[tv_date, 'tradingview_momentum'] if tv_date is not None and abs((pd.to_datetime(tv_date).normalize() - date).days) <= 1 else None
        
        pandas_val = pandas_aligned.loc[date]
        bt_val = bt_aligned.loc[date] if date in bt_aligned.index else None
        
        pandas_diff = abs(pandas_val - tv_val) if tv_val is not None else None
        bt_diff = abs(bt_val - tv_val) if (tv_val is not None and bt_val is not None) else None
        
        if pandas_diff:
            max_pandas_diff = max(max_pandas_diff, pandas_diff)
        if bt_diff:
            max_bt_diff = max(max_bt_diff, bt_diff)
        
        tv_str = f"{tv_val:>15.6f}" if tv_val is not None else "N/A".rjust(15)
        bt_str = f"{bt_val:>15.6f}" if bt_val is not None else "N/A".rjust(15)
        pandas_diff_str = f"{pandas_diff:>15.6f}" if pandas_diff is not None else "N/A".rjust(15)
        bt_diff_str = f"{bt_diff:>15.6f}" if bt_diff is not None else "N/A".rjust(15)
        
        print(f"{date.date()} {tv_str} {pandas_val:>15.6f} {bt_str} {pandas_diff_str} {bt_diff_str}")
    
    print()
    print("=" * 100)
    print(f"Summary:")
    print(f"  Max pandas difference from TradingView: {max_pandas_diff:.6f}")
    print(f"  Max Backtrader difference from TradingView: {max_bt_diff:.6f}")
    print("=" * 100)
    
    # Show detailed comparison for a specific problematic date
    if target_date:
        target_ts = pd.Timestamp(target_date)
        if target_ts in compare_dates:
            print()
            print("=" * 100)
            print(f"Detailed analysis for {target_date}:")
            print("=" * 100)
            # Find closest dates
            tv_date = df.index[df.index.get_indexer([target_ts], method='nearest')[0]] if len(df) > 0 else None
            tv_val = df.loc[tv_date, 'tradingview_momentum'] if tv_date is not None else None
            
            target_normalized = pd.to_datetime(target_ts).normalize()
            pandas_val = pandas_aligned.loc[target_normalized] if target_normalized in pandas_aligned.index else None
            bt_val = bt_aligned.loc[target_normalized] if target_normalized in bt_aligned.index else None
            
            print(f"TradingView: {tv_val:.6f if tv_val else 'N/A'}")
            print(f"Pandas:      {pandas_val:.6f}")
            print(f"Backtrader:  {bt_val:.6f if bt_val else 'N/A'}")
            if tv_val:
                print(f"Pandas diff: {abs(pandas_val - tv_val):.6f}")
                if bt_val:
                    print(f"BT diff:     {abs(bt_val - tv_val):.6f}")

def main():
    csv_path = r'c:\Users\Hezi\Downloads\BATS_HUBB, 1W_db6f3.csv'
    
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    # Load TradingView data
    tv_df = load_tradingview_data(csv_path)
    print(f"Loaded {len(tv_df)} rows from TradingView CSV")
    print(f"Date range: {tv_df.index[0]} to {tv_df.index[-1]}")
    print()
    
    # Compare around the problematic date
    target_date = datetime.date(2021, 11, 19)
    compare_calculations(tv_df, lengthKC=20, target_date=target_date)

if __name__ == '__main__':
    main()

