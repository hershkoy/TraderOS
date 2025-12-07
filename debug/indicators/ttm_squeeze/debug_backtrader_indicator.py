#!/usr/bin/env python3
"""
Debug script to compare Backtrader indicator with pandas calculation.
This will help identify where the Backtrader implementation diverges.
"""

import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import backtrader as bt
from indicators.ttm_squeeze import calculate_squeeze_momentum

def load_tradingview_data(csv_path):
    """Load TradingView CSV data."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('date')
    df['tradingview_momentum'] = df.iloc[:, 5]  # 6th column
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return df[['open', 'high', 'low', 'close', 'volume', 'tradingview_momentum']]

class TestStrategy(bt.Strategy):
    """Simple strategy to capture indicator values."""
    
    def __init__(self):
        from indicators.ttm_squeeze import TTMSqueezeMomentum
        self.mom = TTMSqueezeMomentum(self.data, lengthKC=20)
        self.values = []
        
    def next(self):
        if len(self.mom) > 0:
            self.values.append({
                'date': self.data.datetime.date(0),
                'momentum': float(self.mom.momentum[0]),
                'slope': float(self.mom.slope[0])
            })

def test_backtrader_on_data(df, lengthKC=20):
    """Test Backtrader indicator on the data."""
    cerebro = bt.Cerebro()
    
    # Convert DataFrame to Backtrader format
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Use index
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=-1
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    
    # Run and get strategy instance
    results = cerebro.run()
    strategy = results[0]
    
    # Convert results to DataFrame
    if strategy.values:
        results_df = pd.DataFrame(strategy.values)
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df = results_df.set_index('date')
        return results_df
    return pd.DataFrame()

def manual_calculate_baseline(df, bar_idx, lengthKC):
    """Manually calculate baseline for a specific bar to debug."""
    # Get the L bars ending at bar_idx
    start_idx = max(0, bar_idx - lengthKC + 1)
    end_idx = bar_idx + 1
    
    window = df.iloc[start_idx:end_idx]
    
    highest_high = window['high'].max()
    lowest_low = window['low'].min()
    mid_hl = (highest_high + lowest_low) / 2.0
    sma_close = window['close'].mean()
    baseline = (mid_hl + sma_close) / 2.0
    
    return {
        'bar_idx': bar_idx,
        'date': df.index[bar_idx],
        'close': df.iloc[bar_idx]['close'],
        'highest_high': highest_high,
        'lowest_low': lowest_low,
        'mid_hl': mid_hl,
        'sma_close': sma_close,
        'baseline': baseline,
        'difference': df.iloc[bar_idx]['close'] - baseline
    }

def main():
    csv_path = r'c:\Users\Hezi\Downloads\BATS_HUBB, 1W_db6f3.csv'
    
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    # Load TradingView data
    tv_df = load_tradingview_data(csv_path)
    print(f"Loaded {len(tv_df)} rows")
    print()
    
    # Calculate using pandas
    ohlcv = tv_df[['open', 'high', 'low', 'close', 'volume']].copy()
    pandas_momentum = calculate_squeeze_momentum(ohlcv, lengthKC=20, use_logging=False)
    
    # Calculate using Backtrader
    print("Calculating with Backtrader indicator...")
    backtrader_results = test_backtrader_on_data(ohlcv, lengthKC=20)
    
    if len(backtrader_results) == 0:
        print("ERROR: Backtrader indicator produced no results!")
        return
    
    print(f"Backtrader produced {len(backtrader_results)} values")
    print(f"Backtrader date range: {backtrader_results.index[0]} to {backtrader_results.index[-1]}")
    print(f"Pandas date range: {pandas_momentum.index[0]} to {pandas_momentum.index[-1]}")
    print()
    print("First 10 Backtrader dates:")
    for date in backtrader_results.index[:10]:
        print(f"  {date}")
    print()
    print("First 10 Pandas dates:")
    for date in pandas_momentum.index[:10]:
        print(f"  {date}")
    print()
    
    # Compare around the problematic date
    target_date = datetime.date(2021, 11, 19)
    start_date = pd.Timestamp(2021, 11, 1)
    end_date = pd.Timestamp(2021, 11, 30)
    
    print("=" * 100)
    print(f"Comparison around {target_date}:")
    print("=" * 100)
    print(f"{'Date':<12} {'TV':>12} {'Pandas':>12} {'Backtrader':>12} {'Pandas Diff':>12} {'BT Diff':>12}")
    print("-" * 100)
    
    for date in pandas_momentum.index:
        if start_date <= date <= end_date:
            tv_val = tv_df.loc[date, 'tradingview_momentum'] if date in tv_df.index else None
            pandas_val = pandas_momentum.loc[date]
            bt_val = backtrader_results.loc[date, 'momentum'] if date in backtrader_results.index else None
            
            pandas_diff = abs(pandas_val - tv_val) if tv_val is not None else None
            bt_diff = abs(bt_val - tv_val) if (tv_val is not None and bt_val is not None) else None
            
            tv_str = f"{tv_val:>12.6f}" if tv_val is not None else "N/A".rjust(12)
            bt_str = f"{bt_val:>12.6f}" if bt_val is not None else "N/A".rjust(12)
            pandas_diff_str = f"{pandas_diff:>12.6f}" if pandas_diff is not None else "N/A".rjust(12)
            bt_diff_str = f"{bt_diff:>12.6f}" if bt_diff is not None else "N/A".rjust(12)
            
            print(f"{date.date()} {tv_str} {pandas_val:>12.6f} {bt_str} {pandas_diff_str} {bt_diff_str}")
    
    # Find matching dates and compare
    print()
    print("=" * 100)
    print("Comparing values at matching dates:")
    print("=" * 100)
    
    common_dates = pandas_momentum.index.intersection(backtrader_results.index)
    print(f"Found {len(common_dates)} matching dates")
    print()
    
    if len(common_dates) > 0:
        print(f"{'Date':<12} {'TV':>12} {'Pandas':>12} {'Backtrader':>12} {'Pandas Diff':>12} {'BT Diff':>12}")
        print("-" * 100)
        
        for date in common_dates[:20]:  # First 20 matches
            tv_val = tv_df.loc[date, 'tradingview_momentum'] if date in tv_df.index else None
            pandas_val = pandas_momentum.loc[date]
            bt_val = backtrader_results.loc[date, 'momentum']
            
            pandas_diff = abs(pandas_val - tv_val) if tv_val is not None else None
            bt_diff = abs(bt_val - tv_val) if tv_val is not None else None
            
            tv_str = f"{tv_val:>12.6f}" if tv_val is not None else "N/A".rjust(12)
            pandas_diff_str = f"{pandas_diff:>12.6f}" if pandas_diff is not None else "N/A".rjust(12)
            bt_diff_str = f"{bt_diff:>12.6f}" if bt_diff is not None else "N/A".rjust(12)
            
            print(f"{date.date()} {tv_str} {pandas_val:>12.6f} {bt_val:>12.6f} {pandas_diff_str} {bt_diff_str}")
    
    # Find first divergence by comparing same dates
    print()
    print("=" * 100)
    print("Finding first significant divergence (same dates):")
    print("=" * 100)
    
    for date in sorted(common_dates)[20:]:  # Start after warmup
        pandas_val = pandas_momentum.loc[date]
        bt_val = backtrader_results.loc[date, 'momentum']
        diff = abs(pandas_val - bt_val)
        if diff > 0.01:  # Significant difference
            print(f"First divergence at {date.date()}:")
            print(f"  Pandas: {pandas_val:.6f}")
            print(f"  Backtrader: {bt_val:.6f}")
            print(f"  Difference: {diff:.6f}")
            break
    
    # Debug baseline calculation for a specific bar
    print()
    print("=" * 100)
    print("Debugging baseline calculation for bar at 2021-11-15 (should match TradingView):")
    print("=" * 100)
    
    target_idx = None
    for i, date in enumerate(ohlcv.index):
        if date.date() == datetime.date(2021, 11, 15):
            target_idx = i
            break
    
    if target_idx is not None and target_idx >= 20:
        # Calculate baseline for this bar and surrounding bars
        for offset in [-2, -1, 0, 1, 2]:
            idx = target_idx + offset
            if idx >= 20:
                debug_info = manual_calculate_baseline(ohlcv, idx, 20)
                print(f"Bar {idx} ({debug_info['date'].date()}):")
                print(f"  Close: {debug_info['close']:.2f}")
                print(f"  Baseline: {debug_info['baseline']:.6f}")
                print(f"  Difference: {debug_info['difference']:.6f}")
                print()

if __name__ == '__main__':
    main()

