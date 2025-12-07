#!/usr/bin/env python3
"""
Trace the exact calculation for one bar to see where Backtrader diverges from pandas.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from indicators.ttm_squeeze import calculate_squeeze_momentum

def load_tradingview_data(csv_path):
    """Load TradingView CSV data."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('date')
    df['tradingview_momentum'] = df.iloc[:, 5]
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return df[['open', 'high', 'low', 'close', 'volume', 'tradingview_momentum']]

def manual_calculate_for_bar(df, bar_idx, lengthKC):
    """Manually calculate squeeze momentum for a specific bar using pandas logic."""
    # Get the window of bars we need (L bars ending at bar_idx)
    start_idx = max(0, bar_idx - lengthKC + 1)
    end_idx = bar_idx + 1
    
    # Calculate baseline components for bar at bar_idx
    window = df.iloc[start_idx:end_idx]
    highest_h = window['high'].max()
    lowest_l = window['low'].min()
    mid_hl = (highest_h + lowest_l) / 2.0
    sma_close = window['close'].mean()
    baseline = (mid_hl + sma_close) / 2.0
    difference = df.iloc[bar_idx]['close'] - baseline
    
    return {
        'bar_idx': bar_idx,
        'date': df.index[bar_idx],
        'close': df.iloc[bar_idx]['close'],
        'baseline': baseline,
        'difference': difference,
        'window_start': start_idx,
        'window_end': end_idx,
        'window_size': end_idx - start_idx
    }

def calculate_differences_series(df, lengthKC):
    """Calculate the differences series (close - baseline) for all bars."""
    differences = []
    dates = []
    
    for i in range(lengthKC - 1, len(df)):
        info = manual_calculate_for_bar(df, i, lengthKC)
        differences.append(info['difference'])
        dates.append(info['date'])
    
    return pd.Series(differences, index=dates)

def main():
    csv_path = r'c:\Users\Hezi\Downloads\BATS_HUBB, 1W_db6f3.csv'
    
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    tv_df = load_tradingview_data(csv_path)
    ohlcv = tv_df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Calculate using pandas function
    pandas_momentum = calculate_squeeze_momentum(ohlcv, lengthKC=20, use_logging=False)
    
    # Manually calculate differences series
    differences_series = calculate_differences_series(ohlcv, 20)
    
    # Find a specific bar to trace
    target_date = pd.Timestamp(2021, 11, 15)
    if target_date in ohlcv.index:
        bar_idx = ohlcv.index.get_loc(target_date)
        
        print(f"Tracing calculation for bar at {target_date.date()} (index {bar_idx})")
        print("=" * 80)
        
        # Show the differences for the last 20 bars ending at this bar
        start_idx = max(0, bar_idx - 19)
        print(f"\nDifferences for last 20 bars ending at bar {bar_idx}:")
        print(f"Bar indices: {start_idx} to {bar_idx}")
        print()
        
        for i in range(start_idx, bar_idx + 1):
            info = manual_calculate_for_bar(ohlcv, i, 20)
            print(f"Bar {i} ({info['date'].date()}): close={info['close']:.2f}, "
                  f"baseline={info['baseline']:.6f}, diff={info['difference']:.6f}")
        
        # Show the final momentum value
        if target_date in pandas_momentum.index:
            print(f"\nFinal momentum value: {pandas_momentum.loc[target_date]:.6f}")
            print(f"TradingView value: {tv_df.loc[target_date, 'tradingview_momentum']:.6f}")
        
        # Show differences series values
        if target_date in differences_series.index:
            print(f"\nDifference value for this bar: {differences_series.loc[target_date]:.6f}")
            
            # Show last 20 differences for linear regression
            reg_start = differences_series.index.get_loc(target_date) - 19
            if reg_start >= 0:
                reg_window = differences_series.iloc[reg_start:reg_start+20]
                print(f"\nLast 20 differences for linear regression:")
                for date, diff in reg_window.items():
                    print(f"  {date.date()}: {diff:.6f}")

if __name__ == '__main__':
    main()

