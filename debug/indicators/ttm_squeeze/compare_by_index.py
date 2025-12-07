#!/usr/bin/env python3
"""
Compare Backtrader and pandas by index position to find the calculation difference.
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
    df['tradingview_momentum'] = df.iloc[:, 5]
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return df[['open', 'high', 'low', 'close', 'volume', 'tradingview_momentum']]

class TestStrategy(bt.Strategy):
    def __init__(self):
        from indicators.ttm_squeeze import TTMSqueezeMomentum
        self.mom = TTMSqueezeMomentum(self.data, lengthKC=20)
        self.values = []
        self.dates = []
        
    def next(self):
        if len(self.mom) > 0:
            self.values.append(float(self.mom.momentum[0]))
            self.dates.append(self.data.datetime.date(0))

def main():
    csv_path = r'c:\Users\Hezi\Downloads\BATS_HUBB, 1W_db6f3.csv'
    
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    tv_df = load_tradingview_data(csv_path)
    ohlcv = tv_df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Pandas calculation
    pandas_momentum = calculate_squeeze_momentum(ohlcv, lengthKC=20, use_logging=False)
    
    # Backtrader calculation
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(
        dataname=ohlcv,
        datetime=None,
        open=0, high=1, low=2, close=3, volume=4, openinterest=-1
    )
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    results = cerebro.run()
    strategy = results[0]
    
    # Compare by index (accounting for Backtrader warmup)
    print("Comparing by index position (accounting for warmup):")
    print("=" * 100)
    print(f"{'Index':<8} {'Pandas Date':<12} {'BT Date':<12} {'TV':>12} {'Pandas':>12} {'Backtrader':>12} {'Diff':>12}")
    print("-" * 100)
    
    # Backtrader starts after 20 bars warmup, so BT index 0 = pandas index 20
    warmup = 20
    min_len = min(len(pandas_momentum) - warmup, len(strategy.values))
    
    for i in range(min(50, min_len)):  # First 50 comparisons
        pandas_idx = warmup + i
        bt_idx = i
        
        if pandas_idx < len(pandas_momentum) and bt_idx < len(strategy.values):
            pandas_val = pandas_momentum.iloc[pandas_idx]
            bt_val = strategy.values[bt_idx]
            pandas_date = pandas_momentum.index[pandas_idx]
            bt_date = strategy.dates[bt_idx] if bt_idx < len(strategy.dates) else None
            
            # Find TV value for pandas date
            tv_val = None
            if pandas_date in tv_df.index:
                tv_val = tv_df.loc[pandas_date, 'tradingview_momentum']
            
            diff = abs(pandas_val - bt_val)
            
            tv_str = f"{tv_val:>12.6f}" if tv_val is not None else "N/A".rjust(12)
            bt_date_str = str(bt_date) if bt_date else "N/A".rjust(12)
            
            print(f"{i:>8} {str(pandas_date.date()):<12} {bt_date_str:<12} {tv_str} {pandas_val:>12.6f} {bt_val:>12.6f} {diff:>12.6f}")
            
            if diff > 0.01:
                print(f"  ^^^ DIVERGENCE DETECTED ^^^")

if __name__ == '__main__':
    main()

