#!/usr/bin/env python3

print("Starting simple test...")

try:
    # Test basic functionality
    from pathlib import Path
    import pandas as pd
    import backtrader as bt
    
    print("Imports successful")
    
    # Test minimal backtest
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    
    print("Cerebro created")
    
    # Create minimal data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'open': [100] * 10,
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [102] * 10,
        'volume': [1000] * 10,
    }, index=dates)
    
    print("Data created")
    
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    
    print("Data feed added")
    
    # Add simple strategy
    class SimpleStrategy(bt.Strategy):
        def next(self):
            pass
    
    cerebro.addstrategy(SimpleStrategy)
    
    print("Strategy added")
    
    # Run backtest
    results = cerebro.run()
    
    print("Backtest completed successfully!")
    print(f"Final portfolio value: {cerebro.broker.getvalue()}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
