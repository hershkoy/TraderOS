#!/usr/bin/env python3
"""
Example script demonstrating the MeanReversionRSI_BT strategy with 4h resampling
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import get_strategy
import backtrader as bt
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample 1h data for demonstration"""
    dates = pd.date_range('2023-01-01', periods=500, freq='1H')
    
    # Create realistic price data with some trends
    np.random.seed(42)  # For reproducible results
    
    # Base trend
    trend = np.sin(np.linspace(0, 4*np.pi, 500)) * 10
    
    # Random walk
    random_walk = np.cumsum(np.random.randn(500) * 0.5)
    
    # Combine for realistic prices
    base_price = 100 + trend + random_walk
    
    data = pd.DataFrame({
        'open': base_price + np.random.randn(500) * 0.2,
        'high': base_price + abs(np.random.randn(500) * 0.5),
        'low': base_price - abs(np.random.randn(500) * 0.5),
        'close': base_price + np.random.randn(500) * 0.2,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.randn(500) * 0.3)
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.randn(500) * 0.3)
    
    return data

def run_strategy_example():
    """Run the strategy with sample data"""
    print("Running MeanReversionRSI_BT Strategy Example")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample 1h data...")
    df_data = create_sample_data()
    print(f"Created {len(df_data)} data points from {df_data.index[0]} to {df_data.index[-1]}")
    
    # Setup Cerebro
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # Get strategy class
    strategy_class = get_strategy('mean_reversion_rsi')
    print(f"Strategy: {strategy_class.get_description()}")
    print(f"Data requirements: {strategy_class.get_data_requirements()}")
    
    # Check if strategy requests 4h data
    from backtrader_runner_yaml import _strategy_requests_4h
    needs_4h = _strategy_requests_4h(strategy_class)
    print(f"Strategy requests 4h data: {needs_4h}")
    
    # Setup data feeds
    if needs_4h:
        print("Setting up 1h + 4h data feeds...")
        # Base 1h feed
        data_1h = bt.feeds.PandasData(dataname=df_data)
        cerebro.adddata(data_1h)
        
        # 4h resample
        data_4h = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Minutes, compression=240)
        print(f"Created 1h feed: {type(data_1h).__name__}")
        print(f"Created 4h resample: {type(data_4h).__name__}")
        
        data_feeds = [data_1h, data_4h]
    else:
        print("Setting up single 1h data feed...")
        data_feed = bt.feeds.PandasData(dataname=df_data)
        cerebro.adddata(data_feed)
        data_feeds = [data_feed]
    
    # Add strategy
    strategy_params = {
        'rsi_period': 14,
        'adx_period': 14,
        'atr_period': 14,
        'bb_period': 20,
        'bb_dev': 2,
        'higher_tf_idx': 1
    }
    
    cerebro.addstrategy(strategy_class, **strategy_params)
    print(f"Added strategy with parameters: {strategy_params}")
    
    # Run backtest
    print("\nRunning backtest...")
    initial_value = cerebro.broker.getvalue()
    print(f"Initial portfolio value: ${initial_value:,.2f}")
    
    try:
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        print(f"Final portfolio value: ${final_value:,.2f}")
        print(f"Total return: {((final_value - initial_value) / initial_value * 100):.2f}%")
        
        # Get strategy instance
        strategy = results[0]
        
        # Print some statistics
        if hasattr(strategy, '_trades'):
            print(f"Number of trades executed: {len(strategy._trades)}")
        
        print("\nBacktest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("MeanReversionRSI_BT Strategy Example")
    print("This example demonstrates the 4h resampling functionality")
    print("=" * 60)
    
    success = run_strategy_example()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("The strategy successfully used both 1h and 4h timeframes")
    else:
        print("\n❌ Example failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
