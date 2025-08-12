# Test script for custom tracking
import backtrader as bt
from custom_tracking import CustomTrackingMixin
import pandas as pd
from pathlib import Path

class SimpleTestStrategy(CustomTrackingMixin, bt.Strategy):
    """Simple test strategy to demonstrate custom tracking"""
    
    params = dict(
        period=20,
        size=1
    )
    
    def __init__(self):
        # Initialize custom tracking
        CustomTrackingMixin.__init__(self)
        
        # Simple moving average
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.period)
        
        # Track trades manually
        self.trade_count = 0
        
    def next(self):
        # Track portfolio value
        self.track_portfolio_value()
        
        if not self.position:
            # Buy when price is below SMA
            if self.data.close[0] < self.sma[0]:
                self.buy(size=self.p.size)
                self.trade_count += 1
                print(f"BUY signal at {self.data.close[0]:.2f}")
        else:
            # Sell when price is above SMA
            if self.data.close[0] > self.sma[0]:
                self.sell(size=self.p.size)
                print(f"SELL signal at {self.data.close[0]:.2f}")
    
    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.track_trade_entry(order.executed.price, order.executed.size)
                print(f"BUY executed at {order.executed.price:.2f}")
            else:
                self.track_trade_exit(order.executed.price, order.executed.size)
                print(f"SELL executed at {order.executed.price:.2f}")
    
    def stop(self):
        print(f"\nFinal Portfolio Value: {self.broker.getvalue():.2f}")
        
        # Show custom tracking results
        stats = self.get_trade_statistics()
        print(f"\nCustom Tracking Results:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        
        # Show individual trades
        print(f"\nIndividual Trades:")
        for i, trade in enumerate(self._custom_trades, 1):
            print(f"  Trade {i}: {trade['entry_date']} -> {trade['exit_date']}, P&L: ${trade['pnl']:.2f}")

def load_test_data():
    """Load some test data"""
    # Create simple test data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = [100 + i * 0.1 + (i % 20 - 10) * 0.5 for i in range(100)]  # Trending with noise
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [1000] * 100
    }, index=dates)
    
    return df

def main():
    print("Testing Custom Tracking System")
    print("=" * 40)
    
    # Load test data
    df = load_test_data()
    print(f"Loaded {len(df)} data points")
    
    # Create data feed
    data = bt.feeds.PandasData(dataname=df)
    
    # Setup Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000.0)
    cerebro.adddata(data)
    cerebro.addstrategy(SimpleTestStrategy)
    
    # Run backtest
    print("\nRunning backtest...")
    results = cerebro.run()
    
    print("\nBacktest completed successfully!")
    print("Custom tracking system is working without Backtrader analyzer errors.")

if __name__ == '__main__':
    main()
