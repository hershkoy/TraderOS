#!/usr/bin/env python3
"""
Debug script to identify the None formatting issue
"""

import sys
sys.path.append('.')

from backtrader_runner_yaml import load_config, get_strategy, load_daily_data
from pathlib import Path
import pandas as pd
import backtrader as bt

def test_report_generation():
    """Test report generation with debug output"""
    
    # Load minimal config
    config = load_config('defaults.yaml')
    
    # Setup minimal backtest
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    
    # Load minimal data
    df_data = load_daily_data(Path('data/ALPACA/NFLX/1h/nflx_1h.parquet'))
    df_data = df_data.iloc[:30]  # Use only 30 days to avoid issues
    
    data_feed = bt.feeds.PandasData(dataname=df_data)
    cerebro.adddata(data_feed)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
    
    # Add simple buy and hold strategy for testing
    class TestStrategy(bt.Strategy):
        def next(self):
            if not self.position and len(self.data) == 10:
                self.buy()
            elif self.position and len(self.data) == 20:
                self.sell()
    
    cerebro.addstrategy(TestStrategy)
    
    try:
        results = cerebro.run()
        strategy = results[0]
        
        print("=== ANALYZER DEBUG ===")
        
        # Test each analyzer individually
        try:
            trade_analyzer = strategy.analyzers.trades.get_analysis()
            print(f"trade_analyzer: {trade_analyzer}")
            print(f"trade_analyzer type: {type(trade_analyzer)}")
        except Exception as e:
            print(f"trade_analyzer error: {e}")
            
        try:
            sharpe = strategy.analyzers.sharpe.get_analysis()
            print(f"sharpe: {sharpe}")
            print(f"sharpe type: {type(sharpe)}")
        except Exception as e:
            print(f"sharpe error: {e}")
            
        try:
            drawdown = strategy.analyzers.drawdown.get_analysis()
            print(f"drawdown: {drawdown}")
            print(f"drawdown type: {type(drawdown)}")
        except Exception as e:
            print(f"drawdown error: {e}")
            
        # Test specific values
        print("\\n=== VALUE DEBUG ===")
        trade_analyzer = strategy.analyzers.trades.get_analysis() or {}
        
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        print(f"total_trades: {total_trades} (type: {type(total_trades)})")
        
        winning_trades = trade_analyzer.get('won', {}).get('total', 0)
        print(f"winning_trades: {winning_trades} (type: {type(winning_trades)})")
        
        gross_profit = trade_analyzer.get('won', {}).get('pnl', {}).get('total', 0) or 0
        print(f"gross_profit: {gross_profit} (type: {type(gross_profit)})")
        
        # Test the specific problematic formatting
        try:
            test_format = f"${gross_profit:.2f}"
            print(f"Format test successful: {test_format}")
        except Exception as e:
            print(f"Format test failed: {e}")
            
        print("\\nAll analyzer tests completed successfully!")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_report_generation()
