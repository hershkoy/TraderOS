#!/usr/bin/env python3
"""
Test script to demonstrate TradingView-style report generation
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import backtrader as bt

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.tradingview_report_generator import generate_tradingview_report

def create_mock_data():
    """Create mock data for testing"""
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'open': [100 + i * 0.1 + (i % 10) * 0.5 for i in range(len(dates))],
        'high': [100 + i * 0.1 + (i % 10) * 0.5 + 2 for i in range(len(dates))],
        'low': [100 + i * 0.1 + (i % 10) * 0.5 - 2 for i in range(len(dates))],
        'close': [100 + i * 0.1 + (i % 10) * 0.5 + 1 for i in range(len(dates))],
        'volume': [1000000 + i * 1000 for i in range(len(dates))]
    }
    
    df = pd.DataFrame(data, index=dates)
    return df

def create_mock_strategy():
    """Create a mock strategy with sample trades"""
    class MockStrategy:
        def __init__(self):
            self._custom_trades = [
                {
                    'type': 'Long',
                    'entry_date': '2023-02-15',
                    'exit_date': '2023-03-15',
                    'entry_price': 105.50,
                    'exit_price': 112.30,
                    'size': 1,
                    'pnl': 6.80,
                    'pnl_percent': 6.45,
                    'signal': 'BBandLE',
                    'status': 'Closed'
                },
                {
                    'type': 'Short',
                    'entry_date': '2023-04-10',
                    'exit_date': '2023-05-10',
                    'entry_price': 118.20,
                    'exit_price': 110.50,
                    'size': 1,
                    'pnl': 7.70,
                    'pnl_percent': 6.51,
                    'signal': 'BBandSE',
                    'status': 'Closed'
                },
                {
                    'type': 'Long',
                    'entry_date': '2023-06-20',
                    'exit_date': '2023-07-20',
                    'entry_price': 108.90,
                    'exit_price': 115.60,
                    'size': 1,
                    'pnl': 6.70,
                    'pnl_percent': 6.15,
                    'signal': 'BBandLE',
                    'status': 'Closed'
                }
            ]
    
    return MockStrategy()

def create_mock_cerebro():
    """Create a mock cerebro with sample data"""
    class MockCerebro:
        def __init__(self):
            self.broker = MockBroker()
    
    class MockBroker:
        def getvalue(self):
            return 110000.0  # Final portfolio value
    
    return MockCerebro()

def test_tradingview_report():
    """Test the TradingView report generation"""
    print("Testing TradingView-style report generation...")
    
    # Create mock data
    strategy = create_mock_strategy()
    cerebro = create_mock_cerebro()
    data_df = create_mock_data()
    
    # Create mock config
    config = {
        'global': {
            'cash': 100000.0,
            'fromdate': '2023-01-01',
            'todate': '2023-12-31'
        },
        'strategies': {
            'mean_reversion': {
                'parameters': {
                    'period': 20,
                    'devfactor': 2.0,
                    'commission': 0.001
                }
            }
        }
    }
    
    # Create report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports") / f"test_tradingview_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate the TradingView report
        report_path = generate_tradingview_report(
            strategy=strategy,
            cerebro=cerebro,
            config=config,
            report_dir=report_dir,
            strategy_name="mean_reversion",
            data_df=data_df
        )
        
        print(f"✅ TradingView report generated successfully!")
        print(f"📁 Report location: {report_path}")
        print(f"🌐 Open {report_path} in your browser to view the report")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tradingview_report()
    if success:
        print("\n🎉 Test completed successfully!")
        print("You can now run your actual backtests and they will automatically generate TradingView-style reports.")
    else:
        print("\n💥 Test failed. Please check the error messages above.")
