"""
Test script for the charting server components
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_aggregator import DataAggregator
from indicators import SMA, EMA, RSI, MACD, BollingerBands
import pandas as pd

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    # Get available symbols
    symbols = DataAggregator.get_available_symbols()
    print(f"Available symbols: {symbols}")
    
    if not symbols:
        print("No symbols found in data folder")
        return False
    
    # Test loading data for first symbol
    symbol = symbols[0]
    print(f"Testing with symbol: {symbol}")
    
    # Get available timeframes
    timeframes = DataAggregator.get_available_timeframes(symbol)
    print(f"Available timeframes for {symbol}: {timeframes}")
    
    if not timeframes:
        print("No timeframes found for symbol")
        return False
    
    # Test loading data
    timeframe = timeframes[0]
    df = DataAggregator.get_data(symbol, timeframe)
    
    if df is None:
        print(f"Failed to load data for {symbol} at {timeframe}")
        return False
    
    print(f"Successfully loaded data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return True

def test_indicators():
    """Test indicator calculations"""
    print("\nTesting indicators...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'open': [100 + i * 0.1 + (i % 10) * 0.5 for i in range(100)],
        'high': [100 + i * 0.1 + (i % 10) * 0.5 + 2 for i in range(100)],
        'low': [100 + i * 0.1 + (i % 10) * 0.5 - 1 for i in range(100)],
        'close': [100 + i * 0.1 + (i % 10) * 0.5 + 0.5 for i in range(100)],
        'volume': [1000000 + i * 1000 for i in range(100)]
    }, index=dates)
    
    print(f"Sample data shape: {df.shape}")
    
    # Test SMA
    sma = SMA(df['close'], 20)
    print(f"SMA(20) calculated: {len(sma.dropna())} values")
    
    # Test EMA
    ema = EMA(df['close'], 20)
    print(f"EMA(20) calculated: {len(ema.dropna())} values")
    
    # Test RSI
    rsi = RSI(df['close'], 14)
    print(f"RSI(14) calculated: {len(rsi.dropna())} values")
    
    # Test MACD
    macd_result = MACD(df['close'])
    print(f"MACD calculated: {len(macd_result['macd'].dropna())} values")
    
    # Test Bollinger Bands
    bb_result = BollingerBands(df['close'])
    print(f"Bollinger Bands calculated: {len(bb_result['upper'].dropna())} values")
    
    return True

def main():
    """Run all tests"""
    print("=== Charting Server Test ===\n")
    
    # Test data loading
    data_ok = test_data_loading()
    
    # Test indicators
    indicators_ok = test_indicators()
    
    print("\n=== Test Results ===")
    print(f"Data loading: {'✓ PASS' if data_ok else '✗ FAIL'}")
    print(f"Indicators: {'✓ PASS' if indicators_ok else '✗ FAIL'}")
    
    if data_ok and indicators_ok:
        print("\n🎉 All tests passed! The charting server should work correctly.")
        print("\nTo start the server, run:")
        print("python charting_server.py")
        print("\nThen open http://localhost:5000 in your browser.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
