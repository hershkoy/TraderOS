#!/usr/bin/env python3
"""
Test scanner on a small subset of tickers
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanner_runner import ScannerRunner

def test_scanner_small():
    """Test scanner on a small subset of tickers"""
    
    # Test with just a few well-known tickers
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("Testing Scanner on Small Subset")
    print("=" * 40)
    print(f"Testing with symbols: {', '.join(test_symbols)}")
    print()
    
    # Initialize scanner runner
    runner = ScannerRunner()
    
    # Run multi-scanner on small subset
    print("Running HL After LL and Squeeze scanners...")
    runner.run_multi(['hl_after_ll', 'squeeze'], test_symbols, skip_auto_update=True)
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_scanner_small()
