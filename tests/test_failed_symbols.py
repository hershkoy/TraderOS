#!/usr/bin/env python3
"""
Test script for failed symbols identification
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.identify_failed_symbols import identify_failed_symbols, save_failed_symbols_to_file
except ImportError:
    print("Failed to import identify_failed_symbols module")
    sys.exit(1)

def test_failed_symbols():
    """Test the failed symbols identification"""
    print("Testing failed symbols identification...")
    
    # Test with IB provider and 1h timeframe
    failed_symbols = identify_failed_symbols("ib", "1h")
    
    if failed_symbols:
        print(f"Found {len(failed_symbols)} failed symbols")
        print(f"First 10: {failed_symbols[:10]}")
        
        # Save to file
        if save_failed_symbols_to_file(failed_symbols, "test_failed.txt"):
            print("[SUCCESS] Successfully saved failed symbols to test_failed.txt")
        else:
            print("[ERROR] Failed to save failed symbols")
    else:
        print("[SUCCESS] No failed symbols found")

if __name__ == "__main__":
    test_failed_symbols()
