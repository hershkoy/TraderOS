#!/usr/bin/env python3
"""
Test Auto-Update Functionality
Demonstrates the new automatic data update feature
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanner_runner import ScannerRunner

def test_auto_update():
    """Test the auto-update functionality"""
    print("🧪 Testing Auto-Update Functionality")
    print("=" * 50)
    
    # Test 1: Normal scan with auto-update (default)
    print("\n1. Testing normal scan with auto-update:")
    print("-" * 40)
    
    runner = ScannerRunner()
    
    # Test with a small set of symbols
    test_symbols = ["AAPL", "MSFT"]
    
    print(f"Testing with symbols: {test_symbols}")
    
    # Check data freshness
    freshness = runner.check_data_freshness(test_symbols)
    print(f"Data freshness check:")
    print(f"  Needs update: {freshness['needs_update']}")
    print(f"  Stale symbols: {len(freshness['stale_symbols'])}")
    print(f"  Missing symbols: {len(freshness['missing_symbols'])}")
    print(f"  Fresh symbols: {len(freshness['fresh_symbols'])}")
    
    # Test 2: Skip auto-update
    print("\n2. Testing skip auto-update:")
    print("-" * 40)
    
    print("This would skip auto-update and scan existing data only")
    print("Command: python scanner_runner.py --scanner hl_after_ll --skip-update")
    
    # Test 3: Force update
    print("\n3. Testing force update:")
    print("-" * 40)
    
    print("This would force update even if data appears fresh")
    print("Command: python scanner_runner.py --scanner hl_after_ll --force-update")
    
    print("\n✅ Auto-update functionality test completed!")
    print("\nTo run actual tests:")
    print("1. python scanner_runner.py --scanner hl_after_ll --symbols AAPL MSFT")
    print("2. python scanner_runner.py --scanner hl_after_ll --symbols AAPL MSFT --skip-update")
    print("3. python scanner_runner.py --scanner hl_after_ll --symbols AAPL MSFT --force-update")

def show_config_examples():
    """Show configuration examples for auto-update"""
    print("\n📋 Auto-Update Configuration Examples")
    print("=" * 50)
    
    print("\n1. Enable auto-update (default):")
    print("""
auto_update:
  enabled: true
  max_days_old: 1
  provider: "ALPACA"
  timeframe: "1d"
  days_back: 30
""")
    
    print("\n2. Disable auto-update:")
    print("""
auto_update:
  enabled: false
""")
    
    print("\n3. More aggressive updates:")
    print("""
auto_update:
  enabled: true
  max_days_old: 0        # Update even fresh data
  provider: "ALPACA"
  timeframe: "1d"
  days_back: 60         # Fetch more historical data
  batch_size: 5         # Smaller batches
  timeout: 120          # Longer timeout
""")
    
    print("\n4. Conservative updates:")
    print("""
auto_update:
  enabled: true
  max_days_old: 7       # Only update data older than 7 days
  provider: "ALPACA"
  timeframe: "1d"
  days_back: 14         # Fetch less historical data
  batch_size: 20        # Larger batches
  timeout: 30           # Shorter timeout
""")

def show_usage_examples():
    """Show usage examples"""
    print("\n🚀 Usage Examples")
    print("=" * 50)
    
    examples = [
        ("Basic scan with auto-update", "python scanner_runner.py --scanner hl_after_ll"),
        ("Scan specific symbols", "python scanner_runner.py --scanner hl_after_ll --symbols AAPL MSFT GOOGL"),
        ("Skip auto-update", "python scanner_runner.py --scanner hl_after_ll --skip-update"),
        ("Force update", "python scanner_runner.py --scanner hl_after_ll --force-update"),
        ("Debug mode", "python scanner_runner.py --scanner hl_after_ll --log-level DEBUG"),
        ("Custom config", "python scanner_runner.py --config my_config.yaml --scanner hl_after_ll"),
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")

def main():
    """Main function"""
    print("🔍 Scanner Auto-Update Test Suite")
    print("=" * 50)
    
    try:
        test_auto_update()
        show_config_examples()
        show_usage_examples()
        
        print(f"\n🎯 Test suite completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
