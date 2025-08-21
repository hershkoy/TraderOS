#!/usr/bin/env python3
"""
Test script to run a small universe update with comprehensive logging
"""

import sys
import os
from datetime import datetime

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_small_update():
    """Test a small universe update with logging"""
    try:
        from update_universe_data import UniverseDataUpdater
        
        print(f"[{datetime.now()}] Testing small universe update...")
        
        # Create updater for IB hourly data
        updater = UniverseDataUpdater(provider="ib", timeframe="1h")
        print(f"[{datetime.now()}] ✅ Updater created")
        
        # Run a very small update (just 3 tickers)
        print(f"[{datetime.now()}] 🔍 About to call update_universe_data...")
        
        results = updater.update_universe_data(
            batch_size=2,
            delay_between_batches=2.0,
            delay_between_tickers=1.0,
            max_tickers=3,  # Only process 3 tickers
            use_max_bars=False,  # Use fixed amounts for speed
            skip_existing=False  # Don't skip existing data
        )
        
        print(f"[{datetime.now()}] ✅ update_universe_data returned: {type(results)}")
        
        if "error" in results:
            print(f"[{datetime.now()}] ❌ Update failed: {results['error']}")
            return False
        else:
            print(f"[{datetime.now()}] 🎉 Update completed successfully!")
            print(f"[{datetime.now()}] Results: {results}")
            return True
            
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("=" * 60)
    print("SMALL UNIVERSE UPDATE TEST WITH LOGGING")
    print("=" * 60)
    
    success = test_small_update()
    
    if success:
        print(f"\n[{datetime.now()}] 🎉 Test passed! Check the logs for detailed information.")
        print("\nIf there was a hang, the logs should show exactly where it occurred.")
    else:
        print(f"\n[{datetime.now()}] ❌ Test failed. Check the logs for errors.")

if __name__ == "__main__":
    main()

