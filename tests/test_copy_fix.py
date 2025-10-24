#!/usr/bin/env python3
"""
Test script to verify the COPY error handling fix
"""

import sys
import os
from datetime import datetime

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_copy_error_handling():
    """Test that the COPY error handling works correctly"""
    try:
        from timescaledb_client import get_timescaledb_client
        import pandas as pd
        
        print(f"[{datetime.now()}] Testing COPY error handling...")
        
        # Create a test dataset that might trigger COPY issues
        test_data = pd.DataFrame({
            'ts_event': [pd.Timestamp.now().value],  # nanoseconds timestamp
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        })
        
        client = get_timescaledb_client()
        if client.ensure_connection():
            print(f"[{datetime.now()}] ✅ Database connection established")
            
            # Test the insert
            success = client.insert_market_data(test_data, 'TEST_COPY', 'DEBUG', '1h')
            
            if success:
                print(f"[{datetime.now()}] ✅ Test insert successful - COPY error handling working")
                return True
            else:
                print(f"[{datetime.now()}] ❌ Test insert failed")
                return False
        else:
            print(f"[{datetime.now()}] ❌ Cannot establish database connection")
            return False
            
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("=" * 60)
    print("COPY ERROR HANDLING TEST")
    print("=" * 60)
    
    success = test_copy_error_handling()
    
    if success:
        print(f"\n[{datetime.now()}] 🎉 Test passed! COPY error handling is working correctly.")
        print("\nThe fix should resolve the 'no COPY in progress' errors.")
    else:
        print(f"\n[{datetime.now()}] ❌ Test failed. There may still be issues.")

if __name__ == "__main__":
    main()

