#!/usr/bin/env python3
"""
Debug script for universe update hanging issues
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_timescaledb_connection():
    """Test basic TimescaleDB connectivity"""
    try:
        from timescaledb_client import get_timescaledb_client
        
        print(f"[{datetime.now()}] Testing TimescaleDB connection...")
        client = get_timescaledb_client()
        
        if client.ensure_connection():
            print(f"[{datetime.now()}] ✅ Database connection successful")
            
            # Test a simple query
            cursor = client.connection.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            print(f"[{datetime.now()}] ✅ Database version: {version[0]}")
            cursor.close()
            
            return True
        else:
            print(f"[{datetime.now()}] ❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error testing database: {e}")
        return False

def test_small_insert():
    """Test a small data insert to identify hanging point"""
    try:
        from timescaledb_client import get_timescaledb_client
        import pandas as pd
        
        print(f"[{datetime.now()}] Testing small data insert...")
        
        # Create a small test dataset
        test_data = pd.DataFrame({
            'ts': [datetime.now()],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        })
        
        client = get_timescaledb_client()
        if client.ensure_connection():
            print(f"[{datetime.now()}] ✅ Starting test insert...")
            
            # Test the insert with timeout
            start_time = time.time()
            success = client.insert_market_data(test_data, 'TEST', 'DEBUG', '1h')
            end_time = time.time()
            
            if success:
                print(f"[{datetime.now()}] ✅ Test insert successful in {end_time - start_time:.2f}s")
                return True
            else:
                print(f"[{datetime.now()}] ❌ Test insert failed")
                return False
        else:
            print(f"[{datetime.now()}] ❌ Cannot establish database connection")
            return False
            
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error during test insert: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run diagnostic tests"""
    print("=" * 60)
    print("UNIVERSE UPDATE DEBUG DIAGNOSTICS")
    print("=" * 60)
    
    # Test 1: Basic connectivity
    print(f"\n[{datetime.now()}] Test 1: Database Connectivity")
    print("-" * 40)
    db_ok = test_timescaledb_connection()
    
    if not db_ok:
        print(f"\n[{datetime.now()}] ❌ Database connectivity failed. Check your database configuration.")
        return
    
    # Test 2: Small insert
    print(f"\n[{datetime.now()}] Test 2: Small Data Insert")
    print("-" * 40)
    insert_ok = test_small_insert()
    
    if not insert_ok:
        print(f"\n[{datetime.now()}] ❌ Small insert failed. This indicates a problem with the insert logic.")
        return
    
    print(f"\n[{datetime.now()}] ✅ All basic tests passed!")
    print(f"\n[{datetime.now()}] If the universe update still hangs, the issue is likely:")
    print("1. Large dataset processing")
    print("2. Network latency during data transfer")
    print("3. Database performance under load")
    print("4. Memory issues with large datasets")
    
    print(f"\n[{datetime.now()}] Recommendations:")
    print("- Try running with smaller batch sizes")
    print("- Check database performance (CPU, memory, disk I/O)")
    print("- Monitor network connectivity to database")
    print("- Consider reducing the number of concurrent operations")

if __name__ == "__main__":
    main()

