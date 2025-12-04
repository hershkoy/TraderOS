#!/usr/bin/env python3
"""
Test script for TimescaleDB integration
Verifies that the database connection and data operations work correctly
"""

import sys
import pandas as pd
from datetime import datetime, timezone
import numpy as np

def test_timescaledb_connection():
    """Test basic TimescaleDB connection"""
    print("🔍 Testing TimescaleDB connection...")
    
    try:
        from utils.db.timescaledb_client import test_connection
        if test_connection():
            print("✅ TimescaleDB connection successful")
            return True
        else:
            print("❌ TimescaleDB connection failed")
            return False
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def test_data_operations():
    """Test data insertion and retrieval"""
    print("\n📊 Testing data operations...")
    
    try:
        from utils.db.timescaledb_client import get_timescaledb_client
        
        # Create test data
        test_data = []
        base_time = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)
        
        for i in range(10):
            timestamp = base_time.replace(hour=i)
            test_data.append({
                'ts_event': int(timestamp.timestamp() * 1_000_000_000),
                'open': 100.0 + i,
                'high': 101.0 + i,
                'low': 99.0 + i,
                'close': 100.5 + i,
                'volume': 1000 + i * 100
            })
        
        df = pd.DataFrame(test_data)
        
        # Test insertion
        client = get_timescaledb_client()
        if client.insert_market_data(df, 'TEST', 'TEST', '1h'):
            print("✅ Data insertion successful")
        else:
            print("❌ Data insertion failed")
            return False
        
        # Test retrieval
        retrieved_df = client.get_market_data('TEST', '1h', 'TEST')
        if not retrieved_df.empty and len(retrieved_df) >= 10:
            print("✅ Data retrieval successful")
            print(f"   Retrieved {len(retrieved_df)} records")
        else:
            print("❌ Data retrieval failed")
            return False
        
        # Test summary
        summary = client.get_data_summary()
        if summary and 'total_records' in summary:
            print("✅ Data summary successful")
            print(f"   Total records: {summary['total_records']}")
        else:
            print("❌ Data summary failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data operations error: {e}")
        return False

def test_loader_functions():
    """Test the TimescaleDB loader functions"""
    print("\n📈 Testing loader functions...")
    
    try:
        from utils.timescaledb_loader import load_timescaledb_1h, get_available_data, list_available_symbols
        
        # Test loading data
        df = load_timescaledb_1h('TEST', 'TEST')
        if not df.empty:
            print("✅ Loader function successful")
            print(f"   Loaded {len(df)} records")
        else:
            print("❌ Loader function failed")
            return False
        
        # Test available data
        available = get_available_data()
        if available and 'summary' in available:
            print("✅ Available data function successful")
        else:
            print("❌ Available data function failed")
            return False
        
        # Test symbol listing
        symbols = list_available_symbols()
        if 'TEST' in symbols:
            print("✅ Symbol listing successful")
        else:
            print("❌ Symbol listing failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Loader functions error: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with parquet path format"""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        from utils.timescaledb_loader import load_parquet_1h
        from pathlib import Path
        
        # Test with mock parquet path
        mock_path = Path("data/TEST/TEST/1h/test_1h.parquet")
        
        # This should work by extracting symbol and provider from path
        df = load_parquet_1h(mock_path)
        if not df.empty:
            print("✅ Backward compatibility successful")
            return True
        else:
            print("❌ Backward compatibility failed")
            return False
            
    except Exception as e:
        print(f"❌ Backward compatibility error: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        from utils.db.timescaledb_client import get_timescaledb_client
        client = get_timescaledb_client()
        
        deleted = client.delete_data(symbol='TEST')
        print(f"✅ Cleaned up {deleted} test records")
        return True
        
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TimescaleDB Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Connection Test", test_timescaledb_connection),
        ("Data Operations Test", test_data_operations),
        ("Loader Functions Test", test_loader_functions),
        ("Backward Compatibility Test", test_backward_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    # Cleanup
    cleanup_test_data()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! TimescaleDB integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the setup and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
