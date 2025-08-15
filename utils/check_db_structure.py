#!/usr/bin/env python3
"""
Check TimescaleDB structure and performance
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.timescaledb_client import get_timescaledb_client
import time

def check_database_structure():
    """Check if the database is properly set up with hypertables and indexes."""
    client = get_timescaledb_client()
    if not client.connect():
        print("Failed to connect to database")
        return
    
    cursor = client.connection.cursor()
    
    # Check hypertables
    print("=== HYPERTABLES ===")
    cursor.execute("SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_data'")
    hypertables = cursor.fetchall()
    for row in hypertables:
        print(f"Hypertable: {row}")
    
    # Check indexes
    print("\n=== INDEXES ===")
    cursor.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'market_data'")
    indexes = cursor.fetchall()
    for row in indexes:
        print(f"Index: {row[0]}")
        print(f"  Definition: {row[1]}")
    
    # Check chunks
    print("\n=== CHUNKS ===")
    cursor.execute("SELECT * FROM timescaledb_information.chunks WHERE hypertable_name = 'market_data'")
    chunks = cursor.fetchall()
    for row in chunks:
        print(f"Chunk: {row}")
    
    # Test query performance
    print("\n=== QUERY PERFORMANCE ===")
    start_time = time.time()
    data = client.get_market_data('NFLX', '1h', limit=100)
    end_time = time.time()
    print(f"Query time: {end_time - start_time:.3f} seconds")
    print(f"Records retrieved: {len(data) if data is not None else 0}")
    
    # Test with EXPLAIN ANALYZE
    print("\n=== EXPLAIN ANALYZE ===")
    cursor.execute("""
        EXPLAIN (ANALYZE, BUFFERS) 
        SELECT * FROM market_data 
        WHERE symbol = 'NFLX' AND timeframe = '1h'
        ORDER BY ts DESC 
        LIMIT 100
    """)
    explain_results = cursor.fetchall()
    for row in explain_results:
        print(row[0])
    
    client.disconnect()

if __name__ == "__main__":
    check_database_structure()
