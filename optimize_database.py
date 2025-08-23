#!/usr/bin/env python3
"""
Database optimization script - add missing indexes for faster queries
"""

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def optimize_database():
    """Add missing indexes to improve query performance"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'backtrader'),
            user=os.getenv('DB_USER', 'backtrader_user'),
            password=os.getenv('DB_PASSWORD', 'backtrader_password')
        )
        
        cursor = conn.cursor()
        
        print("Adding database indexes for better performance...")
        
        # Add composite index for the most common query pattern (without CONCURRENTLY)
        print("Adding composite index for provider + timeframe + symbol...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_provider_timeframe_symbol 
            ON market_data (provider, timeframe, symbol)
        """)
        
        # Add index for symbol lookups
        print("Adding index for symbol lookups...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_provider_timeframe 
            ON market_data (symbol, provider, timeframe)
        """)
        
        # Add index for DISTINCT symbol queries
        print("Adding index for distinct symbol queries...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_distinct_symbols 
            ON market_data (provider, timeframe, symbol) 
            WHERE symbol IS NOT NULL
        """)
        
        # Commit changes
        conn.commit()
        print("✅ Database indexes added successfully!")
        
        # Test query performance
        print("\nTesting query performance...")
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) 
            FROM market_data 
            WHERE provider = 'IB' AND timeframe = '1h'
        """)
        result = cursor.fetchone()
        print(f"✅ Query completed successfully! Found {result[0]} IB 1h symbols")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error optimizing database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    optimize_database()
