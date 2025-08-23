#!/usr/bin/env python3
"""
Simple script to check what data exists in the database
"""

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_database_data():
    """Check what data exists in the database"""
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
        
        # Set a longer timeout
        cursor.execute("SET statement_timeout = '300000'")  # 5 minutes
        
        # Check total records
        cursor.execute("SELECT COUNT(*) FROM market_data")
        total_records = cursor.fetchone()[0]
        print(f"Total records in market_data: {total_records}")
        
        # Check specific IB 1h data first (most important)
        print("\nChecking IB 1h data...")
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) 
            FROM market_data 
            WHERE provider = 'IB' AND timeframe = '1h'
        """)
        ib_1h_symbols = cursor.fetchone()[0]
        print(f"IB 1h symbols: {ib_1h_symbols}")
        
        # Show some sample symbols
        cursor.execute("""
            SELECT DISTINCT symbol 
            FROM market_data 
            WHERE provider = 'IB' AND timeframe = '1h'
            ORDER BY symbol 
            LIMIT 10
        """)
        sample_symbols = [row[0] for row in cursor.fetchall()]
        print(f"Sample IB 1h symbols: {sample_symbols}")
        
        # Check other providers quickly
        print("\nChecking other providers...")
        cursor.execute("""
            SELECT provider, timeframe, COUNT(DISTINCT symbol) 
            FROM market_data 
            GROUP BY provider, timeframe
            ORDER BY provider, timeframe
        """)
        symbol_counts = cursor.fetchall()
        print("Unique symbols by provider and timeframe:")
        for provider, timeframe, count in symbol_counts:
            print(f"  {provider} @ {timeframe}: {count} symbols")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_data()
