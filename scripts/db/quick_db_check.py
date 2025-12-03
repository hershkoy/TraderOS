#!/usr/bin/env python3
"""
Quick database check - just verify that some data exists
"""

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def quick_check():
    """Quick check to see if data exists"""
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
        
        # Quick count
        cursor.execute("SELECT COUNT(*) FROM market_data LIMIT 1")
        total = cursor.fetchone()[0]
        print(f"Database has data: {total} total records")
        
        # Check a few specific symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'BF.B', 'BRK.B']
        
        print("\nChecking specific symbols:")
        for symbol in test_symbols:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM market_data 
                WHERE symbol = %s AND provider = 'IB' AND timeframe = '1h'
            """, (symbol,))
            count = cursor.fetchone()[0]
            print(f"  {symbol}: {count} records")
        
        # Check total IB 1h symbols (with LIMIT to avoid timeout)
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) 
            FROM market_data 
            WHERE provider = 'IB' AND timeframe = '1h'
        """)
        ib_symbols = cursor.fetchone()[0]
        print(f"\nTotal IB 1h symbols: {ib_symbols}")
        
        cursor.close()
        conn.close()
        
        if ib_symbols > 0:
            print(f"\n✅ Database has {ib_symbols} IB 1h symbols - the identify_failed_symbols script should work!")
        else:
            print(f"\n❌ No IB 1h symbols found - this explains why all symbols appear to be failed")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_check()
