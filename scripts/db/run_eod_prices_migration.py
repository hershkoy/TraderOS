#!/usr/bin/env python3
"""
Run database migration to add option_eod_prices table
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_connection_string():
    """Get database connection string from environment"""
    host = os.getenv('TIMESCALEDB_HOST', 'localhost')
    port = os.getenv('TIMESCALEDB_PORT', '5432')
    database = os.getenv('TIMESCALEDB_DATABASE', 'backtrader')
    user = os.getenv('TIMESCALEDB_USER', 'backtrader_user')
    password = os.getenv('TIMESCALEDB_PASSWORD', 'backtrader_password')
    
    return f"host={host} port={port} dbname={database} user={user} password={password}"

def run_migration():
    """Run the EOD prices table migration"""
    
    print("Running option_eod_prices table migration...")
    
    # Read the migration SQL
    migration_file = "init-scripts/05-option-eod-prices-schema.sql"
    
    try:
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        print(f"✓ Migration SQL loaded from {migration_file}")
        
    except FileNotFoundError:
        print(f"✗ Migration file not found: {migration_file}")
        return False
    
    # Connect to database and run migration
    try:
        conn_string = get_connection_string()
        print(f"Connecting to database...")
        
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                print("✓ Database connected, running migration...")
                
                # Execute the migration
                cur.execute(migration_sql)
                
                # Check if table was created
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_name = 'option_eod_prices'
                """)
                
                if cur.fetchone():
                    print("✓ option_eod_prices table created successfully!")
                else:
                    print("✗ Table creation failed")
                    return False
                
                # Show table structure
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'option_eod_prices'
                    ORDER BY ordinal_position
                """)
                
                columns = cur.fetchall()
                print(f"\nTable structure:")
                for col in columns:
                    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                    print(f"  {col[0]}: {col[1]} {nullable}")
                
                conn.commit()
                print("\n✓ Migration completed successfully!")
                return True
                
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Option EOD Prices Migration")
    print("=" * 40)
    
    success = run_migration()
    
    if success:
        print("\nNext steps:")
        print("1. Test the EOD pricing: python test_eod_pricing.py")
        print("2. Run the backfill with EOD data: python scripts/api/polygon/polygon_backfill_contracts.py")
    else:
        print("\nMigration failed. Please check the error messages above.")
        sys.exit(1)


