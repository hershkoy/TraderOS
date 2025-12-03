#!/usr/bin/env python3
"""
Script to check for duplicates in the database.
Checks all major tables for duplicate records based on primary keys and logical uniqueness.
"""

import psycopg2
import os
import sys
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

# Load environment variables
load_dotenv()

def get_database_connection():
    """Get a database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'backtrader'),
            user=os.getenv('DB_USER', 'backtrader_user'),
            password=os.getenv('DB_PASSWORD', 'backtrader_password'),
            connect_timeout=10,
            options='-c statement_timeout=300000'  # 5 minutes timeout
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def check_primary_key_duplicates(cursor, table_name: str, primary_key_columns: List[str]) -> Tuple[int, List[Dict]]:
    """
    Check for duplicates based on primary key columns.
    Returns: (count of duplicate groups, list of duplicate details)
    """
    if not primary_key_columns:
        return 0, []
    
    # Build the GROUP BY and HAVING clause
    group_by_clause = ', '.join(primary_key_columns)
    select_clause = ', '.join(primary_key_columns)
    
    query = f"""
        SELECT {select_clause}, COUNT(*) as duplicate_count
        FROM {table_name}
        GROUP BY {group_by_clause}
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC, {group_by_clause}
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        
        duplicates = []
        for row in results:
            dup_dict = {}
            for i, col in enumerate(primary_key_columns):
                dup_dict[col] = row[i]
            dup_dict['count'] = row[-1]
            duplicates.append(dup_dict)
        
        return len(duplicates), duplicates
    except Exception as e:
        print(f"Error checking primary key duplicates for {table_name}: {e}")
        return 0, []

def check_logical_duplicates(cursor, table_name: str, unique_columns: List[str]) -> Tuple[int, List[Dict]]:
    """
    Check for duplicates based on logical uniqueness (columns that should be unique).
    Returns: (count of duplicate groups, list of duplicate details)
    """
    if not unique_columns:
        return 0, []
    
    group_by_clause = ', '.join(unique_columns)
    select_clause = ', '.join(unique_columns)
    
    query = f"""
        SELECT {select_clause}, COUNT(*) as duplicate_count
        FROM {table_name}
        GROUP BY {group_by_clause}
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC, {group_by_clause}
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        
        duplicates = []
        for row in results:
            dup_dict = {}
            for i, col in enumerate(unique_columns):
                dup_dict[col] = row[i]
            dup_dict['count'] = row[-1]
            duplicates.append(dup_dict)
        
        return len(duplicates), duplicates
    except Exception as e:
        print(f"Error checking logical duplicates for {table_name}: {e}")
        return 0, []

def get_table_primary_keys(cursor, table_name: str) -> List[str]:
    """Get primary key columns for a table"""
    query = """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = %s::regclass
        AND i.indisprimary
        ORDER BY a.attnum
    """
    
    try:
        cursor.execute(query, (table_name,))
        results = cursor.fetchall()
        return [row[0] for row in results]
    except Exception as e:
        print(f"Error getting primary keys for {table_name}: {e}")
        return []

def table_exists(cursor, table_name: str) -> bool:
    """Check if a table exists"""
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
        """, (table_name,))
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error checking if table {table_name} exists: {e}")
        return False

def get_table_row_count(cursor, table_name: str) -> int:
    """Get total row count for a table"""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error getting row count for {table_name}: {e}")
        return 0

def check_market_data_duplicates(cursor) -> Dict:
    """Special check for market_data table (should be unique on ts, symbol, provider, timeframe)"""
    query = """
        SELECT ts, symbol, provider, timeframe, COUNT(*) as duplicate_count
        FROM market_data
        GROUP BY ts, symbol, provider, timeframe
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC, ts DESC, symbol
        LIMIT 100
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        
        duplicates = []
        for row in results:
            duplicates.append({
                'ts': row[0],
                'symbol': row[1],
                'provider': row[2],
                'timeframe': row[3],
                'count': row[4]
            })
        
        total_duplicate_groups = len(duplicates)
        
        # Get total duplicate rows
        if total_duplicate_groups > 0:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM (
                    SELECT ts, symbol, provider, timeframe
                    FROM market_data
                    GROUP BY ts, symbol, provider, timeframe
                    HAVING COUNT(*) > 1
                ) dup_groups
            """)
            total_duplicate_rows = cursor.fetchone()[0]
        else:
            total_duplicate_rows = 0
        
        return {
            'duplicate_groups': total_duplicate_groups,
            'duplicate_rows': total_duplicate_rows,
            'details': duplicates
        }
    except Exception as e:
        print(f"Error checking market_data duplicates: {e}")
        return {'duplicate_groups': 0, 'duplicate_rows': 0, 'details': []}

def check_all_tables():
    """Check all tables for duplicates"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Define tables and their expected unique constraints
    tables_config = {
        'market_data': {
            'logical_unique': ['ts', 'symbol', 'provider', 'timeframe'],
            'description': 'Market data time series (should be unique on ts, symbol, provider, timeframe)'
        },
        'ticker_universe': {
            'primary_key': ['index_name', 'symbol'],
            'description': 'Ticker universe (primary key: index_name, symbol)'
        },
        'ticker_cache_metadata': {
            'primary_key': ['index_name'],
            'description': 'Ticker cache metadata (primary key: index_name)'
        },
        'option_contracts': {
            'primary_key': ['option_id'],
            'description': 'Option contracts (primary key: option_id)'
        },
        'option_quotes': {
            'primary_key': ['option_id', 'ts'],
            'description': 'Option quotes time series (primary key: option_id, ts)'
        },
        'option_eod_prices': {
            'primary_key': ['option_id', 'as_of'],
            'description': 'Option EOD prices (primary key: option_id, as_of)'
        }
    }
    
    print("=" * 80)
    print("DATABASE DUPLICATE CHECK")
    print("=" * 80)
    print()
    
    total_issues = 0
    results = {}
    
    for table_name, config in tables_config.items():
        print(f"Checking table: {table_name}")
        print(f"  Description: {config['description']}")
        
        # Check if table exists
        if not table_exists(cursor, table_name):
            print(f"  [SKIP] Table does not exist, skipping...")
            print()
            continue
        
        # Get row count
        row_count = get_table_row_count(cursor, table_name)
        print(f"  Total rows: {row_count:,}")
        
        table_issues = 0
        
        # Check primary key duplicates
        if 'primary_key' in config:
            pk_cols = config['primary_key']
            print(f"  Checking primary key duplicates on: {', '.join(pk_cols)}")
            dup_count, dup_details = check_primary_key_duplicates(cursor, table_name, pk_cols)
            
            if dup_count > 0:
                table_issues += dup_count
                print(f"  [WARNING] Found {dup_count} duplicate group(s) based on primary key!")
                print(f"  Showing first 10 duplicate groups:")
                for i, dup in enumerate(dup_details[:10], 1):
                    pk_values = ', '.join([f"{k}={v}" for k, v in dup.items() if k != 'count'])
                    print(f"    {i}. {pk_values} (appears {dup['count']} times)")
                if len(dup_details) > 10:
                    print(f"    ... and {len(dup_details) - 10} more groups")
            else:
                print(f"  [OK] No primary key duplicates found")
        
        # Check logical duplicates (for market_data)
        if 'logical_unique' in config:
            unique_cols = config['logical_unique']
            print(f"  Checking logical duplicates on: {', '.join(unique_cols)}")
            
            if table_name == 'market_data':
                # Use special function for market_data
                dup_info = check_market_data_duplicates(cursor)
                if dup_info['duplicate_groups'] > 0:
                    table_issues += dup_info['duplicate_groups']
                    print(f"  [WARNING] Found {dup_info['duplicate_groups']} duplicate group(s)!")
                    print(f"  Total duplicate rows: {dup_info['duplicate_rows']:,}")
                    print(f"  Showing first 10 duplicate groups:")
                    for i, dup in enumerate(dup_info['details'][:10], 1):
                        print(f"    {i}. ts={dup['ts']}, symbol={dup['symbol']}, "
                              f"provider={dup['provider']}, timeframe={dup['timeframe']} "
                              f"(appears {dup['count']} times)")
                    if len(dup_info['details']) > 10:
                        print(f"    ... and {len(dup_info['details']) - 10} more groups")
                else:
                    print(f"  [OK] No logical duplicates found")
            else:
                dup_count, dup_details = check_logical_duplicates(cursor, table_name, unique_cols)
                if dup_count > 0:
                    table_issues += dup_count
                    print(f"  [WARNING] Found {dup_count} duplicate group(s)!")
                    print(f"  Showing first 10 duplicate groups:")
                    for i, dup in enumerate(dup_details[:10], 1):
                        unique_values = ', '.join([f"{k}={v}" for k, v in dup.items() if k != 'count'])
                        print(f"    {i}. {unique_values} (appears {dup['count']} times)")
                    if len(dup_details) > 10:
                        print(f"    ... and {len(dup_details) - 10} more groups")
                else:
                    print(f"  [OK] No logical duplicates found")
        
        results[table_name] = {
            'row_count': row_count,
            'issues': table_issues
        }
        
        if table_issues > 0:
            total_issues += table_issues
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tables checked: {len(tables_config)}")
    print(f"Total duplicate issues found: {total_issues}")
    print()
    
    if total_issues > 0:
        print("Tables with duplicates:")
        for table_name, result in results.items():
            if result['issues'] > 0:
                print(f"  - {table_name}: {result['issues']} duplicate group(s)")
        print()
        print("NOTE: To fix duplicates, you may need to:")
        print("  1. Add unique constraints to prevent future duplicates")
        print("  2. Remove duplicate rows (keeping the most recent or best record)")
        print("  3. Review your data insertion logic to prevent duplicates")
    else:
        print("No duplicates found! All tables are clean.")
    
    print()
    print("=" * 80)
    
    cursor.close()
    conn.close()
    
    return results

if __name__ == "__main__":
    try:
        check_all_tables()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

