#!/usr/bin/env python3
"""
Greeks Fill Script

This script backfills missing Greeks (delta, gamma, theta, vega, implied volatility)
for option quotes using the Black-Scholes model. It's used when Greeks are not
available from the data provider.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.greeks import batch_update_greeks
from utils.database import get_database_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/greeks_fill.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_quotes_without_greeks(conn, start_date: str = None, end_date: str = None, underlying: str = 'QQQ') -> list:
    """
    Get option quotes that are missing Greeks.
    
    Args:
        conn: Database connection
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        underlying: Underlying symbol
    
    Returns:
        List of quotes with missing Greeks
    """
    query = """
        SELECT 
            q.ts,
            q.option_id,
            q.bid,
            q.ask,
            q.last,
            c.option_right,
            c.expiration,
            c.strike_cents,
            m.close as underlying_close
        FROM option_quotes q
        JOIN option_contracts c USING (option_id)
        LEFT JOIN market_data m ON (
            m.symbol = c.underlying 
            AND DATE(m.ts) = DATE(q.ts)
            AND m.timeframe = '1d'
        )
        WHERE c.underlying = %s
          AND q.snapshot_type = 'eod'
          AND (q.iv IS NULL OR q.delta IS NULL OR q.gamma IS NULL OR q.theta IS NULL OR q.vega IS NULL)
          AND q.bid IS NOT NULL 
          AND q.ask IS NOT NULL
          AND m.close IS NOT NULL
    """
    
    params = [underlying]
    
    if start_date:
        query += " AND DATE(q.ts) >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND DATE(q.ts) <= %s"
        params.append(end_date)
    
    query += " ORDER BY q.ts DESC"
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()

def prepare_quotes_data(quotes: list) -> list:
    """
    Prepare quotes data for Greeks calculation.
    
    Args:
        quotes: List of quote dictionaries from database
    
    Returns:
        List of prepared quote data
    """
    prepared_data = []
    
    for quote in quotes:
        # Calculate option price as bid/ask mid
        if quote['bid'] is not None and quote['ask'] is not None:
            option_price = (quote['bid'] + quote['ask']) / 2
        elif quote['last'] is not None:
            option_price = quote['last']
        else:
            continue  # Skip if no price available
        
        # Calculate days to expiration
        expiration_date = quote['expiration']
        quote_date = quote['ts'].date()
        days_to_exp = (expiration_date - quote_date).days
        
        if days_to_exp <= 0:
            continue  # Skip expired options
        
        # Extract strike price
        strike = quote['strike_cents'] / 100.0
        
        prepared_data.append({
            'ts': quote['ts'],
            'option_id': quote['option_id'],
            'underlying_price': quote['underlying_close'],
            'option_price': option_price,
            'option_type': quote['option_right'],
            'days_to_exp': days_to_exp,
            'strike': strike
        })
    
    return prepared_data

def main():
    """Main function to backfill missing Greeks."""
    logger.info("Starting Greeks backfill process")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Backfill missing Greeks for option quotes')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--underlying', default='QQQ', help='Underlying symbol')
    parser.add_argument('--risk-free-rate', type=float, default=0.0, help='Risk-free interest rate')
    parser.add_argument('--dividend-yield', type=float, default=0.0, help='Dividend yield')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no database updates)')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Processing quotes for {args.underlying} from {args.start_date} to {args.end_date}")
    
    try:
        # Connect to database
        conn = get_database_connection()
        
        # Get quotes without Greeks
        quotes = get_quotes_without_greeks(conn, args.start_date, args.end_date, args.underlying)
        logger.info(f"Found {len(quotes)} quotes with missing Greeks")
        
        if not quotes:
            logger.info("No quotes with missing Greeks found")
            return
        
        # Prepare data for Greeks calculation
        prepared_data = prepare_quotes_data(quotes)
        logger.info(f"Prepared {len(prepared_data)} quotes for Greeks calculation")
        
        if args.dry_run:
            logger.info("DRY RUN MODE: No database updates will be made")
            for quote in prepared_data[:5]:  # Show first 5 examples
                logger.info(f"Would process: {quote['option_id']} at {quote['ts']}")
            return
        
        # Update Greeks in batches
        batch_size = 100
        total_success = 0
        total_failure = 0
        
        for i in range(0, len(prepared_data), batch_size):
            batch = prepared_data[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(prepared_data) + batch_size - 1)//batch_size}")
            
            result = batch_update_greeks(
                conn, 
                batch, 
                args.risk_free_rate, 
                args.dividend_yield
            )
            
            total_success += result['success_count']
            total_failure += result['failure_count']
            
            logger.info(f"Batch result: {result['success_count']} success, {result['failure_count']} failure")
        
        logger.info(f"Greeks backfill completed: {total_success} success, {total_failure} failure")
        
    except Exception as e:
        logger.error(f"Error during Greeks backfill: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
