#!/usr/bin/env python3
"""
Identify Failed Symbols Script

This script identifies which symbols from the complete universe failed during the update
and saves them to failed.txt for reprocessing.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ticker_universe import TickerUniverseManager
    from utils.timescaledb_client import get_timescaledb_client
except ImportError:
    # Fallback for when running from utils directory
    from ticker_universe import TickerUniverseManager
    from timescaledb_client import get_timescaledb_client

# Configure logging
os.makedirs('logs/data/failed_symbols', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data/failed_symbols/identify_failed_symbols.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_complete_universe():
    """Get the complete universe of 517 symbols"""
    try:
        manager = TickerUniverseManager()
        universe = manager.get_combined_universe()
        logger.info(f"Retrieved {len(universe)} symbols from complete universe")
        return universe
    except Exception as e:
        logger.error(f"Failed to get universe: {e}")
        return []

def get_symbols_with_data(provider, timeframe):
    """Get symbols that already have data in the database"""
    try:
        client = get_timescaledb_client()
        if not client.ensure_connection():
            logger.error("Failed to connect to database")
            return set()
        
        cursor = client.connection.cursor()
        
        # Optimize the query with better indexing and timeout handling
        query = """
            SELECT DISTINCT symbol 
            FROM market_data 
            WHERE provider = %s AND timeframe = %s
            ORDER BY symbol
        """
        
        # Set a longer timeout for this query
        cursor.execute("SET statement_timeout = '300000'")  # 5 minutes
        
        logger.info(f"Querying database for symbols with {provider.upper()} @ {timeframe} data...")
        cursor.execute(query, (provider.upper(), timeframe))
        
        # Fetch results in batches to avoid memory issues
        symbols_with_data = set()
        batch_size = 1000
        
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            symbols_with_data.update(row[0] for row in batch)
            
            # Log progress for large datasets
            if len(symbols_with_data) % 1000 == 0:
                logger.info(f"Found {len(symbols_with_data)} symbols so far...")
        
        cursor.close()
        
        logger.info(f"Found {len(symbols_with_data)} symbols with existing data")
        return symbols_with_data
        
    except Exception as e:
        logger.error(f"Failed to get symbols with data: {e}")
        
        # Try a simpler query as fallback
        try:
            logger.info("Trying fallback query...")
            cursor = client.connection.cursor()
            cursor.execute("SET statement_timeout = '60000'")  # 1 minute
            
            # Simpler query with LIMIT to test
            test_query = """
                SELECT COUNT(DISTINCT symbol) 
                FROM market_data 
                WHERE provider = %s AND timeframe = %s
            """
            cursor.execute(test_query, (provider.upper(), timeframe))
            count = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"Fallback query shows {count} symbols with data")
            
            if count > 0:
                logger.warning("Database query timed out, but data exists. Consider optimizing the query or increasing timeout.")
                # Return a small sample to indicate data exists
                return {"SAMPLE_DATA_EXISTS"}
            else:
                return set()
                
        except Exception as fallback_error:
            logger.error(f"Fallback query also failed: {fallback_error}")
            return set()

def get_symbols_with_data_individual(provider, timeframe, universe_symbols):
    """Get symbols with data by checking each symbol individually (slower but more reliable)"""
    try:
        client = get_timescaledb_client()
        if not client.ensure_connection():
            logger.error("Failed to connect to database")
            return set()
        
        cursor = client.connection.cursor()
        symbols_with_data = set()
        
        logger.info(f"Checking {len(universe_symbols)} symbols individually...")
        
        for i, symbol in enumerate(universe_symbols):
            if i % 50 == 0:  # Log progress every 50 symbols
                logger.info(f"Checked {i}/{len(universe_symbols)} symbols...")
            
            try:
                # Check if this specific symbol has data
                query = """
                    SELECT COUNT(*) 
                    FROM market_data 
                    WHERE symbol = %s AND provider = %s AND timeframe = %s
                    LIMIT 1
                """
                cursor.execute(query, (symbol, provider.upper(), timeframe))
                count = cursor.fetchone()[0]
                
                if count > 0:
                    symbols_with_data.add(symbol)
                    
            except Exception as e:
                logger.warning(f"Error checking symbol {symbol}: {e}")
                continue
        
        cursor.close()
        logger.info(f"Individual check found {len(symbols_with_data)} symbols with data")
        return symbols_with_data
        
    except Exception as e:
        logger.error(f"Failed to check symbols individually: {e}")
        return set()

def identify_failed_symbols(provider="ib", timeframe="1h", use_individual_check=False):
    """Identify symbols that failed during the update"""
    try:
        # Get complete universe
        complete_universe = get_complete_universe()
        if not complete_universe:
            return []
        
        # Get symbols that already have data
        if use_individual_check:
            logger.info("Using individual symbol check method...")
            symbols_with_data = get_symbols_with_data_individual(provider, timeframe, complete_universe)
        else:
            symbols_with_data = get_symbols_with_data(provider, timeframe)
        
        # Handle fallback case when query times out
        if symbols_with_data == {"SAMPLE_DATA_EXISTS"}:
            logger.warning("Database query timed out. Trying individual check method...")
            symbols_with_data = get_symbols_with_data_individual(provider, timeframe, complete_universe)
        
        # Identify failed symbols (those in universe but not in database)
        failed_symbols = []
        for symbol in complete_universe:
            if symbol not in symbols_with_data:
                failed_symbols.append(symbol)
        
        logger.info(f"Identified {len(failed_symbols)} failed symbols out of {len(complete_universe)} total")
        return failed_symbols
        
    except Exception as e:
        logger.error(f"Failed to identify failed symbols: {e}")
        return []

def save_failed_symbols_to_file(failed_symbols, filename="failed.txt"):
    """Save failed symbols to a text file"""
    try:
        with open(filename, 'w') as f:
            for symbol in failed_symbols:
                f.write(f"{symbol}\n")
        
        logger.info(f"Saved {len(failed_symbols)} failed symbols to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save failed symbols to file: {e}")
        return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Identify failed symbols from universe update",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--provider', 
        default='ib', 
        choices=['ib', 'alpaca'],
        help='Data provider (default: ib)'
    )
    
    parser.add_argument(
        '--timeframe', 
        default='1h', 
        choices=['1h', '1d'],
        help='Timeframe (default: 1h)'
    )
    
    parser.add_argument(
        '--output', 
        default='failed.txt',
        help='Output file for failed symbols (default: failed.txt)'
    )
    
    parser.add_argument(
        '--individual-check', 
        action='store_true',
        help='Check symbols individually (slower but more reliable for large datasets)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Identifying failed symbols for {args.provider} @ {args.timeframe}")
    
    # Identify failed symbols
    failed_symbols = identify_failed_symbols(args.provider, args.timeframe, args.individual_check)
    
    if failed_symbols:
        # Save to file
        if save_failed_symbols_to_file(failed_symbols, args.output):
            logger.info(f"[SUCCESS] Successfully identified and saved {len(failed_symbols)} failed symbols")
            logger.info(f"[FILE] Failed symbols saved to: {args.output}")
            
            # Show first 10 failed symbols
            logger.info(f"[LIST] First 10 failed symbols: {', '.join(failed_symbols[:10])}")
            if len(failed_symbols) > 10:
                logger.info(f"... and {len(failed_symbols) - 10} more")
        else:
            logger.error("[ERROR] Failed to save failed symbols to file")
    else:
        logger.info("[SUCCESS] No failed symbols found - all symbols have data!")

if __name__ == "__main__":
    main()
