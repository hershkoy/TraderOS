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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        query = """
            SELECT DISTINCT symbol 
            FROM market_data 
            WHERE provider = %s AND timeframe = %s
        """
        cursor.execute(query, (provider.upper(), timeframe))
        results = cursor.fetchall()
        cursor.close()
        
        symbols_with_data = {row[0] for row in results}
        logger.info(f"Found {len(symbols_with_data)} symbols with existing data")
        return symbols_with_data
        
    except Exception as e:
        logger.error(f"Failed to get symbols with data: {e}")
        return set()

def identify_failed_symbols(provider="ib", timeframe="1h"):
    """Identify symbols that failed during the update"""
    try:
        # Get complete universe
        complete_universe = get_complete_universe()
        if not complete_universe:
            return []
        
        # Get symbols that already have data
        symbols_with_data = get_symbols_with_data(provider, timeframe)
        
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
    
    args = parser.parse_args()
    
    logger.info(f"Identifying failed symbols for {args.provider} @ {args.timeframe}")
    
    # Identify failed symbols
    failed_symbols = identify_failed_symbols(args.provider, args.timeframe)
    
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
