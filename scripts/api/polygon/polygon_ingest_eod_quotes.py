#!/usr/bin/env python3
"""
Ingest daily EOD quotes for discovered option contracts
Fetches end-of-day prices and Greeks from Polygon API
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polygon_client import get_polygon_client

# Configure logging
os.makedirs('logs/api/polygon/ingest', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api/polygon/ingest/options_quotes_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptionsQuotesIngester:
    """Ingests daily EOD quotes for option contracts"""
    
    def __init__(self, db_connection_string: Optional[str] = None):
        """
        Initialize the ingester
        
        Args:
            db_connection_string: PostgreSQL connection string
        """
        self.db_connection_string = db_connection_string or self._get_default_connection_string()
        self.polygon_client = get_polygon_client()
        
    def _get_default_connection_string(self) -> str:
        """Get default database connection string from environment"""
        host = os.getenv('TIMESCALEDB_HOST', 'localhost')
        port = os.getenv('TIMESCALEDB_PORT', '5432')
        database = os.getenv('TIMESCALEDB_DATABASE', 'backtrader')
        user = os.getenv('TIMESCALEDB_USER', 'backtrader_user')
        password = os.getenv('TIMESCALEDB_PASSWORD', 'backtrader_password')
        
        return f"host={host} port={port} dbname={database} user={user} password={password}"
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_connection_string)
    
    def get_active_contracts(self, underlying: str = 'QQQ') -> List[Dict[str, Any]]:
        """
        Get list of active option contracts from the database
        
        Args:
            underlying: Underlying symbol to filter by
        
        Returns:
            List of active contracts
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                SELECT option_id, underlying, expiration, strike_cents, option_right
                FROM option_contracts 
                WHERE underlying = %s AND expiration > CURRENT_DATE
                ORDER BY expiration, strike_cents
                """
                cursor.execute(query, (underlying,))
                contracts = cursor.fetchall()
                
                # Convert to regular dictionaries
                return [dict(contract) for contract in contracts]
    
    def get_quotes_for_contract(self, option_id: str, quote_date: str) -> Optional[Dict[str, Any]]:
        """
        Get EOD quotes for a specific option contract on a given date
        
        Args:
            option_id: Option identifier
            quote_date: Date to get quotes for (YYYY-MM-DD)
        
        Returns:
            Quote data or None if not available
        """
        try:
            # Try to get previous day's close (EOD data)
            data = self.polygon_client.get_options_previous_close(option_id)
            
            if 'results' not in data or not data['results']:
                logger.debug(f"No EOD data for {option_id} on {quote_date}")
                return None
            
            result = data['results'][0]
            
            # Extract quote information
            quote = {
                'ts': datetime.strptime(quote_date, '%Y-%m-%d').replace(tzinfo=datetime.utcnow().tzinfo),
                'option_id': option_id,
                'bid': result.get('bid'),
                'ask': result.get('ask'),
                'last': result.get('c'),  # Close price
                'volume': result.get('v'),  # Volume
                'open_interest': None,  # Not available in aggregates endpoint
                'iv': None,  # Not available in aggregates endpoint
                'delta': None,  # Not available in aggregates endpoint
                'gamma': None,  # Not available in aggregates endpoint
                'theta': None,  # Not available in aggregates endpoint
                'vega': None,  # Not available in aggregates endpoint
                'snapshot_type': 'eod'
            }
            
            return quote
            
        except Exception as e:
            logger.error(f"Error getting quotes for {option_id} on {quote_date}: {e}")
            return None
    
    def upsert_quotes(self, quotes: List[Dict[str, Any]]) -> int:
        """
        Upsert quotes into the database
        
        Args:
            quotes: List of quote dictionaries
        
        Returns:
            Number of quotes processed
        """
        if not quotes:
            return 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Prepare the upsert statement
                upsert_sql = """
                INSERT INTO option_quotes (
                    ts, option_id, bid, ask, last, volume, open_interest,
                    iv, delta, gamma, theta, vega, snapshot_type
                ) VALUES (
                    %(ts)s, %(option_id)s, %(bid)s, %(ask)s, %(last)s, %(volume)s, %(open_interest)s,
                    %(iv)s, %(delta)s, %(gamma)s, %(theta)s, %(vega)s, %(snapshot_type)s
                )
                ON CONFLICT (option_id, ts) DO UPDATE SET
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    last = EXCLUDED.last,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    iv = EXCLUDED.iv,
                    delta = EXCLUDED.delta,
                    gamma = EXCLUDED.gamma,
                    theta = EXCLUDED.theta,
                    vega = EXCLUDED.vega,
                    snapshot_type = EXCLUDED.snapshot_type
                """
                
                processed = 0
                for quote in quotes:
                    try:
                        cursor.execute(upsert_sql, quote)
                        processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error upserting quote for {quote.get('option_id', 'unknown')}: {e}")
                        continue
                
                conn.commit()
                logger.info(f"Successfully processed {processed} quotes")
                return processed
    
    def ingest_quotes_for_date(self, quote_date: str, underlying: str = 'QQQ') -> int:
        """
        Ingest quotes for all active contracts on a specific date
        
        Args:
            quote_date: Date to ingest quotes for (YYYY-MM-DD)
            underlying: Underlying symbol
        
        Returns:
            Number of quotes ingested
        """
        logger.info(f"Ingesting quotes for {underlying} on {quote_date}")
        
        # Get active contracts
        contracts = self.get_active_contracts(underlying)
        logger.info(f"Found {len(contracts)} active contracts for {underlying}")
        
        if not contracts:
            logger.warning(f"No active contracts found for {underlying}")
            return 0
        
        quotes = []
        processed_contracts = 0
        
        for contract in contracts:
            try:
                quote = self.get_quotes_for_contract(contract['option_id'], quote_date)
                if quote:
                    quotes.append(quote)
                    processed_contracts += 1
                
                # Rate limiting is handled by the Polygon client
                
            except Exception as e:
                logger.error(f"Error processing contract {contract['option_id']}: {e}")
                continue
        
        logger.info(f"Retrieved quotes for {processed_contracts} contracts")
        
        # Upsert quotes to database
        if quotes:
            processed_quotes = self.upsert_quotes(quotes)
            logger.info(f"Ingested {processed_quotes} quotes for {quote_date}")
            return processed_quotes
        else:
            logger.warning(f"No quotes retrieved for {quote_date}")
            return 0
    
    def ingest_quotes_for_date_range(self, start_date: str, end_date: str, 
                                   underlying: str = 'QQQ') -> Dict[str, Any]:
        """
        Ingest quotes for a range of dates
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            underlying: Underlying symbol
        
        Returns:
            Summary of ingestion operation
        """
        logger.info(f"Starting quote ingestion for {underlying} from {start_date} to {end_date}")
        
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        total_quotes = 0
        processed_dates = 0
        failed_dates = 0
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                quotes_ingested = self.ingest_quotes_for_date(date_str, underlying)
                if quotes_ingested > 0:
                    total_quotes += quotes_ingested
                    processed_dates += 1
                else:
                    logger.warning(f"No quotes ingested for {date_str}")
                
            except Exception as e:
                logger.error(f"Error ingesting quotes for {date_str}: {e}")
                failed_dates += 1
            
            current_date += timedelta(days=1)
        
        summary = {
            'total_quotes': total_quotes,
            'processed_dates': processed_dates,
            'failed_dates': failed_dates,
            'start_date': start_date,
            'end_date': end_date,
            'underlying': underlying
        }
        
        logger.info(f"Quote ingestion completed. Summary: {summary}")
        return summary
    
    def close(self):
        """Close resources"""
        if self.polygon_client:
            self.polygon_client.close()


def main():
    """Main function to run the quote ingestion"""
    try:
        logger.info("Starting QQQ options quotes ingestion")
        
        ingester = OptionsQuotesIngester()
        
        # Get yesterday's date for EOD quotes
        yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Ingest quotes for yesterday
        quotes_ingested = ingester.ingest_quotes_for_date(yesterday, 'QQQ')
        
        logger.info(f"Quote ingestion completed. Quotes ingested: {quotes_ingested}")
        
    except Exception as e:
        logger.error(f"Quote ingestion failed: {e}")
        sys.exit(1)
    finally:
        if 'ingester' in locals():
            ingester.close()


if __name__ == "__main__":
    main()
