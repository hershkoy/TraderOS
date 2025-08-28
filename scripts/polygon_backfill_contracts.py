#!/usr/bin/env python3
"""
Backfill historical QQQ option contracts for the last 2 years
Uses Polygon API to discover contracts from historical dates
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
from utils.option_utils import build_option_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/options_backfill.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptionsContractBackfiller:
    """Backfills historical QQQ option contracts"""
    
    def __init__(self, db_connection_string: Optional[str] = None):
        """
        Initialize the backfiller
        
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
    
    def get_historical_expiration_dates(self, days_back: int = 730) -> List[str]:
        """
        Generate list of historical expiration dates to check
        
        Args:
            days_back: Number of days to go back from today
        
        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        today = date.today()
        expiration_dates = []
        
        # Start from 2 years ago
        start_date = today - timedelta(days=days_back)
        current_date = start_date
        
        # Generate monthly expiration dates (approximate)
        while current_date <= today:
            # Add the date to our list
            expiration_dates.append(current_date.strftime('%Y-%m-%d'))
            
            # Move to next month (approximate)
            current_date += timedelta(days=30)
        
        logger.info(f"Generated {len(expiration_dates)} historical expiration dates")
        return expiration_dates
    
    def discover_contracts_for_date(self, underlying: str, expiration_date: str) -> List[Dict[str, Any]]:
        """
        Discover option contracts for a specific underlying and expiration date
        
        Args:
            underlying: Underlying symbol (e.g., 'QQQ')
            expiration_date: Expiration date in YYYY-MM-DD format
        
        Returns:
            List of discovered contracts
        """
        try:
            logger.info(f"Discovering {underlying} options for expiration {expiration_date}")
            
            # Get options chain from Polygon
            data = self.polygon_client.get_options_chain(underlying, expiration_date)
            
            if 'results' not in data:
                logger.warning(f"No results found for {underlying} on {expiration_date}")
                return []
            
            contracts = []
            for contract in data['results']:
                try:
                    # Extract contract details
                    contract_info = {
                        'underlying': underlying,
                        'expiration': expiration_date,
                        'strike': float(contract.get('strike_price', 0)),
                        'option_right': contract.get('contract_type', 'call').upper()[0],  # 'call' -> 'C'
                        'multiplier': int(contract.get('multiplier', 100)),
                        'occ_symbol': contract.get('ticker', ''),
                        'first_seen': datetime.now(),
                        'last_seen': datetime.now()
                    }
                    
                    # Build our deterministic option_id
                    contract_info['option_id'] = build_option_id(
                        contract_info['underlying'],
                        contract_info['expiration'],
                        contract_info['strike'],
                        contract_info['option_right']
                    )
                    
                    contracts.append(contract_info)
                    
                except Exception as e:
                    logger.error(f"Error processing contract {contract}: {e}")
                    continue
            
            logger.info(f"Discovered {len(contracts)} contracts for {underlying} on {expiration_date}")
            return contracts
            
        except Exception as e:
            logger.error(f"Error discovering contracts for {underlying} on {expiration_date}: {e}")
            return []
    
    def upsert_contracts(self, contracts: List[Dict[str, Any]]) -> int:
        """
        Upsert contracts into the database
        
        Args:
            contracts: List of contract dictionaries
        
        Returns:
            Number of contracts processed
        """
        if not contracts:
            return 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Prepare the upsert statement
                upsert_sql = """
                INSERT INTO option_contracts (
                    option_id, underlying, expiration, strike_cents, option_right, 
                    multiplier, first_seen, last_seen
                ) VALUES (
                    %(option_id)s, %(underlying)s, %(expiration)s, %(strike_cents)s, 
                    %(option_right)s, %(multiplier)s, %(first_seen)s, %(last_seen)s
                )
                ON CONFLICT (option_id) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen,
                    underlying = EXCLUDED.underlying,
                    expiration = EXCLUDED.expiration,
                    strike_cents = EXCLUDED.strike_cents,
                    option_right = EXCLUDED.option_right,
                    multiplier = EXCLUDED.multiplier
                """
                
                processed = 0
                for contract in contracts:
                    try:
                        # Convert strike to cents
                        contract['strike_cents'] = int(round(contract['strike'] * 100))
                        
                        cursor.execute(upsert_sql, contract)
                        processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error upserting contract {contract.get('option_id', 'unknown')}: {e}")
                        continue
                
                conn.commit()
                logger.info(f"Successfully processed {processed} contracts")
                return processed
    
    def backfill_contracts(self, underlying: str = 'QQQ', days_back: int = 730, 
                          sample_rate: int = 5) -> Dict[str, Any]:
        """
        Backfill contracts for the last N days with sampling to respect rate limits
        
        Args:
            underlying: Underlying symbol
            days_back: Number of days to go back
            sample_rate: Only process every Nth date to respect rate limits
        
        Returns:
            Summary of backfill operation
        """
        logger.info(f"Starting backfill for {underlying} over last {days_back} days")
        logger.info(f"Using sample rate of {sample_rate} (every {sample_rate}th date)")
        
        # Get historical expiration dates
        all_dates = self.get_historical_expiration_dates(days_back)
        
        # Sample dates to respect rate limits
        sampled_dates = all_dates[::sample_rate]
        logger.info(f"Processing {len(sampled_dates)} out of {len(all_dates)} dates")
        
        total_contracts = 0
        processed_dates = 0
        failed_dates = 0
        
        for i, expiration_date in enumerate(sampled_dates):
            try:
                logger.info(f"Processing date {i+1}/{len(sampled_dates)}: {expiration_date}")
                
                contracts = self.discover_contracts_for_date(underlying, expiration_date)
                if contracts:
                    processed = self.upsert_contracts(contracts)
                    total_contracts += processed
                    processed_dates += 1
                else:
                    logger.warning(f"No contracts found for {expiration_date}")
                
                # Add delay between requests to be extra respectful of rate limits
                if i < len(sampled_dates) - 1:  # Don't sleep after the last request
                    time.sleep(2)  # 2 second delay between dates
                
            except Exception as e:
                logger.error(f"Error processing expiration {expiration_date}: {e}")
                failed_dates += 1
                continue
        
        summary = {
            'total_contracts': total_contracts,
            'processed_dates': processed_dates,
            'failed_dates': failed_dates,
            'total_dates_checked': len(sampled_dates),
            'underlying': underlying,
            'days_back': days_back,
            'sample_rate': sample_rate
        }
        
        logger.info(f"Backfill completed. Summary: {summary}")
        return summary
    
    def close(self):
        """Close resources"""
        if self.polygon_client:
            self.polygon_client.close()


def main():
    """Main function to run the contract backfill"""
    try:
        logger.info("Starting QQQ options contract backfill")
        
        backfiller = OptionsContractBackfiller()
        
        # Backfill contracts for the last 2 years with sampling
        # Use sample_rate=5 to respect free plan rate limits
        summary = backfiller.backfill_contracts('QQQ', days_back=730, sample_rate=5)
        
        logger.info(f"Contract backfill completed successfully")
        logger.info(f"Summary: {summary}")
        
    except Exception as e:
        logger.error(f"Contract backfill failed: {e}")
        sys.exit(1)
    finally:
        if 'backfiller' in locals():
            backfiller.close()


if __name__ == "__main__":
    main()
