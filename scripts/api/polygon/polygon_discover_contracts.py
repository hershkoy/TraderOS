#!/usr/bin/env python3
"""
Discover QQQ option contracts daily using Polygon API
Discovers new contracts and updates existing ones in the database
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

from utils.api.polygon_client import get_polygon_client
from utils.options.option_utils import build_option_id

# Configure logging
os.makedirs('logs/api/polygon/discovery', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api/polygon/discovery/options_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptionsContractDiscoverer:
    """Discovers and stores QQQ option contracts"""
    
    def __init__(self, db_connection_string: Optional[str] = None):
        """
        Initialize the discoverer
        
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
    
    def discover_near_expirations(self, underlying: str = 'QQQ', days_ahead: int = 365) -> int:
        """
        Discover contracts for near expirations
        
        Args:
            underlying: Underlying symbol
            days_ahead: Number of days ahead to look for expirations
        
        Returns:
            Total number of contracts discovered
        """
        today = date.today()
        total_contracts = 0
        
        # Get list of expiration dates to check
        # For now, we'll check monthly expirations for the next year
        expiration_dates = []
        current_date = today
        
        while current_date <= today + timedelta(days=days_ahead):
            # Check for monthly expirations (third Friday of each month)
            # This is a simplified approach - in practice you'd want to get actual expiration dates
            expiration_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=30)  # Approximate monthly
        
        logger.info(f"Checking {len(expiration_dates)} expiration dates for {underlying}")
        
        for expiration_date in expiration_dates:
            try:
                contracts = self.discover_contracts_for_date(underlying, expiration_date)
                if contracts:
                    processed = self.upsert_contracts(contracts)
                    total_contracts += processed
                
                # Rate limiting is handled by the Polygon client
                
            except Exception as e:
                logger.error(f"Error processing expiration {expiration_date}: {e}")
                continue
        
        logger.info(f"Total contracts discovered: {total_contracts}")
        return total_contracts
    
    def close(self):
        """Close resources"""
        if self.polygon_client:
            self.polygon_client.close()


def main():
    """Main function to run the contract discovery"""
    try:
        logger.info("Starting QQQ options contract discovery")
        
        discoverer = OptionsContractDiscoverer()
        
        # Discover contracts for the next year
        total_contracts = discoverer.discover_near_expirations('QQQ', days_ahead=365)
        
        logger.info(f"Contract discovery completed. Total contracts: {total_contracts}")
        
    except Exception as e:
        logger.error(f"Contract discovery failed: {e}")
        sys.exit(1)
    finally:
        if 'discoverer' in locals():
            discoverer.close()


if __name__ == "__main__":
    main()
