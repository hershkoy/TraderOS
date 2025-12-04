#!/usr/bin/env python3
"""
Backfill historical QQQ option contracts for the last 2 years
Uses Polygon API to discover contracts from historical dates
Only processes Friday expiration dates to reduce data volume
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dateutil.relativedelta import relativedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api.polygon_client import get_polygon_client
from utils.options.option_utils import build_option_id
from utils.backtesting.date_rules import get_next_friday_after

def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging with specified level and optional file output"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    handlers = []
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # Add default file handler
    os.makedirs('logs/api/polygon/backfill', exist_ok=True)
    default_file_handler = logging.FileHandler('logs/api/polygon/backfill/options_backfill.log')
    default_file_handler.setLevel(numeric_level)
    default_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    default_file_handler.setFormatter(default_file_formatter)
    handlers.append(default_file_handler)
    
    # Add custom log file handler if specified
    if log_file:
        try:
            # Ensure the directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            custom_file_handler = logging.FileHandler(log_file)
            custom_file_handler.setLevel(numeric_level)
            custom_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            custom_file_handler.setFormatter(custom_file_formatter)
            handlers.append(custom_file_handler)
            
            print(f"✓ Logging to custom file: {log_file}")
        except Exception as e:
            print(f"⚠ Warning: Could not create custom log file '{log_file}': {e}")
            print("   Continuing with default logging only.")
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Create and configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(numeric_level)
    
    # Set Polygon client logger to DEBUG if we want detailed API logging
    if numeric_level <= logging.DEBUG:
        logging.getLogger('utils.polygon_client').setLevel(logging.DEBUG)
    
    return logger

logger = setup_logging()  # Default to INFO level


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
        
    def _find_next_friday_expiration(self, start_date: str, lookback_days: int = 365) -> str:
        """
        Find the next Friday expiration date that is approximately lookback_days in the future
        
        Args:
            start_date: Starting date in YYYY-MM-DD format
            lookback_days: Target number of days to look forward (default: 365)
        
        Returns:
            Next Friday expiration date in YYYY-MM-DD format
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        target_dt = start_dt + timedelta(days=lookback_days)
        
        # Find the next Friday after the target date
        while target_dt.weekday() != 4:  # 4 = Friday
            target_dt += timedelta(days=1)
        
        return target_dt.strftime('%Y-%m-%d')

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
        Get historical expiration dates going back N days from today
        
        Args:
            days_back: Number of days to go back from today
        
        Returns:
            List of expiration dates in YYYY-MM-DD format, going backwards in time
        """
        today = date.today()
        start_date = today - timedelta(days=days_back)
        
        try:
            logger.info(f"Fetching historical expiration dates for QQQ from {start_date} to {today}")
            
            # For historical backfill, we need to generate dates going backwards
            # The Polygon API approach was returning future expirations, not historical ones
            logger.info("Generating historical expiration dates for backfill")
            
            # Generate dates going backwards from today to start_date
            # Focus on Friday expirations (weekday 4) as these are most common for options
            historical_dates = []
            current_date = today
            
            while current_date >= start_date:
                # Check if current date is a Friday
                if current_date.weekday() == 4:
                    historical_dates.append(current_date.strftime('%Y-%m-%d'))
                
                # Move backwards one day
                current_date -= timedelta(days=1)
            
            # Sort dates in chronological order (oldest first)
            historical_dates.sort()
            
            # Log the actual date range we're covering
            if historical_dates:
                actual_start = min(historical_dates)
                actual_end = max(historical_dates)
                logger.info(f"Actual date range covered: {actual_start} to {actual_end}")
                logger.info(f"Covering approximately {len(historical_dates)} Friday expirations")
            
            logger.info(f"Generated {len(historical_dates)} historical Friday expiration dates from {start_date} to {today}")
            if historical_dates:
                logger.debug(f"First few dates: {historical_dates[:5]}")
                logger.debug(f"Last few dates: {historical_dates[-5:]}")
            
            return historical_dates
            
        except Exception as e:
            logger.warning(f"Failed to generate historical expiration dates: {e}")
            logger.info("Falling back to approximate monthly dates")
            
            # Fallback to approximate dates if generation fails
            expiration_dates = []
            current_date = start_date
            
            while current_date <= today:
                # Find next Friday after current_date
                while current_date.weekday() != 4:
                    current_date += timedelta(days=1)
                
                if current_date <= today:
                    expiration_dates.append(current_date.strftime('%Y-%m-%d'))
                
                # Move to next month
                current_date += timedelta(days=30)
            
            logger.info(f"Generated {len(expiration_dates)} approximate Friday expiration dates")
            return expiration_dates
            
        except Exception as e:
            logger.warning(f"Failed to get real expirations from Polygon: {e}")
            logger.info("Falling back to approximate monthly dates")
            
            # Fallback to approximate dates if API fails
            expiration_dates = []
            current_date = start_date
            
            while current_date <= today:
                expiration_dates.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=30)
            
            logger.info(f"Generated {len(expiration_dates)} approximate expiration dates")
            return expiration_dates
    
    def discover_contracts_for_date(self, underlying: str, discovery_date: str, as_of: str = None, expired: bool = None) -> List[Dict[str, Any]]:
        """
        Discover option contracts for a specific underlying and discovery date
        
        Args:
            underlying: Underlying symbol (e.g., 'QQQ')
            discovery_date: Date to look back from (YYYY-MM-DD format) - when we want to see what contracts existed
            as_of: Discover contracts "as of" a future date (YYYY-MM-DD) - when contracts will expire
            expired: Filter by expired status (True/False/None for all)
        
        Note:
            For LEAPS contracts, discovery_date should be the historical date (e.g., 2023-10-06)
            and as_of should be when the contracts expire (e.g., 2024-10-06)
        
        Returns:
            List of discovered contracts
        """
        try:
            logger.info(f"Discovering {underlying} options expiring on {as_of} as they existed on {discovery_date}")
            
            # Get options chain from Polygon for the specific expiration date
            # We want contracts that expire on the as_of date (future) but existed on discovery_date (past)
            data = self.polygon_client.get_options_chain(
                underlying, 
                as_of,  # Use as_of as the expiration date (future)
                as_of=discovery_date,  # Use discovery_date as the as_of date (past)
                expired=expired
            )
            
            # Debug: Log the raw API response
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Polygon API response for {underlying} on {discovery_date}: {data}")
            
            if 'results' not in data:
                logger.warning(f"No results found for {underlying} as of {discovery_date}")
                logger.debug(f"API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return []
            
            contracts = []
            for contract in data['results']:
                try:
                    # Extract contract details
                    contract_info = {
                        'underlying': underlying,
                        'expiration': contract.get('expiration_date', discovery_date),  # Use actual expiration from contract
                        'strike': float(contract.get('strike_price', 0)),
                        'option_right': contract.get('contract_type', 'call').upper()[0],  # 'call' -> 'C'
                        'multiplier': int(contract.get('multiplier', 100)),
                        'occ_symbol': contract.get('ticker', ''),
                        'polygon_ticker': contract.get('ticker', ''),  # Store Polygon's ticker for API calls
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
            
            logger.info(f"Discovered {len(contracts)} contracts for {underlying} as of {discovery_date}")
            
            # Debug: Log some contract details
            if logger.isEnabledFor(logging.DEBUG) and contracts:
                logger.debug(f"Sample contracts discovered:")
                for i, contract in enumerate(contracts[:3]):  # Show first 3 contracts
                    logger.debug(f"  Contract {i+1}: {contract.get('polygon_ticker')} - Strike: {contract.get('strike')} - Exp: {contract.get('expiration')}")
            
            # Return contracts without fetching EOD data - that should happen after contracts are stored
            return contracts
            
        except Exception as e:
            logger.error(f"Error discovering contracts for {underlying} as of {discovery_date}: {e}")
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
                    multiplier, first_seen, last_seen, polygon_ticker
                ) VALUES (
                    %(option_id)s, %(underlying)s, %(expiration)s, %(strike_cents)s, 
                    %(option_right)s, %(multiplier)s, %(first_seen)s, %(last_seen)s, %(polygon_ticker)s
                )
                ON CONFLICT (option_id) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen,
                    underlying = EXCLUDED.underlying,
                    expiration = EXCLUDED.expiration,
                    strike_cents = EXCLUDED.strike_cents,
                    option_right = EXCLUDED.option_right,
                    multiplier = EXCLUDED.multiplier,
                    polygon_ticker = EXCLUDED.polygon_ticker
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
    
    def fetch_and_store_eod_data(self, contracts: List[Dict[str, Any]], as_of_date: str) -> int:
        """
        Fetch and store EOD data for a list of contracts after they've been stored in the database
        
        Args:
            contracts: List of contracts that have already been stored
            as_of_date: Date to fetch EOD data for (YYYY-MM-DD format)
        
        Returns:
            Number of EOD prices successfully stored
        """
        if not contracts:
            return 0
        
        logger.info(f"Fetching EOD data for {len(contracts)} contracts as of {as_of_date}")
        
        historical_prices = []
        seen = set()

        for contract in contracts:
            ticker = contract.get('polygon_ticker')
            # Avoid duplicate API calls if the same ticker appears multiple times
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)

            try:
                # Fetch EOD data using the proper endpoint
                eod_data = self.polygon_client.get_option_eod(
                    option_ticker=ticker,
                    date=as_of_date
                )
                
                # Log the response structure for debugging
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"EOD response for {ticker}: {eod_data}")

                # Handle the response structure properly
                if not isinstance(eod_data, dict):
                    logger.debug(f"Unexpected EOD response type for {ticker}: {type(eod_data)}")
                    continue
                
                # Check if we have actual data (aggregates endpoint returns results array)
                if not eod_data or not isinstance(eod_data, dict):
                    logger.debug(f"No EOD data available for {ticker} as of {as_of_date}")
                    continue
                
                # Extract results from aggregates response
                results = eod_data.get('results', [])
                if not results or len(results) == 0:
                    logger.debug(f"No EOD data available for {ticker} as of {as_of_date}")
                    continue
                
                # Take the first (and should be only) result
                bar_data = results[0]
                
                # Extract EOD data from the bar
                open_price = bar_data.get('o')  # aggregates uses 'o' for open
                high_price = bar_data.get('h')  # aggregates uses 'h' for high
                low_price = bar_data.get('l')   # aggregates uses 'l' for low
                close_price = bar_data.get('c') # aggregates uses 'c' for close
                volume = bar_data.get('v')      # aggregates uses 'v' for volume
                vwap = bar_data.get('vw')      # aggregates uses 'vw' for vwap
                transactions = bar_data.get('n') # aggregates uses 'n' for number of transactions

                # Keep a record if we have at least some pricing data
                if any(price is not None for price in [open_price, high_price, low_price, close_price]):
                    historical_prices.append({
                        'option_id': contract['option_id'],
                        'as_of': as_of_date,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'vwap': vwap,
                        'transactions': transactions
                    })
                    
                    logger.debug(f"Successfully extracted EOD data for {ticker}: O={open_price}, H={high_price}, L={low_price}, C={close_price}, V={volume}")
                else:
                    logger.debug(f"No pricing data found for {ticker} as of {as_of_date}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch EOD data for {ticker}: {e}")
                # Log more details for debugging
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    logger.debug(f"EOD error details for {ticker}: {traceback.format_exc()}")
                continue
        
        # Store the historical EOD prices if we have any
        if historical_prices:
            self.upsert_option_eod_prices(historical_prices)
            logger.info(f"Stored {len(historical_prices)} historical EOD prices for {as_of_date}")
        else:
            logger.info(f"No historical EOD prices found for {as_of_date}")
        
        # Log summary of EOD processing
        total_attempted = len(seen)
        total_successful = len(historical_prices)
        total_skipped = total_attempted - total_successful
        logger.info(f"EOD summary for {as_of_date}: {total_successful} successful, {total_skipped} skipped out of {total_attempted} attempted")
        
        return total_successful

    def upsert_option_eod_prices(self, rows):
        """
        Upsert option EOD prices into the database
        
        Args:
            rows: List of EOD price dictionaries
        
        Returns:
            Number of EOD prices processed
        """
        if not rows:
            return 0
        
        sql = """
        INSERT INTO option_eod_prices (option_id, as_of, open, high, low, close, volume, vwap, transactions)
        VALUES (%(option_id)s, %(as_of)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(vwap)s, %(transactions)s)
        ON CONFLICT (option_id, as_of) DO UPDATE SET
            open = EXCLUDED.open, 
            high = EXCLUDED.high, 
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vwap = EXCLUDED.vwap,
            transactions = EXCLUDED.transactions
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for r in rows:
                    cur.execute(sql, r)
            conn.commit()
        
        logger.info(f"Successfully processed {len(rows)} option EOD prices")
        return len(rows)
    
    def backfill_contracts(self, underlying: str = 'QQQ', days_back: int = 730, 
                          sample_rate: int = 1, max_dates_per_run: int = None, 
                          lookback_days: int = 365) -> Dict[str, Any]:
        """
        Backfill contracts for the last N days with continuous processing and rate limit respect
        
        Args:
            underlying: Underlying symbol
            days_back: Number of days to go back
            sample_rate: Only process every Nth date to respect rate limits
            max_dates_per_run: Maximum dates to process in one run (None = all dates)
            lookback_days: Days to look forward from discovery date to find expiration (default: 365 for LEAPS)
        
        Returns:
            Summary of backfill operation
        """
        logger.info(f"Starting backfill for {underlying} over last {days_back} days")
        logger.info(f"Using sample rate of {sample_rate} (every {sample_rate}th date)")
        if max_dates_per_run:
            logger.info(f"Processing max {max_dates_per_run} dates per run")
        
        # Get historical expiration dates
        all_dates = self.get_historical_expiration_dates(days_back)
        logger.info(f"Generated {len(all_dates)} historical expiration dates")
        
        if not all_dates:
            logger.error("No historical expiration dates generated. Cannot proceed with backfill.")
            return {
                'total_contracts': 0,
                'processed_dates': 0,
                'failed_dates': 0,
                'total_dates_checked': 0,
                'underlying': underlying,
                'days_back': days_back,
                'sample_rate': sample_rate,
                'max_dates_per_run': max_dates_per_run,
                'error': 'No historical dates generated'
            }
        
        # Sample dates to respect rate limits
        sampled_dates = all_dates[::sample_rate]
        logger.info(f"Processing {len(sampled_dates)} out of {len(all_dates)} dates")
        
        # Log the date range being processed
        if sampled_dates:
            logger.info(f"Date range: {sampled_dates[0]} to {sampled_dates[-1]}")
            logger.debug(f"First 5 dates: {sampled_dates[:5]}")
            logger.debug(f"Last 5 dates: {sampled_dates[-5:]}")
        
        # Limit dates per run if specified
        if max_dates_per_run:
            sampled_dates = sampled_dates[:max_dates_per_run]
            logger.info(f"Limited to {len(sampled_dates)} dates for this run")
        
        total_contracts = 0
        processed_dates = 0
        failed_dates = 0
        
        for i, discovery_date in enumerate(sampled_dates):
            try:
                logger.info(f"Processing date {i+1}/{len(sampled_dates)}: {discovery_date}")
                
                # For historical dates, we want to discover contracts that expire in the future
                # but were available for trading on the discovery date (typically 1 year later for LEAPS)
                # Set as_of to be lookback_days after discovery to find contracts that expire then
                as_of_date = self._find_next_friday_expiration(discovery_date, lookback_days)
                logger.info(f"Discovering contracts expiring on {as_of_date} as they existed on {discovery_date} (lookback: {lookback_days} days)")
                
                contracts = self.discover_contracts_for_date(
                    underlying, 
                    discovery_date,  # discovery date
                    as_of=as_of_date,  # future expiration date
                    expired=True  # Get contracts that existed on the discovery date
                )
                if contracts:
                    processed = self.upsert_contracts(contracts)
                    total_contracts += processed
                    processed_dates += 1
                    
                    # Now fetch and store EOD data for these contracts
                    logger.info(f"Fetching EOD data for {discovery_date}...")
                    eod_processed = self.fetch_and_store_eod_data(contracts, discovery_date)
                    logger.info(f"EOD data processing complete: {eod_processed} prices stored for {discovery_date}")
                else:
                    logger.warning(f"No contracts found for {discovery_date}")
                
                # Add delay between requests to be extra respectful of rate limits
                if i < len(sampled_dates) - 1:  # Don't sleep after the last request
                    logger.info(f"Waiting 3 seconds before next date...")
                    time.sleep(3)  # 3 second delay between dates
                
            except Exception as e:
                logger.error(f"Error processing discovery date {discovery_date}: {e}")
                failed_dates += 1
                continue
        
        summary = {
            'total_contracts': total_contracts,
            'processed_dates': processed_dates,
            'failed_dates': failed_dates,
            'total_dates_checked': len(sampled_dates),
            'underlying': underlying,
            'days_back': days_back,
            'sample_rate': sample_rate,
            'max_dates_per_run': max_dates_per_run
        }
        
        logger.info(f"Backfill completed. Summary: {summary}")
        return summary
    
    def backfill_contracts_continuous(self, underlying: str = 'QQQ', days_back: int = 730, 
                                    sample_rate: int = 1, dates_per_batch: int = 10,
                                    delay_between_batches: int = 60, lookback_days: int = 365) -> Dict[str, Any]:
        """
        Continuous backfill that processes dates in batches with delays between batches
        
        Args:
            underlying: Underlying symbol
            days_back: Number of days to go back
            sample_rate: Only process every Nth date to respect rate limits
            dates_per_batch: Number of dates to process in each batch
            delay_between_batches: Seconds to wait between batches
            lookback_days: Days to look forward from discovery date to find expiration (default: 365 for LEAPS)
        
        Returns:
            Summary of backfill operation
        """
        logger.info(f"Starting continuous backfill for {underlying} over last {days_back} days")
        logger.info(f"Processing {dates_per_batch} dates per batch with {delay_between_batches}s delay between batches")
        
        # Get historical expiration dates
        all_dates = self.get_historical_expiration_dates(days_back)
        logger.info(f"Generated {len(all_dates)} historical expiration dates")
        
        if not all_dates:
            logger.error("No historical expiration dates generated. Cannot proceed with backfill.")
            return {
                'total_contracts': 0,
                'processed_dates': 0,
                'failed_dates': 0,
                'total_dates_checked': 0,
                'underlying': underlying,
                'days_back': days_back,
                'sample_rate': sample_rate,
                'dates_per_batch': dates_per_batch,
                'delay_between_batches': delay_between_batches,
                'batch_count': 0,
                'error': 'No historical dates generated'
            }
        
        # Sample dates to respect rate limits
        sampled_dates = all_dates[::sample_rate]
        logger.info(f"Total dates to process: {len(sampled_dates)}")
        
        # Log the date range being processed
        if sampled_dates:
            logger.info(f"Date range: {sampled_dates[0]} to {sampled_dates[-1]}")
            logger.debug(f"First 5 dates: {sampled_dates[:5]}")
            logger.debug(f"Last 5 dates: {sampled_dates[-5:]}")
        
        total_contracts = 0
        processed_dates = 0
        failed_dates = 0
        batch_count = 0
        
        # Process dates in batches
        for i in range(0, len(sampled_dates), dates_per_batch):
            batch_count += 1
            batch_dates = sampled_dates[i:i + dates_per_batch]
            
            logger.info(f"Processing batch {batch_count}: dates {i+1}-{min(i+dates_per_batch, len(sampled_dates))} of {len(sampled_dates)}")
            
            # Process this batch
            for j, discovery_date in enumerate(batch_dates):
                try:
                    logger.info(f"Processing date {i+j+1}/{len(sampled_dates)}: {discovery_date}")
                    
                    # For historical dates, we want to discover contracts that expire in the future
                    # but were available for trading on the discovery date (typically 1 year later for LEAPS)
                    # Set as_of to be lookback_days after discovery to find contracts that expire then
                    as_of_date = self._find_next_friday_expiration(discovery_date, lookback_days)
                    logger.info(f"Discovering contracts expiring on {as_of_date} as they existed on {discovery_date} (lookback: {lookback_days} days)")
                    
                    contracts = self.discover_contracts_for_date(
                        underlying, 
                        discovery_date,  # discovery date
                        as_of=as_of_date,  # future expiration date
                        expired=True  # Get contracts that existed on the discovery date
                    )
                    if contracts:
                        processed = self.upsert_contracts(contracts)
                        total_contracts += processed
                        processed_dates += 1
                        
                        # Now fetch and store EOD data for these contracts
                        logger.info(f"Fetching EOD data for {discovery_date}...")
                        eod_processed = self.fetch_and_store_eod_data(contracts, discovery_date)
                        logger.info(f"EOD data processing complete: {eod_processed} prices stored for {discovery_date}")
                    else:
                        logger.warning(f"No contracts found for {discovery_date}")
                    
                    # Add delay between requests to be extra respectful of rate limits
                    if j < len(batch_dates) - 1:  # Don't sleep after the last date in batch
                        logger.info(f"Waiting 3 seconds before next date...")
                        time.sleep(3)  # 3 second delay between dates
                    
                except Exception as e:
                    logger.error(f"Error processing discovery date {discovery_date}: {e}")
                    failed_dates += 1
                    continue
            
            # Delay between batches (except after the last batch)
            if i + dates_per_batch < len(sampled_dates):
                logger.info(f"Batch {batch_count} completed. Waiting {delay_between_batches} seconds before next batch...")
                time.sleep(delay_between_batches)
        
        summary = {
            'total_contracts': total_contracts,
            'processed_dates': processed_dates,
            'failed_dates': failed_dates,
            'total_dates_checked': len(sampled_dates),
            'underlying': underlying,
            'days_back': days_back,
            'sample_rate': sample_rate,
            'dates_per_batch': dates_per_batch,
            'delay_between_batches': delay_between_batches,
            'batch_count': batch_count
        }
        
        logger.info(f"Continuous backfill completed. Summary: {summary}")
        return summary
    
    def close(self):
        """Close resources"""
        if self.polygon_client:
            self.polygon_client.close()


def main():
    """Main function to run the contract backfill"""
    parser = argparse.ArgumentParser(
        description='Backfill historical QQQ option contracts',
        epilog="""
Logging Behavior:
  - Console output: Always enabled
  - Default file: logs/options_backfill.log (always created)
  - Custom file: Optional, specified with --log-file
  - All log files receive the same level and format

Backfill Modes:
  - Standard: Process all dates in one run
  - Continuous: Process dates in batches with delays (use --continuous)
  - Rate Limited: Automatically respects Polygon's 100 req/sec limit
        """
    )
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Custom log file path (optional, will also log to default location). Examples: "my_backfill.log", "logs/custom_backfill.log", "/path/to/backfill.log"')
    parser.add_argument('--underlying', type=str, default='QQQ',
                       help='Underlying symbol (default: QQQ)')
    parser.add_argument('--days-back', type=int, default=730,
                       help='Number of days to go back (default: 730)')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Sample rate for dates (default: 1 = every date)')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous backfill with batching and delays')
    parser.add_argument('--dates-per-batch', type=int, default=10,
                       help='Number of dates to process per batch (default: 10)')
    parser.add_argument('--delay-between-batches', type=int, default=60,
                       help='Seconds to wait between batches (default: 60)')
    parser.add_argument('--max-dates-per-run', type=int, default=None,
                       help='Maximum dates to process in one run (default: all dates)')
    parser.add_argument('--test-date', type=str, default=None,
                       help='Test a single specific date (YYYY-MM-DD format) for debugging')
    parser.add_argument('--lookback-days', type=int, default=365,
                       help='Days to look forward from discovery date to find expiration (default: 365 for LEAPS)')
    
    args = parser.parse_args()
    
    try:
        # Setup logging with CLI level and optional custom log file
        global logger
        logger = setup_logging(args.log_level, args.log_file)
        
        logger.info("Starting QQQ options contract backfill")
        logger.info(f"Log level: {args.log_level}")
        if args.log_file:
            logger.info(f"Custom log file: {args.log_file}")
        logger.info(f"Parameters: underlying={args.underlying}, days_back={args.days_back}, sample_rate={args.sample_rate}")
        
        if args.continuous:
            logger.info(f"Continuous mode: {args.dates_per_batch} dates per batch, {args.delay_between_batches}s delay")
        if args.max_dates_per_run:
            logger.info(f"Limited to {args.max_dates_per_run} dates per run")
        
        backfiller = OptionsContractBackfiller()
        
        # Handle test date option for debugging
        if args.test_date:
            logger.info(f"Testing single date: {args.test_date}")
            try:
                # Test contract discovery for this specific date
                # For LEAPS contracts, we want to discover contracts that expire in the future
                # but were available for trading on the test date (based on lookback_days)
                as_of_date = backfiller._find_next_friday_expiration(args.test_date, args.lookback_days)
                logger.info(f"Testing: Discovering contracts expiring on {as_of_date} as they existed on {args.test_date} (lookback: {args.lookback_days} days)")
                
                contracts = backfiller.discover_contracts_for_date(
                    args.underlying, 
                    args.test_date,  # discovery date
                    as_of=as_of_date,  # future expiration date
                    expired=True
                )
                
                if contracts:
                    logger.info(f"Found {len(contracts)} contracts for {args.test_date}")
                    # Store the contracts
                    processed = backfiller.upsert_contracts(contracts)
                    logger.info(f"Stored {processed} contracts in database")
                    
                    # Now fetch and store EOD data for these contracts
                    logger.info(f"Fetching EOD data for {args.test_date}...")
                    eod_processed = backfiller.fetch_and_store_eod_data(contracts, args.test_date)
                    logger.info(f"EOD data processing complete: {eod_processed} prices stored for {args.test_date}")
                    
                else:
                    logger.warning(f"No contracts found for {args.test_date}")
                
                return
                
            except Exception as e:
                logger.error(f"Test date processing failed: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Choose backfill method based on CLI parameters
        if args.continuous:
            # Continuous backfill with batching
            summary =                 backfiller.backfill_contracts_continuous(
                    underlying=args.underlying, 
                    days_back=args.days_back, 
                    sample_rate=args.sample_rate,
                    dates_per_batch=args.dates_per_batch,
                    delay_between_batches=args.delay_between_batches,
                    lookback_days=args.lookback_days
                )
        else:
            # Standard backfill
            summary = backfiller.backfill_contracts(
                underlying=args.underlying, 
                days_back=args.days_back, 
                sample_rate=args.sample_rate,
                max_dates_per_run=args.max_dates_per_run,
                lookback_days=args.lookback_days
            )
        
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
