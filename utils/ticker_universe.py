"""
Ticker Universe Management for BackTrader Framework
Handles fetching, caching, and managing ticker lists from various indices
"""

import pandas as pd
import psycopg2
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from .timescaledb_client import TimescaleDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CACHE_EXPIRY_HOURS = 24
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

class TickerUniverseManager:
    """Manages ticker universe with TimescaleDB backend"""
    
    def __init__(self, db_client: TimescaleDBClient = None):
        """Initialize the ticker universe manager"""
        self.db_client = db_client or TimescaleDBClient()
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure the ticker universe tables exist"""
        try:
            with self.db_client as client:
                cursor = client.connection.cursor()
                
                # Create ticker_universe table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ticker_universe (
                        index_name VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        company_name VARCHAR(255),
                        sector VARCHAR(100),
                        last_updated TIMESTAMPTZ NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        PRIMARY KEY (index_name, symbol)
                    )
                ''')
                
                # Create ticker_cache_metadata table for tracking cache status
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ticker_cache_metadata (
                        index_name VARCHAR(50) PRIMARY KEY,
                        last_fetched TIMESTAMPTZ NOT NULL,
                        ticker_count INTEGER NOT NULL,
                        cache_expiry_hours INTEGER DEFAULT 24
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ticker_universe_symbol 
                    ON ticker_universe (symbol)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ticker_universe_last_updated 
                    ON ticker_universe (last_updated)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ticker_universe_active 
                    ON ticker_universe (is_active) WHERE is_active = TRUE
                ''')
                
                client.connection.commit()
                logger.info("Ticker universe tables created/verified successfully")
                
        except Exception as e:
            logger.error(f"Failed to create ticker universe tables: {e}")
            raise
    
    def _is_cache_valid(self, index_name: str) -> bool:
        """Check if cache for given index is still valid"""
        try:
            with self.db_client as client:
                cursor = client.connection.cursor()
                cursor.execute('''
                    SELECT last_fetched, cache_expiry_hours 
                    FROM ticker_cache_metadata 
                    WHERE index_name = %s
                ''', (index_name,))
                
                row = cursor.fetchone()
                if not row:
                    return False
                
                last_fetched, expiry_hours = row
                expiry_hours = expiry_hours or CACHE_EXPIRY_HOURS
                
                # Check if cache has expired
                if datetime.now(last_fetched.tzinfo) - last_fetched > timedelta(hours=expiry_hours):
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to check cache validity for {index_name}: {e}")
            return False
    
    def _fetch_sp500_tickers(self) -> List[Dict[str, Any]]:
        """Fetch S&P 500 tickers from Wikipedia"""
        try:
            logger.info("Fetching S&P 500 tickers from Wikipedia...")
            df = pd.read_html(SP500_URL)[0]
            
            tickers = []
            for _, row in df.iterrows():
                tickers.append({
                    'symbol': row['Symbol'].replace('.', '-'),
                    'company_name': row['Security'],
                    'sector': row.get('GICS Sector', 'Unknown')
                })
            
            logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers: {e}")
            return []
    
    def _fetch_nasdaq100_tickers(self) -> List[Dict[str, Any]]:
        """Fetch NASDAQ-100 tickers from Wikipedia"""
        try:
            logger.info("Fetching NASDAQ-100 tickers from Wikipedia...")
            df = pd.read_html(NASDAQ100_URL)[4]  # Index 4 contains the ticker table
            
            tickers = []
            for _, row in df.iterrows():
                tickers.append({
                    'symbol': row['Ticker'].replace('.', '-'),
                    'company_name': row.get('Company', 'Unknown'),
                    'sector': 'Technology'  # NASDAQ-100 is tech-heavy
                })
            
            logger.info(f"Successfully fetched {len(tickers)} NASDAQ-100 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ-100 tickers: {e}")
            return []
    
    def _cache_tickers(self, index_name: str, tickers: List[Dict[str, Any]]):
        """Cache tickers in the database"""
        try:
            with self.db_client as client:
                cursor = client.connection.cursor()
                now = datetime.now()
                
                # Clear existing tickers for this index
                cursor.execute('''
                    UPDATE ticker_universe 
                    SET is_active = FALSE 
                    WHERE index_name = %s
                ''', (index_name,))
                
                # Insert new tickers
                for ticker in tickers:
                    cursor.execute('''
                        INSERT INTO ticker_universe 
                        (index_name, symbol, company_name, sector, last_updated, is_active)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (index_name, symbol) 
                        DO UPDATE SET
                            company_name = EXCLUDED.company_name,
                            sector = EXCLUDED.sector,
                            last_updated = EXCLUDED.last_updated,
                            is_active = EXCLUDED.is_active
                    ''', (
                        index_name,
                        ticker['symbol'],
                        ticker.get('company_name', 'Unknown'),
                        ticker.get('sector', 'Unknown'),
                        now,
                        True
                    ))
                
                # Update cache metadata
                cursor.execute('''
                    INSERT INTO ticker_cache_metadata 
                    (index_name, last_fetched, ticker_count, cache_expiry_hours)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (index_name) 
                    DO UPDATE SET
                        last_fetched = EXCLUDED.last_fetched,
                        ticker_count = EXCLUDED.ticker_count
                ''', (index_name, now, len(tickers), CACHE_EXPIRY_HOURS))
                
                client.connection.commit()
                logger.info(f"Cached {len(tickers)} tickers for {index_name}")
                
        except Exception as e:
            logger.error(f"Failed to cache tickers for {index_name}: {e}")
            raise
    
    def _get_cached_tickers(self, index_name: str) -> Optional[List[str]]:
        """Get cached tickers for given index"""
        try:
            with self.db_client as client:
                cursor = client.connection.cursor()
                cursor.execute('''
                    SELECT symbol 
                    FROM ticker_universe 
                    WHERE index_name = %s AND is_active = TRUE
                    ORDER BY symbol
                ''', (index_name,))
                
                rows = cursor.fetchall()
                if not rows:
                    return None
                
                return [row[0] for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get cached tickers for {index_name}: {e}")
            return None
    
    def get_sp500_tickers(self, force_refresh: bool = False) -> List[str]:
        """Get S&P 500 tickers, using cache if available"""
        if not force_refresh and self._is_cache_valid('sp500'):
            cached = self._get_cached_tickers('sp500')
            if cached:
                logger.info(f"Using cached S&P 500 tickers ({len(cached)} symbols)")
                return cached
        
        # Fetch fresh data
        tickers = self._fetch_sp500_tickers()
        if tickers:
            self._cache_tickers('sp500', tickers)
            return [t['symbol'] for t in tickers]
        
        # Fallback to cached data if fetch failed
        cached = self._get_cached_tickers('sp500')
        if cached:
            logger.warning("Using stale cached S&P 500 data due to fetch failure")
            return cached
        
        return []
    
    def get_nasdaq100_tickers(self, force_refresh: bool = False) -> List[str]:
        """Get NASDAQ-100 tickers, using cache if available"""
        if not force_refresh and self._is_cache_valid('nasdaq100'):
            cached = self._get_cached_tickers('nasdaq100')
            if cached:
                logger.info(f"Using cached NASDAQ-100 tickers ({len(cached)} symbols)")
                return cached
        
        # Fetch fresh data
        tickers = self._fetch_nasdaq100_tickers()
        if tickers:
            self._cache_tickers('nasdaq100', tickers)
            return [t['symbol'] for t in tickers]
        
        # Fallback to cached data if fetch failed
        cached = self._get_cached_tickers('nasdaq100')
        if cached:
            logger.warning("Using stale cached NASDAQ-100 data due to fetch failure")
            return cached
        
        return []
    
    def get_combined_universe(self, force_refresh: bool = False) -> List[str]:
        """Get combined universe of all tickers"""
        sp500 = self.get_sp500_tickers(force_refresh)
        nasdaq100 = self.get_nasdaq100_tickers(force_refresh)
        
        # Combine and deduplicate
        combined = list(set(sp500 + nasdaq100))
        combined.sort()  # Sort for consistent ordering
        
        logger.info(f"Combined universe contains {len(combined)} unique symbols")
        return combined
    
    def get_cached_combined_universe(self) -> List[str]:
        """Get combined universe from cache only (no fetching)"""
        sp500 = self._get_cached_tickers('sp500') or []
        nasdaq100 = self._get_cached_tickers('nasdaq100') or []
        
        combined = list(set(sp500 + nasdaq100))
        combined.sort()
        
        logger.info(f"Cached combined universe contains {len(combined)} unique symbols")
        return combined
    
    def get_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific ticker"""
        try:
            with self.db_client as client:
                cursor = client.connection.cursor()
                cursor.execute('''
                    SELECT index_name, symbol, company_name, sector, last_updated, is_active
                    FROM ticker_universe 
                    WHERE symbol = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                ''', (symbol,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'index_name': row[0],
                        'symbol': row[1],
                        'company_name': row[2],
                        'sector': row[3],
                        'last_updated': row[4],
                        'is_active': row[5]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get ticker info for {symbol}: {e}")
            return None
    
    def get_universe_stats(self) -> Dict[str, Any]:
        """Get statistics about the ticker universe"""
        try:
            with self.db_client as client:
                cursor = client.connection.cursor()
                
                # Get counts by index
                cursor.execute('''
                    SELECT index_name, COUNT(*) as count
                    FROM ticker_universe 
                    WHERE is_active = TRUE
                    GROUP BY index_name
                ''')
                
                index_counts = dict(cursor.fetchall())
                
                # Get total unique symbols
                cursor.execute('''
                    SELECT COUNT(DISTINCT symbol) as total_unique
                    FROM ticker_universe 
                    WHERE is_active = TRUE
                ''')
                
                total_unique = cursor.fetchone()[0]
                
                # Get cache status
                cursor.execute('''
                    SELECT index_name, last_fetched, ticker_count
                    FROM ticker_cache_metadata
                ''')
                
                cache_status = {}
                for row in cursor.fetchall():
                    cache_status[row[0]] = {
                        'last_fetched': row[1],
                        'ticker_count': row[2]
                    }
                
                return {
                    'total_unique_symbols': total_unique,
                    'index_counts': index_counts,
                    'cache_status': cache_status
                }
                
        except Exception as e:
            logger.error(f"Failed to get universe stats: {e}")
            return {}
    
    def refresh_all_indices(self):
        """Force refresh of all indices"""
        logger.info("Refreshing all ticker indices...")
        
        # Refresh S&P 500
        sp500_tickers = self.get_sp500_tickers(force_refresh=True)
        
        # Refresh NASDAQ-100
        nasdaq100_tickers = self.get_nasdaq100_tickers(force_refresh=True)
        
        logger.info(f"Refresh complete. S&P 500: {len(sp500_tickers)}, NASDAQ-100: {len(nasdaq100_tickers)}")
        
        return {
            'sp500': len(sp500_tickers),
            'nasdaq100': len(nasdaq100_tickers)
        }


# Convenience functions for backward compatibility
def get_sp500_tickers(force_refresh: bool = False) -> List[str]:
    """Get S&P 500 tickers"""
    manager = TickerUniverseManager()
    return manager.get_sp500_tickers(force_refresh)

def get_nasdaq100_tickers(force_refresh: bool = False) -> List[str]:
    """Get NASDAQ-100 tickers"""
    manager = TickerUniverseManager()
    return manager.get_nasdaq100_tickers(force_refresh)

def get_combined_universe(force_refresh: bool = False) -> List[str]:
    """Get combined universe of all tickers"""
    manager = TickerUniverseManager()
    return manager.get_combined_universe(force_refresh)

def get_cached_combined_universe() -> List[str]:
    """Get combined universe from cache only"""
    manager = TickerUniverseManager()
    return manager.get_cached_combined_universe()
