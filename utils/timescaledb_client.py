"""
TimescaleDB Client for BackTrader Framework
Handles database operations for market data storage and retrieval
"""

import psycopg2
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import os
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimescaleDBClient:
    """Client for TimescaleDB operations"""
    
    def __init__(self, host='localhost', port=5432, database='backtrader', 
                 user='backtrader_user', password='backtrader_password'):
        """Initialize TimescaleDB client"""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from TimescaleDB")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def insert_market_data(self, df: pd.DataFrame, symbol: str, provider: str, timeframe: str) -> bool:
        """
        Insert market data into TimescaleDB
        
        Args:
            df: DataFrame with columns: ts_event, open, high, low, close, volume
            symbol: Stock symbol
            provider: Data provider (ALPACA, IB, etc.)
            timeframe: Time interval (1h, 1d, etc.)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection:
            logger.error("No database connection")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Prepare data for insertion
            data_to_insert = []
            for _, row in df.iterrows():
                data_to_insert.append((
                    int(row['ts_event']),
                    symbol.upper(),
                    provider.upper(),
                    timeframe,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                ))
            
            # Use batch insert for better performance
            cursor.executemany("""
                INSERT INTO market_data 
                (ts_event, symbol, provider, timeframe, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ts_event, symbol, provider, timeframe) 
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    created_at = NOW()
            """, data_to_insert)
            
            self.connection.commit()
            logger.info(f"Inserted {len(data_to_insert)} records for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert market data: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def get_market_data(self, symbol: str, timeframe: str, provider: Optional[str] = None,
                       start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                       limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve market data from TimescaleDB
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval (1h, 1d, etc.)
            provider: Data provider (optional filter)
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            limit: Maximum number of records to return (optional)
        
        Returns:
            DataFrame with market data or None if error
        """
        if not self.connection:
            logger.error("No database connection")
            return None
        
        try:
            cursor = self.connection.cursor()
            
            # Build query
            query = """
                SELECT 
                    ts_event,
                    symbol,
                    provider,
                    timeframe,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    created_at
                FROM market_data
                WHERE symbol = %s AND timeframe = %s
            """
            params = [symbol.upper(), timeframe]
            
            if provider:
                query += " AND provider = %s"
                params.append(provider.upper())
            
            if start_time:
                query += " AND ts_event >= %s"
                params.append(int(start_time.timestamp() * 1_000_000_000))
            
            if end_time:
                query += " AND ts_event <= %s"
                params.append(int(end_time.timestamp() * 1_000_000_000))
            
            query += " ORDER BY ts_event"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=[
                'ts_event', 'symbol', 'provider', 'timeframe',
                'open', 'high', 'low', 'close', 'volume', 'created_at'
            ])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)
            
            logger.info(f"Retrieved {len(df)} records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve market data: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_available_symbols(self, provider: Optional[str] = None, timeframe: Optional[str] = None) -> List[str]:
        """
        Get list of available symbols in the database
        
        Args:
            provider: Filter by provider (optional)
            timeframe: Filter by timeframe (optional)
        
        Returns:
            List of available symbols
        """
        if not self.connection:
            logger.error("No database connection")
            return []
        
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT DISTINCT symbol FROM market_data"
            params = []
            
            if provider or timeframe:
                query += " WHERE"
                conditions = []
                
                if provider:
                    conditions.append("provider = %s")
                    params.append(provider.upper())
                
                if timeframe:
                    conditions.append("timeframe = %s")
                    params.append(timeframe)
                
                query += " AND ".join(conditions)
            
            query += " ORDER BY symbol"
            
            cursor.execute(query, params)
            symbols = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Found {len(symbols)} symbols in database")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of data in the database
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.connection:
            logger.error("No database connection")
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM market_data")
            total_records = cursor.fetchone()[0]
            
            # Get unique symbols
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
            unique_symbols = cursor.fetchone()[0]
            
            # Get unique providers
            cursor.execute("SELECT COUNT(DISTINCT provider) FROM market_data")
            unique_providers = cursor.fetchone()[0]
            
            # Get unique timeframes
            cursor.execute("SELECT COUNT(DISTINCT timeframe) FROM market_data")
            unique_timeframes = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("""
                SELECT 
                    MIN(to_timestamp(ts_event / 1000000000)) as min_date,
                    MAX(to_timestamp(ts_event / 1000000000)) as max_date
                FROM market_data
            """)
            date_range = cursor.fetchone()
            
            summary = {
                'total_records': total_records,
                'unique_symbols': unique_symbols,
                'unique_providers': unique_providers,
                'unique_timeframes': unique_timeframes,
                'date_range': {
                    'start': date_range[0] if date_range[0] else None,
                    'end': date_range[1] if date_range[1] else None
                }
            }
            
            logger.info(f"Database summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
    
    def delete_data(self, symbol: Optional[str] = None, provider: Optional[str] = None,
                   timeframe: Optional[str] = None, before_date: Optional[datetime] = None) -> int:
        """
        Delete data from the database
        
        Args:
            symbol: Delete data for specific symbol (optional)
            provider: Delete data for specific provider (optional)
            timeframe: Delete data for specific timeframe (optional)
            before_date: Delete data before this date (optional)
        
        Returns:
            Number of deleted records
        """
        if not self.connection:
            logger.error("No database connection")
            return 0
        
        try:
            cursor = self.connection.cursor()
            
            query = "DELETE FROM market_data WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol.upper())
            
            if provider:
                query += " AND provider = %s"
                params.append(provider.upper())
            
            if timeframe:
                query += " AND timeframe = %s"
                params.append(timeframe)
            
            if before_date:
                query += " AND ts_event < %s"
                params.append(int(before_date.timestamp() * 1_000_000_000))
            
            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            self.connection.commit()
            
            logger.info(f"Deleted {deleted_count} records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()

# Global client instance
_timescaledb_client = None

def get_timescaledb_client() -> TimescaleDBClient:
    """Get or create a global TimescaleDB client instance"""
    global _timescaledb_client
    
    if _timescaledb_client is None:
        # Load configuration from environment variables
        host = os.getenv('TIMESCALEDB_HOST', 'localhost')
        port = int(os.getenv('TIMESCALEDB_PORT', '5432'))
        database = os.getenv('TIMESCALEDB_DATABASE', 'backtrader')
        user = os.getenv('TIMESCALEDB_USER', 'backtrader_user')
        password = os.getenv('TIMESCALEDB_PASSWORD', 'backtrader_password')
        
        _timescaledb_client = TimescaleDBClient(host, port, database, user, password)
    
    return _timescaledb_client

def test_connection() -> bool:
    """Test TimescaleDB connection"""
    try:
        with get_timescaledb_client() as client:
            if client.connect():
                summary = client.get_data_summary()
                logger.info("TimescaleDB connection test successful")
                return True
            else:
                logger.error("TimescaleDB connection test failed")
                return False
    except Exception as e:
        logger.error(f"TimescaleDB connection test error: {e}")
        return False
