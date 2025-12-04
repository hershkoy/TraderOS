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
            # If we already have a connection, check if it's still valid
            if self.connection:
                try:
                    # Test the connection with a simple query
                    cursor = self.connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.close()
                    logger.debug("Existing connection is still valid")
                    return True
                except Exception:
                    logger.info("Existing connection is invalid, creating new connection")
                    self.connection = None
            
            # Create new connection
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                # Add timeout settings to prevent hanging
                connect_timeout=10,
                options='-c statement_timeout=300000'  # 5 minutes timeout
            )
            logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            self.connection = None
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                logger.debug("Disconnected from TimescaleDB")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self.connection = None
    
    def ensure_connection(self):
        """Ensure we have a valid connection, reconnect if necessary"""
        if not self.connection:
            return self.connect()
        
        try:
            # Test the connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            logger.info("Connection lost, reconnecting...")
            return self.connect()
    
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
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            symbol: Stock symbol
            provider: Data provider (ALPACA, IB, etc.)
            timeframe: Time interval (1h, 1d, etc.)
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Create a fresh connection for this operation to avoid conflicts
        fresh_connection = None
        cursor = None
        
        try:
            # Create a new connection specifically for this operation
            fresh_connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10,
                options='-c statement_timeout=600000'  # 10 minutes timeout
            )
            
            cursor = fresh_connection.cursor()
            
            # Set a reasonable timeout for the entire operation
            cursor.execute("SET statement_timeout = '600000'")  # 10 minutes
            
            # Prepare data for insertion
            logger.info(f"Preparing {len(df)} records for insertion...")
            data_to_insert = []
            for idx, (_, row) in enumerate(df.iterrows()):
                try:
                    # Validate data before insertion
                    # Convert ts_event (nanoseconds) back to timestamp
                    ts_event = int(row['ts_event'])
                    ts = pd.Timestamp(ts_event, unit='ns').isoformat()
                    open_price = float(row['open'])
                    high_price = float(row['high'])
                    low_price = float(row['low'])
                    close_price = float(row['close'])
                    volume = int(row['volume'])
                    
                    # Basic validation
                    if not all(isinstance(x, (int, float)) and x >= 0 for x in [open_price, high_price, low_price, close_price, volume]):
                        logger.warning(f"Skipping invalid record at index {idx}: {row.to_dict()}")
                        continue
                    
                    if high_price < low_price:
                        logger.warning(f"Skipping record with high < low at index {idx}: high={high_price}, low={low_price}")
                        continue
                    
                    data_to_insert.append((
                        ts,
                        symbol.upper(),
                        provider.upper(),
                        timeframe,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume
                    ))
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid record at index {idx}: {e}")
                    continue
                
                # Log progress every 1000 records
                if (idx + 1) % 1000 == 0:
                    logger.info(f"Prepared {idx + 1}/{len(df)} records...")
            
            logger.info(f"Data preparation complete. Starting batch insertion...")
            
            # Use larger batches since COPY is much more efficient
            batch_size = 2000  # Increased since COPY is much faster
            total_inserted = 0
            
            total_batches = (len(data_to_insert) + batch_size - 1) // batch_size
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i + batch_size]
                current_batch = i//batch_size + 1
                logger.info(f"Inserting batch {current_batch}/{total_batches} ({len(batch)} records)...")
                
                try:
                    logger.info(f"Inserting batch {i//batch_size + 1} using COPY...")
                    
                    # Ensure cursor is fresh for each batch
                    if cursor:
                        cursor.close()
                    cursor = fresh_connection.cursor()
                    
                    # Use COPY for much faster bulk insertion
                    from io import StringIO
                    import csv
                    
                    # Create a CSV-like string buffer
                    buffer = StringIO()
                    writer = csv.writer(buffer)
                    
                    # Write data to buffer
                    for record in batch:
                        writer.writerow(record)
                    
                    buffer.seek(0)
                    
                    # Use COPY FROM STDIN for fast bulk insertion
                    # Add timeout to prevent hanging
                    cursor.execute("SET statement_timeout = '300000'")  # 5 minutes
                    
                    cursor.copy_from(
                        buffer,
                        'market_data',
                        columns=('ts', 'symbol', 'provider', 'timeframe', 'open', 'high', 'low', 'close', 'volume'),
                        sep=','
                    )
                    
                    logger.info(f"Batch {i//batch_size + 1} completed successfully using COPY")
                    total_inserted += len(batch)
                    logger.info(f"Batch {i//batch_size + 1} inserted successfully. Total: {total_inserted}/{len(data_to_insert)}")
                    
                except Exception as batch_error:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {batch_error}")
                    logger.error(f"First few records in batch: {batch[:3] if batch else 'No batch data'}")
                    
                    # Try fallback to regular INSERT if COPY fails
                    logger.info(f"Trying fallback INSERT for batch {i//batch_size + 1}...")
                    try:
                        # IMPORTANT: Reset cursor state after COPY failure
                        # Close the corrupted cursor and create a new one
                        cursor.close()
                        cursor = fresh_connection.cursor()
                        
                        # Use regular INSERT as fallback
                        placeholders = ','.join(['%s'] * 9)  # 9 columns
                        insert_query = f"""
                            INSERT INTO market_data (ts, symbol, provider, timeframe, open, high, low, close, volume)
                            VALUES ({placeholders})
                            ON CONFLICT (symbol, ts) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume
                        """
                        
                        cursor.executemany(insert_query, batch)
                        logger.info(f"Fallback INSERT successful for batch {i//batch_size + 1}")
                        total_inserted += len(batch)
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback INSERT also failed for batch {i//batch_size + 1}: {fallback_error}")
                        # Don't raise the original batch_error, just log and continue
                        logger.error(f"Both COPY and INSERT failed for batch {i//batch_size + 1}, skipping...")
                        continue
            
            logger.info(f"Committing transaction...")
            fresh_connection.commit()
            logger.info(f"Successfully inserted {total_inserted} records for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert market data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            if fresh_connection and not fresh_connection.closed:
                try:
                    fresh_connection.rollback()
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")
            return False
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if fresh_connection:
                try:
                    fresh_connection.close()
                except Exception:
                    pass
    
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
        if not self.ensure_connection():
            logger.error("No database connection")
            return None
        
        try:
            cursor = self.connection.cursor()
            
            # Build query
            query = """
                SELECT 
                    ts,
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
                query += " AND ts >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND ts <= %s"
                params.append(end_time)
            
            query += " ORDER BY ts"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(rows)} raw rows from database")
            logger.info(f"Sample raw row: {rows[0] if rows else 'No rows'}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=[
                'ts', 'symbol', 'provider', 'timeframe',
                'open', 'high', 'low', 'close', 'volume', 'created_at'
            ])
            
            logger.info(f"DataFrame created with shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame data types: {df.dtypes.to_dict()}")
            logger.info(f"Sample data (first 3 rows):")
            logger.info(f"  {df.head(3).to_dict('records')}")
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['ts'], utc=True)
            logger.info(f"Timestamp conversion completed")
            logger.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
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
                    MIN(ts) as min_date,
                    MAX(ts) as max_date
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
                query += " AND ts < %s"
                params.append(before_date)
            
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
    
    def reset_database(self) -> bool:
        """
        Reset the database by dropping and recreating the market_data table
        with proper TIMESTAMPTZ structure
        """
        if not self.connection:
            logger.error("No database connection")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Drop existing table and related objects
            logger.info("Dropping existing market_data table...")
            cursor.execute("DROP TABLE IF EXISTS market_data CASCADE")
            cursor.execute("DROP VIEW IF EXISTS market_data_view CASCADE")
            cursor.execute("DROP FUNCTION IF EXISTS get_market_data CASCADE")
            cursor.execute("DROP FUNCTION IF EXISTS insert_market_data CASCADE")
            
            # Recreate the table with proper TIMESTAMPTZ structure
            logger.info("Creating new market_data table with TIMESTAMPTZ...")
            cursor.execute("""
                CREATE TABLE market_data (
                    ts TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    provider VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    open DECIMAL(15,6) NOT NULL,
                    high DECIMAL(15,6) NOT NULL,
                    low DECIMAL(15,6) NOT NULL,
                    close DECIMAL(15,6) NOT NULL,
                    volume BIGINT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create hypertable for time-series data with proper time partitioning
            logger.info("Creating hypertable...")
            cursor.execute("""
                SELECT create_hypertable('market_data', 'ts', chunk_time_interval => INTERVAL '7 days')
            """)
            
            # Create indexes for optimal query performance
            logger.info("Creating indexes...")
            cursor.execute("""
                CREATE INDEX idx_market_data_symbol_ts ON market_data (symbol, ts DESC)
            """)
            cursor.execute("""
                CREATE INDEX idx_market_data_provider_ts ON market_data (provider, ts DESC)
            """)
            cursor.execute("""
                CREATE INDEX idx_market_data_timeframe_ts ON market_data (timeframe, ts DESC)
            """)
            cursor.execute("""
                CREATE INDEX idx_market_data_symbol_provider_timeframe_ts ON market_data (symbol, provider, timeframe, ts DESC)
            """)
            
            # Create a view for easier data access
            logger.info("Creating view...")
            cursor.execute("""
                CREATE VIEW market_data_view AS
                SELECT 
                    ts as timestamp,
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
                ORDER BY ts
            """)
            
            # Create functions
            logger.info("Creating functions...")
            cursor.execute("""
                CREATE OR REPLACE FUNCTION get_market_data(
                    p_symbol VARCHAR(20),
                    p_timeframe VARCHAR(10),
                    p_provider VARCHAR(20) DEFAULT NULL,
                    p_start_time TIMESTAMPTZ DEFAULT NULL,
                    p_end_time TIMESTAMPTZ DEFAULT NULL
                )
                RETURNS TABLE (
                    ts_out TIMESTAMPTZ,
                    symbol VARCHAR(20),
                    provider VARCHAR(20),
                    timeframe VARCHAR(10),
                    open DECIMAL(15,6),
                    high DECIMAL(15,6),
                    low DECIMAL(15,6),
                    close DECIMAL(15,6),
                    volume BIGINT
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        md.ts as ts_out,
                        md.symbol,
                        md.provider,
                        md.timeframe,
                        md.open,
                        md.high,
                        md.low,
                        md.close,
                        md.volume
                    FROM market_data md
                    WHERE md.symbol = p_symbol
                      AND md.timeframe = p_timeframe
                      AND (p_provider IS NULL OR md.provider = p_provider)
                      AND (p_start_time IS NULL OR md.ts >= p_start_time)
                      AND (p_end_time IS NULL OR md.ts <= p_end_time)
                    ORDER BY md.ts;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            cursor.execute("""
                CREATE OR REPLACE FUNCTION insert_market_data(
                    p_ts TIMESTAMPTZ,
                    p_symbol VARCHAR(20),
                    p_provider VARCHAR(20),
                    p_timeframe VARCHAR(10),
                    p_open DECIMAL(15,6),
                    p_high DECIMAL(15,6),
                    p_low DECIMAL(15,6),
                    p_close DECIMAL(15,6),
                    p_volume BIGINT
                )
                RETURNS VOID AS $$
                BEGIN
                    INSERT INTO market_data (
                        ts, symbol, provider, timeframe, 
                        open, high, low, close, volume
                    ) VALUES (
                        p_ts, p_symbol, p_provider, p_timeframe,
                        p_open, p_high, p_low, p_close, p_volume
                    )
                    ON CONFLICT (ts, symbol, provider, timeframe) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        created_at = NOW();
                END;
                $$ LANGUAGE plpgsql
            """)
            
            # Grant permissions
            cursor.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO backtrader_user")
            cursor.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO backtrader_user")
            cursor.execute("GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO backtrader_user")
            
            self.connection.commit()
            logger.info("Database reset completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def execute_query(self, query: str, params: tuple = None):
        """Execute a custom query and return results"""
        if not self.ensure_connection():
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None

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

def close_timescaledb_client():
    """Close the global TimescaleDB client instance"""
    global _timescaledb_client
    
    if _timescaledb_client is not None:
        _timescaledb_client.disconnect()
        _timescaledb_client = None
        logger.info("Global TimescaleDB client closed")

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
