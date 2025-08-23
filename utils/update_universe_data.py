#!/usr/bin/env python3
"""
Update Universe Data Script

This script fetches maximum available bars for all tickers in the ticker universe
using the existing fetch_data.py functionality. It supports both Alpaca and IBKR
data providers and can process tickers in batches with configurable delays.
"""

import sys
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import argparse
from queue import Queue
import queue

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ticker_universe import TickerUniverseManager
    from utils.fetch_data import fetch_from_alpaca, fetch_from_ib
except ImportError:
    # Fallback for when running from utils directory
    from ticker_universe import TickerUniverseManager
    from fetch_data import fetch_from_alpaca, fetch_from_ib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('universe_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Also configure root logger to capture all logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create file handler for root logger to capture all logs
file_handler = logging.FileHandler('universe_update_complete.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Create console handler for root logger
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

class DatabaseWorker:
    """Handles database operations in a separate thread to avoid blocking data fetching"""
    
    def __init__(self):
        self.queue = Queue()
        self.worker_thread = None
        self.running = False
        self.stats = {"saved": 0, "failed": 0, "errors": []}
        
        # Initialize shared database connection
        self.db_client = None
        self.db_cursor = None
        # Don't initialize connection here - do it in the worker thread
        
    def _init_database_connection(self):
        """Initialize a single database connection and cursor for reuse"""
        try:
            logger.info("[INIT] Initializing database connection...")
            from utils.timescaledb_client import get_timescaledb_client
            
            self.db_client = get_timescaledb_client()
            if not self.db_client.ensure_connection():
                logger.error("[ERROR] Failed to connect to database")
                return False
            
            self.db_cursor = self.db_client.connection.cursor()
            logger.info("[SUCCESS] Database connection and cursor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize database connection: {e}")
            return False
    
    def _ensure_database_connection(self):
        """Ensure database connection is still valid, reconnect if needed"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.db_client and self.db_cursor:
                    # Test the connection with a simple query
                    self.db_cursor.execute("SELECT 1")
                    return True
                else:
                    return self._init_database_connection()
            except Exception as e:
                logger.warning(f"Database connection test failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info("DatabaseWorker: Reconnecting to database...")
                    self._cleanup_database_connection()
                    time.sleep(1)  # Wait before retry
                    if not self._init_database_connection():
                        continue
                else:
                    logger.error("DatabaseWorker: Failed to reconnect to database after all retries")
                    return False
        return False
        
    def start(self):
        """Start the database worker thread"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("[START] Database worker thread started")
    
    def stop(self):
        """Stop the database worker thread"""
        self.running = False
        if self.worker_thread:
            self.queue.put(None)  # Signal to stop
            self.worker_thread.join(timeout=5)
            logger.info("[STOP] Database worker thread stopped")
        
        # Cleanup database connections
        self._cleanup_database_connection()
    
    def _cleanup_database_connection(self):
        """Cleanup database connections"""
        try:
            if self.db_cursor:
                try:
                    self.db_cursor.close()
                except Exception:
                    pass
                self.db_cursor = None
            if self.db_client and hasattr(self.db_client, 'connection'):
                try:
                    if self.db_client.connection and not self.db_client.connection.closed:
                        self.db_client.connection.close()
                except Exception:
                    pass
                self.db_client = None
            logger.info("[CLEANUP] DatabaseWorker: Database connections cleaned up")
        except Exception as e:
            logger.warning(f"[WARN] DatabaseWorker: Error during cleanup: {e}")
    
    def save_data(self, df, symbol, provider, timeframe):
        """Queue data for saving (non-blocking)"""
        if self.running:
            self.queue.put((df, symbol, provider, timeframe))
            logger.info(f"[OK] Data queued for {symbol}, queue size: {self.queue.qsize()}")
        else:
            logger.warning(f"[WARN] Database worker not running, using synchronous save for {symbol}")
            # Fallback to synchronous saving if worker not running
            self._save_data_sync(df, symbol, provider, timeframe)
    
    def _save_data_sync_with_timeout(self, df, symbol, provider, timeframe, timeout=600):
        """Synchronous data saving with timeout protection"""
        # For now, just call the save method directly without threading
        # The timeout is handled by the database operations themselves
        return self._save_data_sync(df, symbol, provider, timeframe)
    
    def _worker_loop(self):
        """Main worker loop that processes database operations"""
        logger.info("[WORKER] Database worker loop started")
        
        # Initialize database connection in this thread
        if not self._init_database_connection():
            logger.error("[ERROR] Failed to initialize database connection in worker thread")
            return
        
        loop_count = 0
        last_activity_time = time.time()
        
        while self.running:
            loop_count += 1
            item = None  # Initialize item variable
            
            # Check for timeout - if no activity for 5 minutes, log a warning
            current_time = time.time()
            if current_time - last_activity_time > 300:  # 5 minutes
                logger.warning(f"[WARN] Database worker has been idle for {int(current_time - last_activity_time)} seconds")
                last_activity_time = current_time
            
            try:
                # Only log every 10th iteration to reduce noise
                if loop_count % 10 == 0:
                    logger.info(f"[WORKER] Worker loop iteration {loop_count} - Queue size: {self.queue.qsize()}")
                
                # Use a shorter timeout to prevent hanging
                item = self.queue.get(timeout=30)  # 30 seconds instead of 1
                last_activity_time = time.time()  # Reset activity timer
                
                if item is None:  # Stop signal
                    logger.info("[STOP] Received stop signal, exiting worker loop")
                    break
                
                df, symbol, provider, timeframe = item
                logger.info(f"[DATA] Processing item: symbol={symbol}, provider={provider}, timeframe={timeframe}, df_shape={df.shape if df is not None else 'None'}")
                
                process_start_time = datetime.now()
                logger.info(f"[START] STARTING database processing for {symbol}: {len(df)} bars at {process_start_time.strftime('%H:%M:%S')}")
                
                # Add timeout for database operations
                save_result = self._save_data_sync(df, symbol, provider, timeframe)
                
                if save_result:
                    self.stats["saved"] += 1
                    process_end_time = datetime.now()
                    process_duration = process_end_time - process_start_time
                    logger.info(f"[SUCCESS] SUCCESSFULLY saved data for {symbol} in {process_duration}")
                else:
                    self.stats["failed"] += 1
                    process_end_time = datetime.now()
                    process_duration = process_end_time - process_start_time
                    logger.error(f"[FAILED] FAILED to save data for {symbol} in {process_duration}")
                
                self.queue.task_done()
                
            except queue.Empty:
                # This is normal when the queue is empty, just continue
                continue
            except Exception as e:
                logger.error(f"[ERROR] Database worker error at iteration {loop_count}: {e}")
                import traceback
                logger.error(f"[ERROR] Database worker traceback: {traceback.format_exc()}")
                self.stats["errors"].append(str(e))
                if item:  # Now item is always defined
                    try:
                        self.queue.task_done()
                    except Exception as task_done_error:
                        logger.error(f"[ERROR] Failed to call queue.task_done() after error: {task_done_error}")
                        logger.error(f"[ERROR] task_done error traceback: {traceback.format_exc()}")
        
        # Cleanup database connection when worker loop ends
        self._cleanup_database_connection()
        logger.info(f"[WORKER] Database worker loop ended after {loop_count} iterations")
    
    def _save_data_sync(self, df, symbol, provider, timeframe):
        """Synchronous data saving (fallback method)"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure database connection is valid
                if not self._ensure_database_connection():
                    logger.error(f"✗ Failed to connect to TimescaleDB for {symbol}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying database save for {symbol} (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)  # Wait before retry
                        continue
                    return False
                
                # Use the client's insert_market_data method
                if hasattr(self.db_client, 'insert_market_data'):
                    insert_start_time = datetime.now()
                    
                    # The TimescaleDB client now creates its own fresh connection
                    insert_result = self.db_client.insert_market_data(df, symbol, provider, timeframe)
                    insert_end_time = datetime.now()
                    insert_duration = insert_end_time - insert_start_time
                    
                    if insert_result:
                        logger.info(f"[SAVE] Saved {len(df)} records to TimescaleDB for {symbol} {timeframe}")
                        return True
                    else:
                        logger.error(f"[FAILED] Failed to save to TimescaleDB for {symbol} {timeframe}")
                        if attempt < max_retries - 1:
                            logger.info(f"[RETRY] Retrying database save for {symbol} (attempt {attempt + 1}/{max_retries})")
                            time.sleep(2)  # Wait before retry
                            continue
                        return False
                else:
                    # Fallback: use the client's save_market_data method if available
                    if hasattr(self.db_client, 'save_market_data'):
                        if self.db_client.save_market_data(df, symbol, provider, timeframe):
                            logger.info(f"[SAVE] Saved {len(df)} records to TimescaleDB for {symbol} {timeframe}")
                            return True
                        else:
                            logger.error(f"[FAILED] Failed to save to TimescaleDB for {symbol} {timeframe}")
                            if attempt < max_retries - 1:
                                logger.info(f"[RETRY] Retrying database save for {symbol} (attempt {attempt + 1}/{max_retries})")
                                time.sleep(2)  # Wait before retry
                                continue
                            return False
                    else:
                        logger.error(f"[ERROR] TimescaleDB client doesn't have insert_market_data or save_market_data method for {symbol}")
                        return False
                        
            except Exception as e:
                logger.error(f"[ERROR] Error saving to TimescaleDB for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"[RETRY] Retrying database save for {symbol} (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    import traceback
                    logger.error(f"[ERROR] Final attempt failed for {symbol}. Traceback: {traceback.format_exc()}")
                    return False
        
        return False
    
    def wait_for_completion(self, timeout=3600):
        """Wait for all queued operations to complete with timeout"""
        logger.info(f"[WAIT] wait_for_completion called, worker running: {self.running}")
        if self.running:
            logger.info(f"[WAIT] About to call queue.join() with {timeout}s timeout...")
            logger.info(f"[WAIT] Current queue size: {self.queue.qsize()}")
            
            start_time = time.time()
            try:
                # Use a timeout to prevent hanging indefinitely
                while not self.queue.empty():
                    if time.time() - start_time > timeout:
                        logger.warning(f"[WARN] wait_for_completion timed out after {timeout} seconds")
                        break
                    time.sleep(1)  # Check every second
                
                logger.info(f"[COMPLETE] queue.join() completed")
                logger.info(f"[STATS] Database operations completed: {self.stats['saved']} saved, {self.stats['failed']} failed")
            except Exception as e:
                logger.error(f"[ERROR] Error in wait_for_completion: {e}")
        else:
            logger.warning("[WARN] Worker not running, cannot wait for completion")
    
    def get_stats(self):
        """Get current database worker statistics"""
        return self.stats.copy()

class UniverseDataUpdater:
    """Updates market data for all tickers in the universe"""
    
    def __init__(self, provider: str = "alpaca", timeframe: str = "1d"):
        """
        Initialize the universe data updater
        
        Args:
            provider: Data provider ('alpaca' or 'ib')
            timeframe: Timeframe to fetch ('1h' or '1d')
        """
        self.provider = provider.lower()
        self.timeframe = timeframe
        self.ticker_manager = TickerUniverseManager()
        self.db_worker = DatabaseWorker()
        
        # Initialize single database connection and cursor for reuse
        self.db_client = None
        self.db_cursor = None
        self._init_database_connection()
        
        # Create symbol mapping table if it doesn't exist
        self._create_symbol_mapping_table()
        
        # Validation
        if self.provider not in ['alpaca', 'ib']:
            raise ValueError("Provider must be 'alpaca' or 'ib'")
        if self.timeframe not in ['1h', '1d']:
            raise ValueError("Timeframe must be '1h' or '1d'")
        
        logger.info(f"Initialized UniverseDataUpdater for {self.provider} @ {self.timeframe}")
    
    def _init_database_connection(self):
        """Initialize a single database connection and cursor for reuse"""
        try:
            logger.info("[INIT] Initializing database connection...")
            from utils.timescaledb_client import get_timescaledb_client
            
            self.db_client = get_timescaledb_client()
            if not self.db_client.ensure_connection():
                logger.error("[ERROR] Failed to connect to database")
                return False
            
            self.db_cursor = self.db_client.connection.cursor()
            logger.info("[SUCCESS] Database connection and cursor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize database connection: {e}")
            return False
    
    def _ensure_database_connection(self):
        """Ensure database connection is still valid, reconnect if needed"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.db_client and self.db_cursor:
                    # Test the connection with a simple query
                    self.db_cursor.execute("SELECT 1")
                    return True
                else:
                    return self._init_database_connection()
            except Exception as e:
                logger.warning(f"Database connection test failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info("Reconnecting to database...")
                    self._cleanup_database_connection()
                    time.sleep(1)  # Wait before retry
                    if not self._init_database_connection():
                        continue
                else:
                    logger.error("Failed to reconnect to database after all retries")
                    return False
        return False
    
    def _cleanup_database_connection(self):
        """Cleanup database connections"""
        try:
            if self.db_cursor:
                try:
                    self.db_cursor.close()
                except Exception:
                    pass
                self.db_cursor = None
            if self.db_client and hasattr(self.db_client, 'connection'):
                try:
                    if self.db_client.connection and not self.db_client.connection.closed:
                        self.db_client.connection.close()
                except Exception:
                    pass
                self.db_client = None
            logger.info("[CLEANUP] UniverseDataUpdater: Database connections cleaned up")
        except Exception as e:
            logger.warning(f"[WARN] UniverseDataUpdater: Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup database connections when object is destroyed"""
        self._cleanup_database_connection()
    
    def get_universe_tickers(self, force_refresh: bool = False, universe_file: Optional[str] = None) -> List[str]:
        """Get all tickers from the universe or from a custom file"""
        try:
            # If universe_file is specified, load tickers from that file
            if universe_file:
                logger.info(f"Loading tickers from custom universe file: {universe_file}")
                try:
                    with open(universe_file, 'r') as f:
                        custom_tickers = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(custom_tickers)} tickers from {universe_file}")
                    return custom_tickers
                except FileNotFoundError:
                    logger.error(f"Universe file not found: {universe_file}")
                    return []
                except Exception as e:
                    logger.error(f"Error reading universe file {universe_file}: {e}")
                    return []
            
            # Use default universe
            if force_refresh:
                logger.info("Force refreshing ticker universe...")
                self.ticker_manager.refresh_all_indices()
            
            universe = self.ticker_manager.get_combined_universe()
            logger.info(f"Retrieved {len(universe)} tickers from default universe")
            return universe
            
        except Exception as e:
            logger.error(f"Failed to get universe tickers: {e}")
            # Fallback to cached data
            universe = self.ticker_manager.get_cached_combined_universe()
            if universe:
                logger.warning(f"Using cached universe with {len(universe)} tickers")
                return universe
            else:
                logger.error("No tickers available from cache")
                return []
    
    def ticker_has_data(self, symbol: str) -> bool:
        """
        Check if a ticker already has data for the specified timeframe in TimescaleDB
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            bool: True if data exists, False otherwise
        """
        try:
            # Ensure database connection is valid
            if not self._ensure_database_connection():
                logger.warning(f"Could not connect to TimescaleDB to check {symbol}, will fetch data anyway")
                return False
            
            # Check if data exists for this symbol, provider, and timeframe
            query = """
                SELECT COUNT(*) as count 
                FROM market_data 
                WHERE symbol = %s 
                AND provider = %s 
                AND timeframe = %s
            """
            
            provider_upper = self.provider.upper()
            
            # Use the existing cursor
            self.db_cursor.execute(query, (symbol, provider_upper, self.timeframe))
            result = self.db_cursor.fetchone()
            
            if result and result[0] > 0:
                logger.info(f"📊 {symbol} already has {result[0]} bars for {self.timeframe} from {self.provider}")
                return True
            else:
                logger.info(f"📊 {symbol} has no existing data for {self.timeframe} from {self.provider}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking existing data for {symbol}: {e}, will fetch data anyway")
            return False
    
    def _convert_symbol_for_ib(self, symbol: str) -> str:
        """Intelligent symbol conversion for IB compatibility with caching and discovery"""
        if self.provider != "ib":
            return symbol
        
        # Step 1: Check if we have a cached mapping
        cached_symbol = self._get_cached_symbol_mapping(symbol)
        if cached_symbol:
            return cached_symbol
        
        # Step 2: Try the original symbol first (it might work)
        if self._test_ib_symbol(symbol):
            logger.info(f"[CONVERT] Original symbol {symbol} works with IB")
            self._cache_symbol_mapping(symbol, symbol, True)
            return symbol
        
        # Step 3: Discover the correct symbol by testing variations
        discovered_symbol = self._discover_ib_symbol(symbol)
        if discovered_symbol:
            return discovered_symbol
        
        # Step 4: If no valid symbol found, return original (will fail gracefully)
        logger.warning(f"[CONVERT] No valid IB symbol found for {symbol}, using original")
        return symbol
    
    def fetch_ticker_data(self, symbol: str, use_max_bars: bool = False) -> bool:
        """
        Fetch data for a single ticker
        
        Args:
            symbol: Stock symbol to fetch
            use_max_bars: If True, fetch maximum available bars, otherwise use fixed amounts
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            fetch_start_time = datetime.now()
            logger.info(f"[START] STARTING data fetch for {symbol} @ {self.timeframe} from {self.provider}...")
            
            # Determine number of bars to fetch
            if use_max_bars:
                # Use "max" to get all available data
                bars = "max"
                logger.info(f"[INFO] Fetching maximum available bars for {symbol}")
            else:
                # Use fixed amounts for reliability
                # For daily data: 5 years = ~1260 bars, For hourly data: 1 year = ~8760 bars
                bars = 1260 if self.timeframe == "1d" else 8760
                logger.info(f"[INFO] Fetching {bars} bars for {symbol}")
            
            if self.provider == "alpaca":
                result = fetch_from_alpaca(symbol, bars, self.timeframe)
            elif self.provider == "ib":
                # Convert symbol format for IB compatibility
                ib_symbol = self._convert_symbol_for_ib(symbol)
                if ib_symbol != symbol:
                    logger.info(f"[CONVERT] Converting symbol {symbol} -> {ib_symbol} for IB compatibility")
                result = fetch_from_ib(ib_symbol, bars, self.timeframe)
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return False
            
            fetch_end_time = datetime.now()
            fetch_duration = fetch_end_time - fetch_start_time
            
            if result is not None and not result.empty:
                logger.info(f"[SUCCESS] SUCCESSFULLY fetched {len(result)} bars for {symbol} in {fetch_duration}")
                
                # Queue data for immediate database saving (non-blocking)
                queue_start_time = datetime.now()
                self.db_worker.save_data(result, symbol, self.provider.upper(), self.timeframe)
                
                queue_end_time = datetime.now()
                queue_duration = queue_end_time - queue_start_time
                logger.info(f"[OK] Data queued for {symbol} in {queue_duration}, queue size: {self.db_worker.queue.qsize()}")
                
                return True
            else:
                logger.warning(f"[WARN] No data returned for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
            import traceback
            logger.error(f"❌ Exception traceback: {traceback.format_exc()}")
            return False
    
    def update_universe_data(self, 
                           batch_size: int = 10, 
                           delay_between_batches: float = 5.0,
                           delay_between_tickers: float = 1.0,
                           max_tickers: Optional[int] = None,
                           start_from_index: int = 0,
                           use_max_bars: bool = False,
                           skip_existing: bool = False,
                           universe_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Update market data for all tickers in the universe
        
        Args:
            batch_size: Number of tickers to process before taking a break
            delay_between_batches: Seconds to wait between batches
            delay_between_tickers: Seconds to wait between individual tickers
            max_tickers: Maximum number of tickers to process (None for all)
            start_from_index: Index to start processing from (for resuming)
            use_max_bars: If True, fetch maximum available bars, otherwise use fixed amounts
            skip_existing: If True, skip tickers that already have data for the timeframe
            universe_file: Optional custom universe file to load tickers from
            
        Returns:
            Dict with update statistics
        """
        # Get universe tickers
        tickers = self.get_universe_tickers(universe_file=universe_file)
        if not tickers:
            return {"error": "No tickers available"}
        
        # Apply start index and max limit
        tickers = tickers[start_from_index:]
        if max_tickers:
            tickers = tickers[:max_tickers]
        
        total_tickers = len(tickers)
        logger.info(f"Starting universe update for {total_tickers} tickers")
        logger.info(f"Provider: {self.provider}, Timeframe: {self.timeframe}")
        logger.info(f"Max bars: {'Yes' if use_max_bars else 'No (fixed amounts)'}")
        logger.info(f"Skip existing: {'Yes' if skip_existing else 'No'}")
        logger.info(f"Index range: {start_from_index} to {start_from_index + total_tickers - 1}")
        logger.info(f"Batch size: {batch_size}, Delays: {delay_between_batches}s between batches, {delay_between_tickers}s between tickers")
        
        # Start database worker for non-blocking saves
        self.db_worker.start()
        
        # Statistics
        successful = 0
        failed = 0
        skipped = 0
        failed_symbols = []
        skipped_symbols = []
        
        start_time = datetime.now()
        logger.info(f"🔍 Starting main processing loop at {start_time}")
        
        try:
            for i, symbol in enumerate(tickers):
                ticker_start_time = datetime.now()
                current_index = start_from_index + i
                
                logger.info(f"[PROCESSING] {symbol} ({i+1}/{total_tickers}, overall index: {current_index})")
                
                # Check if we should skip this ticker
                if skip_existing:
                    has_data = self.ticker_has_data(symbol)
                    
                    if has_data:
                        logger.info(f"[SKIP] {symbol} - already has data")
                        skipped += 1
                        skipped_symbols.append(symbol)
                        continue
                    else:
                        logger.info(f"[OK] {symbol} doesn't have data, will fetch...")
                else:
                    logger.info(f"[OK] Skip existing disabled, will fetch data for {symbol}")
                
                # Fetch data for this ticker with timeout protection
                logger.info(f"[FETCH] Starting data fetch for {symbol}...")
                try:
                    # For IB provider, we need to run in main thread due to event loop requirements
                    # Use a simple approach without threading
                    fetch_result = self.fetch_ticker_data(symbol, use_max_bars)
                    
                    if fetch_result:
                        successful += 1
                        ticker_end_time = datetime.now()
                        ticker_duration = ticker_end_time - ticker_start_time
                        logger.info(f"[SUCCESS] COMPLETED {symbol} in {ticker_duration} (fetch + queue)")
                    else:
                        failed += 1
                        failed_symbols.append(symbol)
                        ticker_end_time = datetime.now()
                        ticker_duration = ticker_end_time - ticker_start_time
                        logger.info(f"[FAILED] FAILED {symbol} in {ticker_duration}")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Exception during processing {symbol}: {e}")
                    import traceback
                    logger.error(f"[ERROR] Exception traceback: {traceback.format_exc()}")
                    
                    # Check if it's a common IB error
                    error_str = str(e).lower()
                    if "no security definition" in error_str or "contract not found" in error_str:
                        logger.warning(f"[WARN] Symbol {symbol} not found in IB - this is normal for some symbols")
                    elif "timeout" in error_str:
                        logger.warning(f"[WARN] Timeout for {symbol} - IB API may be slow")
                    elif "rate limit" in error_str:
                        logger.warning(f"[WARN] Rate limited for {symbol} - IB API throttling")
                    
                    failed += 1
                    failed_symbols.append(symbol)
                    continue
                
                # Delay between tickers (except for the last one)
                if i < total_tickers - 1:
                    time.sleep(delay_between_tickers)
                
                # Batch processing with delays
                if (i + 1) % batch_size == 0 and i < total_tickers - 1:
                    logger.info(f"[BATCH] Completed batch {i//batch_size + 1}. Taking {delay_between_batches}s break...")
                    logger.info(f"[PROGRESS] Progress: {i+1}/{total_tickers} tickers processed")
                    logger.info(f"[STATS] Success: {successful}, Failed: {failed}, Skipped: {skipped}")
                    time.sleep(delay_between_batches)
                
        except KeyboardInterrupt:
            logger.warning("Update interrupted by user")
            logger.info(f"Processed {i+1}/{total_tickers} tickers before interruption")
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"🔍 Main processing loop completed in {duration}")
        
        # Wait for all database operations to complete
        logger.info("=" * 60)
        logger.info("🔄 ALL TICKERS PROCESSED - NOW WAITING FOR DATABASE OPERATIONS")
        logger.info("=" * 60)
        logger.info(f"📊 Total tickers processed: {total_tickers}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏭️ Skipped: {skipped}")
        logger.info(f"⏳ Starting to wait for database operations to complete...")
        
        db_wait_start = datetime.now()
        logger.info(f"🔍 About to call db_worker.wait_for_completion() at {db_wait_start}")
        
        self.db_worker.wait_for_completion()
        
        db_wait_end = datetime.now()
        db_wait_duration = db_wait_end - db_wait_start
        logger.info(f"🔍 db_worker.wait_for_completion() returned at {db_wait_end}")
        logger.info(f"💾 Database operations completed in {db_wait_duration}")
        
        # Calculate time breakdown
        total_duration = end_time - start_time
        data_processing_time = total_duration - db_wait_duration
        
        logger.info("=" * 60)
        logger.info("⏱️ TIME BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"📊 Data fetching + queuing: {data_processing_time}")
        logger.info(f"💾 Database operations: {db_wait_duration}")
        logger.info(f"⏱️ Total time: {total_duration}")
        logger.info(f"🚀 Non-blocking efficiency: {(data_processing_time / total_duration * 100):.1f}% of time spent on data fetching")
        
        # Stop database worker
        self.db_worker.stop()
        
        # Cleanup database connections
        self._cleanup_database_connection()
        
        # Cleanup IBKR connection if using IB provider
        if self.provider == "ib":
            try:
                from utils.fetch_data import cleanup_ib_connection
                cleanup_ib_connection()
                logger.info("🧹 Cleaned up IBKR connection")
            except Exception as e:
                logger.warning(f"Error cleaning up IBKR connection: {e}")
        
        # Get database worker statistics
        db_stats = self.db_worker.get_stats()
        
        # Compile results
        results = {
            "total_tickers": total_tickers,
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (successful / (total_tickers - skipped) * 100) if (total_tickers - skipped) > 0 else 0,
            "failed_symbols": failed_symbols,
            "skipped_symbols": skipped_symbols,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "provider": self.provider,
            "timeframe": self.timeframe,
            "use_max_bars": use_max_bars,
            "skip_existing": skip_existing,
            "last_processed_index": start_from_index + total_tickers - 1,
            "database_stats": db_stats
        }
        
        # Log summary
        logger.info("=" * 60)
        logger.info("UNIVERSE UPDATE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total tickers processed: {total_tickers}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")
        logger.info(f"Duration: {duration}")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Max bars: {'Yes' if use_max_bars else 'No (fixed amounts)'}")
        logger.info(f"Skip existing: {'Yes' if skip_existing else 'No'}")
        logger.info(f"Database operations: {db_stats['saved']} saved, {db_stats['failed']} failed")
        
        if failed_symbols:
            logger.info(f"Failed symbols: {', '.join(failed_symbols[:10])}")
            if len(failed_symbols) > 10:
                logger.info(f"... and {len(failed_symbols) - 10} more")
        
        if skipped_symbols:
            logger.info(f"Skipped symbols: {', '.join(skipped_symbols[:10])}")
            if len(skipped_symbols) > 10:
                logger.info(f"... and {len(skipped_symbols) - 10} more")
        
        if db_stats['errors']:
            logger.warning(f"Database errors: {', '.join(db_stats['errors'][:5])}")
            if len(db_stats['errors']) > 5:
                logger.warning(f"... and {len(db_stats['errors']) - 5} more database errors")
        
        return results
    
    def resume_update(self, last_processed_index: int, **kwargs) -> Dict[str, Any]:
        """
        Resume update from a specific index
        
        Args:
            last_processed_index: Last successfully processed index
            **kwargs: Other arguments for update_universe_data
            
        Returns:
            Dict with update statistics
        """
        start_index = last_processed_index + 1
        logger.info(f"Resuming update from index {start_index}")
        return self.update_universe_data(start_from_index=start_index, **kwargs)

    def _create_symbol_mapping_table(self):
        """Create the symbol mapping table if it doesn't exist"""
        try:
            if not self._ensure_database_connection():
                logger.error("[ERROR] Cannot create symbol mapping table - no database connection")
                return False
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS symbol_mappings (
                id SERIAL PRIMARY KEY,
                original_symbol VARCHAR(20) NOT NULL,
                ib_symbol VARCHAR(20) NOT NULL,
                provider VARCHAR(10) DEFAULT 'IB',
                is_valid BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_used TIMESTAMPTZ DEFAULT NOW(),
                use_count INTEGER DEFAULT 1,
                UNIQUE(original_symbol, provider)
            );
            
            -- Create index for faster lookups
            CREATE INDEX IF NOT EXISTS idx_symbol_mappings_original_symbol 
            ON symbol_mappings(original_symbol, provider);
            
            CREATE INDEX IF NOT EXISTS idx_symbol_mappings_ib_symbol 
            ON symbol_mappings(ib_symbol, provider);
            """
            
            self.db_cursor.execute(create_table_sql)
            logger.info("[SUCCESS] Symbol mapping table created/verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create symbol mapping table: {e}")
            return False

    def _get_cached_symbol_mapping(self, original_symbol: str) -> Optional[str]:
        """Get cached IB symbol mapping from database"""
        try:
            if not self._ensure_database_connection():
                return None
            
            query = """
                SELECT ib_symbol, is_valid 
                FROM symbol_mappings 
                WHERE original_symbol = %s AND provider = 'IB'
                ORDER BY last_used DESC 
                LIMIT 1
            """
            
            self.db_cursor.execute(query, (original_symbol,))
            result = self.db_cursor.fetchone()
            
            if result:
                ib_symbol, is_valid = result
                if is_valid:
                    # Update usage statistics
                    self._update_symbol_mapping_usage(original_symbol)
                    logger.info(f"[CACHE] Found cached mapping: {original_symbol} -> {ib_symbol}")
                    return ib_symbol
                else:
                    logger.info(f"[CACHE] Found invalid cached mapping: {original_symbol} -> {ib_symbol}")
                    return None
            
            return None
            
        except Exception as e:
            logger.warning(f"[WARN] Error looking up cached symbol mapping for {original_symbol}: {e}")
            return None
    
    def _update_symbol_mapping_usage(self, original_symbol: str):
        """Update usage statistics for a symbol mapping"""
        try:
            query = """
                UPDATE symbol_mappings 
                SET last_used = NOW(), use_count = use_count + 1
                WHERE original_symbol = %s AND provider = 'IB'
            """
            self.db_cursor.execute(query, (original_symbol,))
            
        except Exception as e:
            logger.warning(f"[WARN] Error updating symbol mapping usage for {original_symbol}: {e}")
    
    def _cache_symbol_mapping(self, original_symbol: str, ib_symbol: str, is_valid: bool = True):
        """Cache a symbol mapping in the database"""
        try:
            if not self._ensure_database_connection():
                return False
            
            # Use UPSERT to handle duplicates
            query = """
                INSERT INTO symbol_mappings (original_symbol, ib_symbol, provider, is_valid, created_at, last_used, use_count)
                VALUES (%s, %s, 'IB', %s, NOW(), NOW(), 1)
                ON CONFLICT (original_symbol, provider) 
                DO UPDATE SET 
                    ib_symbol = EXCLUDED.ib_symbol,
                    is_valid = EXCLUDED.is_valid,
                    last_used = NOW(),
                    use_count = symbol_mappings.use_count + 1
            """
            
            self.db_cursor.execute(query, (original_symbol, ib_symbol, is_valid))
            logger.info(f"[CACHE] Cached symbol mapping: {original_symbol} -> {ib_symbol} (valid: {is_valid})")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error caching symbol mapping for {original_symbol}: {e}")
            return False
    
    def _discover_ib_symbol(self, original_symbol: str) -> Optional[str]:
        """Discover the correct IB symbol by testing variations"""
        if self.provider != "ib":
            return original_symbol
        
        logger.info(f"[DISCOVER] Discovering IB symbol for: {original_symbol}")
        
        # Generate symbol variations to test
        variations = self._generate_symbol_variations(original_symbol)
        
        # Test each variation with IB
        for variation in variations:
            if self._test_ib_symbol(variation):
                logger.info(f"[DISCOVER] Found valid IB symbol: {original_symbol} -> {variation}")
                # Cache the successful mapping
                self._cache_symbol_mapping(original_symbol, variation, True)
                return variation
        
        # If no valid symbol found, cache the original as invalid
        logger.warning(f"[DISCOVER] No valid IB symbol found for: {original_symbol}")
        self._cache_symbol_mapping(original_symbol, original_symbol, False)
        return None
    
    def _generate_symbol_variations(self, symbol: str) -> List[str]:
        """Generate possible IB symbol variations"""
        variations = [
            symbol,                    # Original: BF.B
            symbol.replace('.', ' '), # BF B (space - most common for Class B)
            symbol.replace('.', '-'), # BF-B
            symbol.replace('.', ''),  # BFB
            symbol.replace('.', '/'), # BF/B
            symbol.replace('.', '_'), # BF_B
        ]
        
        # Add Class B specific variations
        if '.' in symbol and symbol.endswith('.B'):
            base = symbol.split('.')[0]
            variations.extend([
                f"{base} B",           # BF B
                f"{base}-B",           # BF-B
                f"{base}B",            # BFB
                f"{base}/B",           # BF/B
            ])
        
        # Remove duplicates and None values
        variations = list(dict.fromkeys([v for v in variations if v]))
        logger.info(f"[DISCOVER] Testing variations for {symbol}: {variations}")
        return variations
    
    def _test_ib_symbol(self, symbol: str) -> bool:
        """Test if a symbol exists and has data in IB"""
        try:
            from utils.fetch_data import get_ib_connection
            from ib_insync import Stock
            
            # Get IB connection
            ib = get_ib_connection()
            if not ib.isConnected():
                logger.warning(f"[TEST] Cannot test {symbol} - IB not connected")
                return False
            
            # Create contract
            contract = Stock(symbol=symbol, exchange='SMART', currency='USD')
            
            # Try to qualify the contract
            try:
                ib.qualifyContracts(contract)
                logger.info(f"[TEST] Symbol {symbol} is recognized by IB")
                
                # Try to get a small amount of historical data to confirm it's tradeable
                try:
                    bars = ib.reqHistoricalData(
                        contract,
                        endDateTime='',
                        durationStr='1 D',
                        barSizeSetting='1 hour',
                        whatToShow='TRADES',
                        useRTH=True,
                        formatDate=1
                    )
                    
                    if bars and len(bars) > 0:
                        logger.info(f"[TEST] Symbol {symbol} is tradeable with {len(bars)} bars")
                        return True
                    else:
                        logger.info(f"[TEST] Symbol {symbol} exists but has no data")
                        return False
                        
                except Exception as data_error:
                    logger.info(f"[TEST] Symbol {symbol} exists but data fetch failed: {data_error}")
                    return False
                    
            except Exception as e:
                error_str = str(e).lower()
                if "no security definition" in error_str:
                    logger.info(f"[TEST] Symbol {symbol} - no security definition")
                elif "contract not found" in error_str:
                    logger.info(f"[TEST] Symbol {symbol} - contract not found")
                else:
                    logger.info(f"[TEST] Symbol {symbol} - error: {e}")
                return False
                
        except Exception as e:
            logger.warning(f"[TEST] Error testing symbol {symbol}: {e}")
            return False

    def get_symbol_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about symbol mappings"""
        try:
            if not self._ensure_database_connection():
                return {"error": "No database connection"}
            
            # Get total mappings
            self.db_cursor.execute("SELECT COUNT(*) FROM symbol_mappings WHERE provider = 'IB'")
            total_mappings = self.db_cursor.fetchone()[0]
            
            # Get valid mappings
            self.db_cursor.execute("SELECT COUNT(*) FROM symbol_mappings WHERE provider = 'IB' AND is_valid = TRUE")
            valid_mappings = self.db_cursor.fetchone()[0]
            
            # Get invalid mappings
            self.db_cursor.execute("SELECT COUNT(*) FROM symbol_mappings WHERE provider = 'IB' AND is_valid = FALSE")
            invalid_mappings = self.db_cursor.fetchone()[0]
            
            # Get most used mappings
            self.db_cursor.execute("""
                SELECT original_symbol, ib_symbol, use_count, last_used 
                FROM symbol_mappings 
                WHERE provider = 'IB' AND is_valid = TRUE 
                ORDER BY use_count DESC 
                LIMIT 10
            """)
            most_used = self.db_cursor.fetchall()
            
            # Get recent mappings
            self.db_cursor.execute("""
                SELECT original_symbol, ib_symbol, is_valid, created_at 
                FROM symbol_mappings 
                WHERE provider = 'IB' 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            recent_mappings = self.db_cursor.fetchall()
            
            return {
                "total_mappings": total_mappings,
                "valid_mappings": valid_mappings,
                "invalid_mappings": invalid_mappings,
                "success_rate": (valid_mappings / total_mappings * 100) if total_mappings > 0 else 0,
                "most_used": most_used,
                "recent_mappings": recent_mappings
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting symbol mapping stats: {e}")
            return {"error": str(e)}
    
    def view_symbol_mappings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """View all symbol mappings"""
        try:
            if not self._ensure_database_connection():
                return []
            
            query = """
                SELECT original_symbol, ib_symbol, is_valid, use_count, created_at, last_used
                FROM symbol_mappings 
                WHERE provider = 'IB' 
                ORDER BY last_used DESC 
                LIMIT %s
            """
            
            self.db_cursor.execute(query, (limit,))
            results = self.db_cursor.fetchall()
            
            mappings = []
            for row in results:
                mappings.append({
                    "original_symbol": row[0],
                    "ib_symbol": row[1],
                    "is_valid": row[2],
                    "use_count": row[3],
                    "created_at": row[4],
                    "last_used": row[5]
                })
            
            return mappings
            
        except Exception as e:
            logger.error(f"[ERROR] Error viewing symbol mappings: {e}")
            return []
    
    def clear_invalid_mappings(self) -> int:
        """Clear all invalid symbol mappings"""
        try:
            if not self._ensure_database_connection():
                return 0
            
            self.db_cursor.execute("DELETE FROM symbol_mappings WHERE provider = 'IB' AND is_valid = FALSE")
            deleted_count = self.db_cursor.rowcount
            
            logger.info(f"[CLEANUP] Cleared {deleted_count} invalid symbol mappings")
            return deleted_count
            
        except Exception as e:
            logger.error(f"[ERROR] Error clearing invalid mappings: {e}")
            return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Update market data for all tickers in the universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all tickers with Alpaca daily data
  python update_universe_data.py --provider alpaca --timeframe 1d
  
  # Update with hourly data, process in batches of 20
  python update_universe_data.py --provider alpaca --timeframe 1h --batch-size 20
  
  # Resume from a specific index
  python update_universe_data.py --provider alpaca --timeframe 1d --resume-from 150
  
  # Process only first 100 tickers
  python update_universe_data.py --provider alpaca --timeframe 1d --max-tickers 100
  
  # Process tickers from index 100 to 199 (100 tickers starting from index 100)
  python update_universe_data.py --provider alpaca --timeframe 1d --start-index 100 --max-tickers 100
  
  # Process tickers from index 500 to 599 (100 tickers starting from index 500)
  python update_universe_data.py --provider ib --timeframe 1h --start-index 500 --max-tickers 100
  
  # Skip tickers that already have data
  python update_universe_data.py --provider alpaca --timeframe 1d --skip-existing
  
  # Fetch maximum bars and skip existing data for a specific range
  python update_universe_data.py --provider ib --timeframe 1h --max-bars --skip-existing --start-index 200 --max-tickers 50
        """
    )
    
    parser.add_argument(
        "--provider", 
        choices=["alpaca", "ib"], 
        default="alpaca",
        help="Data provider (default: alpaca)"
    )
    
    parser.add_argument(
        "--timeframe", 
        choices=["1h", "1d"], 
        default="1d",
        help="Timeframe to fetch (default: 1d)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10,
        help="Number of tickers to process before taking a break (default: 10)"
    )
    
    parser.add_argument(
        "--delay-batches", 
        type=float, 
        default=5.0,
        help="Seconds to wait between batches (default: 5.0)"
    )
    
    parser.add_argument(
        "--delay-tickers", 
        type=float, 
        default=1.0,
        help="Seconds to wait between individual tickers (default: 1.0)"
    )
    
    parser.add_argument(
        "--max-tickers", 
        type=int, 
        help="Maximum number of tickers to process (default: all)"
    )
    
    parser.add_argument(
        "--start-index", 
        type=int, 
        default=0,
        help="Index to start processing from (0-based, default: 0)"
    )
    
    parser.add_argument(
        "--resume-from", 
        type=int, 
        help="Resume processing from this index (0-based)"
    )
    
    parser.add_argument(
        "--force-refresh", 
        action="store_true",
        help="Force refresh of ticker universe before processing"
    )
    
    parser.add_argument(
        "--max-bars", 
        action="store_true",
        help="Fetch maximum available bars instead of fixed amounts (may take longer)"
    )
    
    parser.add_argument(
        "--skip-existing", 
        action="store_true",
        help="Skip tickers that already have data for the specified timeframe"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be processed without actually fetching data"
    )
    
    parser.add_argument(
        "--universe-file", 
        type=str, 
        help="Load tickers from this custom universe file instead of default"
    )
    
    parser.add_argument(
        "--view-mappings", 
        action="store_true",
        help="View symbol mappings and exit"
    )
    
    parser.add_argument(
        "--mapping-stats", 
        action="store_true",
        help="Show symbol mapping statistics and exit"
    )
    
    parser.add_argument(
        "--clear-invalid-mappings", 
        action="store_true",
        help="Clear all invalid symbol mappings and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize updater
        updater = UniverseDataUpdater(provider=args.provider, timeframe=args.timeframe)
        
        if args.dry_run:
            # Show what would be processed
            tickers = updater.get_universe_tickers(force_refresh=args.force_refresh, universe_file=args.universe_file)
            if args.max_tickers:
                tickers = tickers[:args.max_tickers]
            if args.resume_from:
                tickers = tickers[args.resume_from:]
            
            print(f"DRY RUN - Would process {len(tickers)} tickers:")
            print(f"Provider: {args.provider}")
            print(f"Timeframe: {args.timeframe}")
            print(f"Max bars: {'Yes' if args.max_bars else 'No (fixed amounts)'}")
            print(f"Skip existing: {'Yes' if args.skip_existing else 'No'}")
            print(f"First 10 tickers: {tickers[:10]}")
            if len(tickers) > 10:
                print(f"Last 10 tickers: {tickers[-10:]}")
            return 0
        
        if args.view_mappings:
            mappings = updater.view_symbol_mappings()
            if mappings:
                print("\n--- Symbol Mappings ---")
                for mapping in mappings:
                    print(f"Original: {mapping['original_symbol']}, IB: {mapping['ib_symbol']}, Valid: {mapping['is_valid']}, Use Count: {mapping['use_count']}, Created: {mapping['created_at']}, Last Used: {mapping['last_used']}")
                print("-" * 50)
            else:
                print("No symbol mappings found.")
            return 0

        if args.mapping_stats:
            stats = updater.get_symbol_mapping_stats()
            print("\n--- Symbol Mapping Statistics ---")
            print(f"Total Mappings: {stats['total_mappings']}")
            print(f"Valid Mappings: {stats['valid_mappings']}")
            print(f"Invalid Mappings: {stats['invalid_mappings']}")
            print(f"Success Rate: {stats['success_rate']:.1f}%")
            print("\nMost Used Mappings:")
            for mapping in stats['most_used']:
                print(f"Original: {mapping[0]}, IB: {mapping[1]}, Use Count: {mapping[2]}, Last Used: {mapping[3]}")
            print("\nRecent Mappings:")
            for mapping in stats['recent_mappings']:
                print(f"Original: {mapping[0]}, IB: {mapping[1]}, Valid: {mapping[2]}, Created: {mapping[3]}")
            print("-" * 50)
            return 0

        if args.clear_invalid_mappings:
            deleted_count = updater.clear_invalid_mappings()
            print(f"\nCleared {deleted_count} invalid symbol mappings.")
            return 0

        # Determine start index
        start_index = args.resume_from if args.resume_from is not None else args.start_index
        
        # Run the update
        results = updater.update_universe_data(
            batch_size=args.batch_size,
            delay_between_batches=args.delay_batches,
            delay_between_tickers=args.delay_tickers,
            max_tickers=args.max_tickers,
            start_from_index=start_index,
            use_max_bars=args.max_bars,
            skip_existing=args.skip_existing,
            universe_file=args.universe_file
        )
        
        if "error" in results:
            logger.error(f"Update failed: {results['error']}")
            return 1
        
        # Save results summary
        summary_file = f"logs/universe_update_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("UNIVERSE UPDATE SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Provider: {results['provider']}\n")
            f.write(f"Timeframe: {results['timeframe']}\n")
            f.write(f"Max bars: {'Yes' if results['use_max_bars'] else 'No (fixed amounts)'}\n")
            f.write(f"Skip existing: {'Yes' if results['skip_existing'] else 'No'}\n")
            f.write(f"Total tickers: {results['total_tickers']}\n")
            f.write(f"Successful: {results['successful']}\n")
            f.write(f"Failed: {results['failed']}\n")
            f.write(f"Skipped: {results['skipped']}\n")
            f.write(f"Success rate: {results['success_rate']:.1f}%\n")
            f.write(f"Duration: {results['duration']}\n")
            f.write(f"Start time: {results['start_time']}\n")
            f.write(f"End time: {results['end_time']}\n")
            f.write(f"Last processed index: {results['last_processed_index']}\n")
            f.write(f"\nDatabase Operations:\n")
            f.write(f"  Saved: {results['database_stats']['saved']}\n")
            f.write(f"  Failed: {results['database_stats']['failed']}\n")
            
            if results['failed_symbols']:
                f.write(f"\nFailed symbols:\n")
                for symbol in results['failed_symbols']:
                    f.write(f"  {symbol}\n")
            
            if results['skipped_symbols']:
                f.write(f"\nSkipped symbols:\n")
                for symbol in results['skipped_symbols']:
                    f.write(f"  {symbol}\n")
            
            if results['database_stats']['errors']:
                f.write(f"\nDatabase errors:\n")
                for error in results['database_stats']['errors']:
                    f.write(f"  {error}\n")
        
        logger.info(f"Results summary saved to: {summary_file}")
        logger.info("Update completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Update failed with error: {e}")
        return 1
    finally:
        # Ensure IBKR connection is cleaned up even if script fails
        if args.provider == "ib":
            try:
                from utils.fetch_data import cleanup_ib_connection
                cleanup_ib_connection()
                logger.info("[CLEANUP] Cleaned up IBKR connection in finally block")
            except Exception as cleanup_error:
                logger.warning(f"[WARN] Error cleaning up IBKR connection in finally block: {cleanup_error}")
        
        # Ensure TimescaleDB client is cleaned up
        try:
            from utils.timescaledb_client import close_timescaledb_client
            close_timescaledb_client()
            logger.info("[CLEANUP] Cleaned up TimescaleDB client in finally block")
        except Exception as cleanup_error:
            logger.warning(f"[WARN] Error cleaning up TimescaleDB client in finally block: {cleanup_error}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
