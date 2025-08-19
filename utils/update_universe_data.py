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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('universe_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatabaseWorker:
    """Handles database operations in a separate thread to avoid blocking data fetching"""
    
    def __init__(self):
        self.queue = Queue()
        self.worker_thread = None
        self.running = False
        self.stats = {"saved": 0, "failed": 0, "errors": []}
        
    def start(self):
        """Start the database worker thread"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("🚀 Database worker thread started")
    
    def stop(self):
        """Stop the database worker thread"""
        self.running = False
        if self.worker_thread:
            self.queue.put(None)  # Signal to stop
            self.worker_thread.join(timeout=5)
            logger.info("🛑 Database worker thread stopped")
    
    def save_data(self, df, symbol, provider, timeframe):
        """Queue data for saving (non-blocking)"""
        if self.running:
            logger.info(f"🔍 Queueing data for {symbol}: {len(df)} bars, provider: {provider}, timeframe: {timeframe}")
            self.queue.put((df, symbol, provider, timeframe))
            logger.info(f"✅ Data queued for {symbol}, queue size: {self.queue.qsize()}")
        else:
            logger.warning(f"⚠️ Database worker not running, using synchronous save for {symbol}")
            # Fallback to synchronous saving if worker not running
            self._save_data_sync(df, symbol, provider, timeframe)
    
    def _worker_loop(self):
        """Main worker loop that processes database operations"""
        logger.info("🔄 Database worker loop started")
        while self.running:
            item = None  # Initialize item variable
            try:
                logger.debug(f"🔍 Waiting for data in queue (size: {self.queue.qsize()})...")
                item = self.queue.get(timeout=1)
                if item is None:  # Stop signal
                    logger.info("🛑 Received stop signal, exiting worker loop")
                    break
                
                df, symbol, provider, timeframe = item
                process_start_time = datetime.now()
                logger.info(f"📥 STARTING database processing for {symbol}: {len(df)} bars at {process_start_time.strftime('%H:%M:%S')}")
                
                if self._save_data_sync(df, symbol, provider, timeframe):
                    self.stats["saved"] += 1
                    process_end_time = datetime.now()
                    process_duration = process_end_time - process_start_time
                    logger.info(f"✅ SUCCESSFULLY saved data for {symbol} in {process_duration}")
                else:
                    self.stats["failed"] += 1
                    process_end_time = datetime.now()
                    process_duration = process_end_time - process_start_time
                    logger.error(f"❌ FAILED to save data for {symbol} in {process_duration}")
                
                self.queue.task_done()
                
            except queue.Empty:
                # This is normal when the queue is empty, just continue
                logger.debug("⏳ Queue is empty, continuing...")
                continue
            except Exception as e:
                logger.error(f"Database worker error: {e}")
                import traceback
                logger.error(f"Database worker traceback: {traceback.format_exc()}")
                self.stats["errors"].append(str(e))
                if item:  # Now item is always defined
                    self.queue.task_done()
        
        logger.info("🔄 Database worker loop ended")
    
    def _save_data_sync(self, df, symbol, provider, timeframe):
        """Synchronous data saving (fallback method)"""
        try:
            from utils.timescaledb_client import get_timescaledb_client
            
            client = get_timescaledb_client()
            if not client.ensure_connection():
                logger.error(f"✗ Failed to connect to TimescaleDB for {symbol}")
                return False
            
            # Use the client's insert_market_data method
            if hasattr(client, 'insert_market_data'):
                if client.insert_market_data(df, symbol, provider, timeframe):
                    logger.info(f"💾 Saved {len(df)} records to TimescaleDB for {symbol} {timeframe}")
                    return True
                else:
                    logger.error(f"✗ Failed to save to TimescaleDB for {symbol} {timeframe}")
                    return False
            else:
                # Fallback: use the client's save_market_data method if available
                if hasattr(client, 'save_market_data'):
                    if client.save_market_data(df, symbol, provider, timeframe):
                        logger.info(f"💾 Saved {len(df)} records to TimescaleDB for {symbol} {timeframe}")
                        return True
                    else:
                        logger.error(f"✗ Failed to save to TimescaleDB for {symbol} {timeframe}")
                        return False
                else:
                    logger.error(f"✗ TimescaleDB client doesn't have insert_market_data or save_market_data method")
                    return False
                
        except Exception as e:
            logger.error(f"✗ Error saving to TimescaleDB for {symbol}: {e}")
            import traceback
            logger.error(f"✗ Traceback: {traceback.format_exc()}")
            return False
    
    def wait_for_completion(self):
        """Wait for all queued operations to complete"""
        if self.running:
            self.queue.join()
            logger.info(f"📊 Database operations completed: {self.stats['saved']} saved, {self.stats['failed']} failed")
    
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
        
        # Validation
        if self.provider not in ['alpaca', 'ib']:
            raise ValueError("Provider must be 'alpaca' or 'ib'")
        if self.timeframe not in ['1h', '1d']:
            raise ValueError("Timeframe must be '1h' or '1d'")
        
        logger.info(f"Initialized UniverseDataUpdater for {self.provider} @ {self.timeframe}")
    
    def get_universe_tickers(self, force_refresh: bool = False) -> List[str]:
        """Get all tickers from the universe"""
        try:
            if force_refresh:
                logger.info("Force refreshing ticker universe...")
                self.ticker_manager.refresh_all_indices()
            
            universe = self.ticker_manager.get_combined_universe()
            logger.info(f"Retrieved {len(universe)} tickers from universe")
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
            from utils.timescaledb_client import get_timescaledb_client
            
            client = get_timescaledb_client()
            if not client.ensure_connection():
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
            # Use cursor directly since the client doesn't have execute_query method
            cursor = client.connection.cursor()
            cursor.execute(query, (symbol, provider_upper, self.timeframe))
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0] > 0:
                logger.info(f"📊 {symbol} already has {result[0]} bars for {self.timeframe} from {self.provider}")
                return True
            else:
                logger.info(f"📊 {symbol} has no existing data for {self.timeframe} from {self.provider}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking existing data for {symbol}: {e}, will fetch data anyway")
            return False
    
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
            logger.info(f"🚀 STARTING data fetch for {symbol} @ {self.timeframe} from {self.provider}...")
            
            # Determine number of bars to fetch
            if use_max_bars:
                # Use "max" to get all available data
                bars = "max"
                logger.info(f"📊 Fetching maximum available bars for {symbol}")
            else:
                # Use fixed amounts for reliability
                # For daily data: 5 years = ~1260 bars, For hourly data: 1 year = ~8760 bars
                bars = 1260 if self.timeframe == "1d" else 8760
                logger.info(f"📊 Fetching {bars} bars for {symbol}")
            
            if self.provider == "alpaca":
                result = fetch_from_alpaca(symbol, bars, self.timeframe)
            elif self.provider == "ib":
                result = fetch_from_ib(symbol, bars, self.timeframe)
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return False
            
            fetch_end_time = datetime.now()
            fetch_duration = fetch_end_time - fetch_start_time
            
            if result is not None and not result.empty:
                logger.info(f"✅ SUCCESSFULLY fetched {len(result)} bars for {symbol} in {fetch_duration}")
                
                # Queue data for immediate database saving (non-blocking)
                queue_start_time = datetime.now()
                logger.info(f"🔍 QUEUEING data for {symbol}: {len(result)} bars, provider: {self.provider.upper()}, timeframe: {self.timeframe}")
                self.db_worker.save_data(result, symbol, self.provider.upper(), self.timeframe)
                queue_end_time = datetime.now()
                queue_duration = queue_end_time - queue_start_time
                logger.info(f"✅ Data queued for {symbol} in {queue_duration}, queue size: {self.db_worker.queue.qsize()}")
                
                return True
            else:
                logger.warning(f"⚠️ No data returned for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
            return False
    
    def update_universe_data(self, 
                           batch_size: int = 10, 
                           delay_between_batches: float = 5.0,
                           delay_between_tickers: float = 1.0,
                           max_tickers: Optional[int] = None,
                           start_from_index: int = 0,
                           use_max_bars: bool = False,
                           skip_existing: bool = False) -> Dict[str, Any]:
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
            
        Returns:
            Dict with update statistics
        """
        # Get universe tickers
        tickers = self.get_universe_tickers()
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
        
        try:
            for i, symbol in enumerate(tickers):
                ticker_start_time = datetime.now()
                current_index = start_from_index + i
                
                logger.info(f"🔄 PROCESSING {symbol} ({i+1}/{total_tickers}, overall index: {current_index})")
                
                # Check if we should skip this ticker
                if skip_existing and self.ticker_has_data(symbol):
                    logger.info(f"⏭️ Skipping {symbol} - already has data")
                    skipped += 1
                    skipped_symbols.append(symbol)
                    continue
                
                # Fetch data for this ticker
                logger.info(f"📡 Starting data fetch for {symbol}...")
                if self.fetch_ticker_data(symbol, use_max_bars):
                    successful += 1
                    ticker_end_time = datetime.now()
                    ticker_duration = ticker_end_time - ticker_start_time
                    logger.info(f"🎯 COMPLETED {symbol} in {ticker_duration} (fetch + queue)")
                else:
                    failed += 1
                    failed_symbols.append(symbol)
                    ticker_end_time = datetime.now()
                    ticker_duration = ticker_end_time - ticker_start_time
                    logger.info(f"❌ FAILED {symbol} in {ticker_duration}")
                
                # Delay between tickers (except for the last one)
                if i < total_tickers - 1:
                    logger.info(f"⏱️ Waiting {delay_between_tickers}s before next ticker...")
                    time.sleep(delay_between_tickers)
                
                # Batch processing with delays
                if (i + 1) % batch_size == 0 and i < total_tickers - 1:
                    logger.info(f"📦 Completed batch {i//batch_size + 1}. Taking {delay_between_batches}s break...")
                    logger.info(f"📊 Progress: {i+1}/{total_tickers} tickers processed")
                    logger.info(f"📈 Success: {successful}, Failed: {failed}, Skipped: {skipped}")
                    time.sleep(delay_between_batches)
                
        except KeyboardInterrupt:
            logger.warning("Update interrupted by user")
            logger.info(f"Processed {i+1}/{total_tickers} tickers before interruption")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
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
        self.db_worker.wait_for_completion()
        db_wait_end = datetime.now()
        db_wait_duration = db_wait_end - db_wait_start
        
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
    
    args = parser.parse_args()
    
    try:
        # Initialize updater
        updater = UniverseDataUpdater(provider=args.provider, timeframe=args.timeframe)
        
        if args.dry_run:
            # Show what would be processed
            tickers = updater.get_universe_tickers(force_refresh=args.force_refresh)
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
            skip_existing=args.skip_existing
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
                logger.info("🧹 Cleaned up IBKR connection in finally block")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up IBKR connection in finally block: {cleanup_error}")
        
        # Ensure TimescaleDB client is cleaned up
        try:
            from utils.timescaledb_client import close_timescaledb_client
            close_timescaledb_client()
            logger.info("🧹 Cleaned up TimescaleDB client in finally block")
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up TimescaleDB client in finally block: {cleanup_error}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
