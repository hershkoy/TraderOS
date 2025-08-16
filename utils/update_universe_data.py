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
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import argparse

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
    
    def fetch_ticker_data(self, symbol: str) -> bool:
        """
        Fetch data for a single ticker
        
        Args:
            symbol: Stock symbol to fetch
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching {self.timeframe} data for {symbol} from {self.provider}...")
            
            # Use large fixed numbers instead of "max" to avoid complex logic issues
            if self.provider == "alpaca":
                # For daily data: 5 years = ~1260 bars, For hourly data: 1 year = ~8760 bars
                bars = 1260 if self.timeframe == "1d" else 8760
                result = fetch_from_alpaca(symbol, bars, self.timeframe)
            elif self.provider == "ib":
                # For daily data: 5 years = ~1260 bars, For hourly data: 1 year = ~8760 bars
                bars = 1260 if self.timeframe == "1d" else 8760
                result = fetch_from_ib(symbol, bars, self.timeframe)
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return False
            
            if result is not None and not result.empty:
                logger.info(f"✅ Successfully fetched {len(result)} bars for {symbol}")
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
                           start_from_index: int = 0) -> Dict[str, Any]:
        """
        Update market data for all tickers in the universe
        
        Args:
            batch_size: Number of tickers to process before taking a break
            delay_between_batches: Seconds to wait between batches
            delay_between_tickers: Seconds to wait between individual tickers
            max_tickers: Maximum number of tickers to process (None for all)
            start_from_index: Index to start processing from (for resuming)
            
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
        logger.info(f"Batch size: {batch_size}, Delays: {delay_between_batches}s between batches, {delay_between_tickers}s between tickers")
        
        # Statistics
        successful = 0
        failed = 0
        failed_symbols = []
        
        start_time = datetime.now()
        
        try:
            for i, symbol in enumerate(tickers):
                current_index = start_from_index + i
                
                logger.info(f"Processing {symbol} ({i+1}/{total_tickers}, overall index: {current_index})")
                
                # Fetch data for this ticker
                if self.fetch_ticker_data(symbol):
                    successful += 1
                else:
                    failed += 1
                    failed_symbols.append(symbol)
                
                # Delay between tickers (except for the last one)
                if i < total_tickers - 1:
                    time.sleep(delay_between_tickers)
                
                # Batch processing with delays
                if (i + 1) % batch_size == 0 and i < total_tickers - 1:
                    logger.info(f"Completed batch {i//batch_size + 1}. Taking {delay_between_batches}s break...")
                    logger.info(f"Progress: {i+1}/{total_tickers} tickers processed")
                    logger.info(f"Success: {successful}, Failed: {failed}")
                    time.sleep(delay_between_batches)
                
        except KeyboardInterrupt:
            logger.warning("Update interrupted by user")
            logger.info(f"Processed {i+1}/{total_tickers} tickers before interruption")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Compile results
        results = {
            "total_tickers": total_tickers,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_tickers * 100) if total_tickers > 0 else 0,
            "failed_symbols": failed_symbols,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "provider": self.provider,
            "timeframe": self.timeframe,
            "last_processed_index": start_from_index + total_tickers - 1
        }
        
        # Log summary
        logger.info("=" * 60)
        logger.info("UNIVERSE UPDATE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total tickers processed: {total_tickers}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")
        logger.info(f"Duration: {duration}")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"Timeframe: {self.timeframe}")
        
        if failed_symbols:
            logger.info(f"Failed symbols: {', '.join(failed_symbols[:10])}")
            if len(failed_symbols) > 10:
                logger.info(f"... and {len(failed_symbols) - 10} more")
        
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
            print(f"First 10 tickers: {tickers[:10]}")
            if len(tickers) > 10:
                print(f"Last 10 tickers: {tickers[-10:]}")
            return 0
        
        # Determine start index
        start_index = args.resume_from if args.resume_from is not None else 0
        
        # Run the update
        results = updater.update_universe_data(
            batch_size=args.batch_size,
            delay_between_batches=args.delay_batches,
            delay_between_tickers=args.delay_tickers,
            max_tickers=args.max_tickers,
            start_from_index=start_index
        )
        
        if "error" in results:
            logger.error(f"Update failed: {results['error']}")
            return 1
        
        # Save results summary
        summary_file = f"universe_update_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("UNIVERSE UPDATE SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Provider: {results['provider']}\n")
            f.write(f"Timeframe: {results['timeframe']}\n")
            f.write(f"Total tickers: {results['total_tickers']}\n")
            f.write(f"Successful: {results['successful']}\n")
            f.write(f"Failed: {results['failed']}\n")
            f.write(f"Success rate: {results['success_rate']:.1f}%\n")
            f.write(f"Duration: {results['duration']}\n")
            f.write(f"Start time: {results['start_time']}\n")
            f.write(f"End time: {results['end_time']}\n")
            f.write(f"Last processed index: {results['last_processed_index']}\n")
            
            if results['failed_symbols']:
                f.write(f"\nFailed symbols:\n")
                for symbol in results['failed_symbols']:
                    f.write(f"  {symbol}\n")
        
        logger.info(f"Results summary saved to: {summary_file}")
        logger.info("Update completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Update failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
