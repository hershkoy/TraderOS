#!/usr/bin/env python3
"""
Example usage of the Universe Data Updater

This script demonstrates how to use the UniverseDataUpdater class to fetch
maximum bars for all tickers in the ticker universe.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.update_universe_data import UniverseDataUpdater
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main example function"""
    print("🚀 Universe Data Update Example")
    print("=" * 50)
    
    try:
        # Example 1: Basic usage with Alpaca daily data
        print("\n1. Basic usage with Alpaca daily data...")
        updater = UniverseDataUpdater(provider="alpaca", timeframe="1d")
        
        # Get universe tickers first
        tickers = updater.get_universe_tickers()
        print(f"✅ Universe contains {len(tickers)} tickers")
        print(f"   First 10: {tickers[:10]}")
        
        # Example 2: Process a small batch for demonstration
        print("\n2. Processing first 5 tickers as a demo...")
        results = updater.update_universe_data(
            batch_size=2,  # Process 2 tickers at a time
            delay_between_batches=2.0,  # 2 seconds between batches
            delay_between_tickers=0.5,  # 0.5 seconds between tickers
            max_tickers=5  # Only process first 5 tickers
        )
        
        print(f"✅ Demo completed!")
        print(f"   Total processed: {results['total_tickers']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        print(f"   Success rate: {results['success_rate']:.1f}%")
        
        # Example 3: Show how to resume from a specific point
        print("\n3. Resume functionality example...")
        if results['last_processed_index'] >= 0:
            print(f"   Last processed index: {results['last_processed_index']}")
            print(f"   To resume from here, use: --resume-from {results['last_processed_index'] + 1}")
        
        # Example 4: Different provider and timeframe
        print("\n4. IBKR hourly data example (dry run)...")
        ib_updater = UniverseDataUpdater(provider="ib", timeframe="1h")
        ib_tickers = ib_updater.get_universe_tickers()
        print(f"   Would process {len(ib_tickers)} tickers with IBKR hourly data")
        
        # Example 5: Command line usage examples
        print("\n5. Command line usage examples:")
        print("   # Update all tickers with Alpaca daily data")
        print("   python utils/update_universe_data.py --provider alpaca --timeframe 1d")
        print("")
        print("   # Update with hourly data, process in batches of 20")
        print("   python utils/update_universe_data.py --provider alpaca --timeframe 1h --batch-size 20")
        print("")
        print("   # Resume from a specific index")
        print(f"   python utils/update_universe_data.py --provider alpaca --timeframe 1d --resume-from {results['last_processed_index'] + 1}")
        print("")
        print("   # Process only first 100 tickers")
        print("   python utils/update_universe_data.py --provider alpaca --timeframe 1d --max-tickers 100")
        print("")
        print("   # Dry run to see what would be processed")
        print("   python utils/update_universe_data.py --provider alpaca --timeframe 1d --dry-run")
        
        print("\n🎉 Example completed successfully!")
        print("\n💡 Tips:")
        print("   - Start with small batches to test your setup")
        print("   - Use --dry-run to see what would be processed")
        print("   - Monitor the log file: universe_update.log")
        print("   - Use --resume-from to continue interrupted updates")
        print("   - Adjust delays based on your API rate limits")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        logging.error(f"Example failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
