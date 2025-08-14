#!/usr/bin/env python3
"""
Example script demonstrating fetch_data.py functionality
"""

import subprocess
import sys
from pathlib import Path

def run_fetch_command(symbol, provider, timeframe, bars, description):
    """Run a fetch_data.py command and display the result."""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        "utils/fetch_data.py",
        "--symbol", symbol,
        "--provider", provider,
        "--timeframe", timeframe,
        "--bars", str(bars)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Description: {description}")
    print("\nNote: This is a demonstration. To actually run the command:")
    print(f"  {' '.join(cmd)}")
    
    return cmd

def main():
    """Demonstrate various fetch_data.py usage patterns."""
    print("Fetch Data Examples")
    print("=" * 60)
    
    examples = [
        {
            "symbol": "NFLX",
            "provider": "alpaca", 
            "timeframe": "1h",
            "bars": "max",
            "description": "Fetch maximum available 1-hour bars for Netflix from Alpaca"
        },
        {
            "symbol": "AAPL",
            "provider": "alpaca",
            "timeframe": "1d", 
            "bars": "1000",
            "description": "Fetch 1000 daily bars for Apple from Alpaca"
        },
        {
            "symbol": "TSLA",
            "provider": "ib",
            "timeframe": "1h",
            "bars": "max",
            "description": "Fetch maximum available 1-hour bars for Tesla from IBKR"
        },
        {
            "symbol": "MSFT",
            "provider": "ib",
            "timeframe": "1d",
            "bars": "3000",
            "description": "Fetch 3000 daily bars for Microsoft from IBKR"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        cmd = run_fetch_command(
            example["symbol"],
            example["provider"], 
            example["timeframe"],
            example["bars"],
            example["description"]
        )
    
    print(f"\n{'='*60}")
    print("Key Features Demonstrated:")
    print("✓ --bars max: Fetches maximum available historical data")
    print("✓ Automatic pagination: Loops through multiple requests")
    print("✓ Rate limiting: Respects API limits with delays")
    print("✓ Error handling: Retries failed requests")
    print("✓ Data deduplication: Removes duplicate data points")
    print("✓ TimescaleDB storage: Saves to optimized time-series database")
    print("="*60)
    
    print(f"\nPrerequisites:")
    print("1. Set up .env file with API credentials:")
    print("   ALPACA_API_KEY_ID=your_key")
    print("   ALPACA_API_SECRET=your_secret")
    print("2. For IBKR: Ensure TWS/IB Gateway is running on localhost:4001")
    print("3. Install dependencies: pip install python-dotenv alpaca-py ib_insync")

if __name__ == "__main__":
    main()
