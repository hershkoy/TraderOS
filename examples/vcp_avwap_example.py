#!/usr/bin/env python3
"""
Example: VCP AVWAP Breakout Strategy
====================================

This example shows how to run the VCP AVWAP breakout strategy using your existing runner.

The strategy looks for:
1. Stocks near 52-week highs (within 8%)
2. Valid VCP (Volatility Contraction Pattern) bases (6-12 weeks)
3. Breakouts above the pivot with above-average volume
4. Entries above the anchored VWAP from the base start

Usage:
    python examples/vcp_avwap_example.py

Or use the runner directly:
    python backtrader_runner_yaml.py \
      --config defaults.yaml \
      --symbol NFLX \
      --provider ALPACA \
      --timeframe 1h \
      --strategy vcp_avwap_breakout \
      --log-level DEBUG
"""

import sys
import os

# Add the parent directory to the path so we can import the strategy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.vcp_avwap_breakout import VcpAvwapBreakoutStrategy

def main():
    """Demonstrate the VCP AVWAP strategy"""
    print("VCP AVWAP Breakout Strategy Example")
    print("=" * 40)
    
    # Show strategy information
    print(f"Strategy: {VcpAvwapBreakoutStrategy.__name__}")
    print(f"Description: {VcpAvwapBreakoutStrategy.get_description()}")
    print(f"Data Requirements: {VcpAvwapBreakoutStrategy.get_data_requirements()}")
    
    # Show strategy info
    print("\nStrategy Features:")
    print("  - VCP base detection (6-12 weeks)")
    print("  - 52-week high proximity check")
    print("  - Anchored VWAP from base start")
    print("  - Relative volume breakout detection")
    print("  - Multi-timeframe analysis (1h/daily/weekly)")
    print("  - Automatic stop management")
    print("  - Partial profit taking")
    
    print("\n" + "=" * 40)
    print("To run this strategy:")
    print("1. Use your existing runner with --strategy vcp_avwap_breakout")
    print("2. Ensure you have 1h data with daily/weekly resampling")
    print("3. The strategy will automatically detect VCP bases and AVWAP levels")
    print("4. Look for stocks near 52-week highs for best results")
    
    print("\nExample command:")
    print("python backtrader_runner_yaml.py \\")
    print("  --config defaults.yaml \\")
    print("  --symbol NFLX \\")
    print("  --provider ALPACA \\")
    print("  --timeframe 1h \\")
    print("  --strategy vcp_avwap_breakout \\")
    print("  --log-level DEBUG")

if __name__ == "__main__":
    main()
