#!/usr/bin/env python3
"""
Test script to demonstrate the multi-strategy architecture
"""

import subprocess
import sys
from pathlib import Path

def run_strategy_test(strategy_name, extra_args=""):
    """Run a strategy test and capture output"""
    cmd = f"python backtrader_runner.py --parquet data/ALPACA/NFLX/1h/nflx_1h.parquet --strategy {strategy_name} --fromdate 2024-10-01 --todate 2024-11-30 --quiet {extra_args}"
    
    print(f"\n🧪 Testing {strategy_name} strategy...")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Error running {strategy_name}: {e}")

def main():
    print("🚀 Multi-Strategy Backtrader Testing")
    print("=" * 60)
    
    # Test mean reversion strategy
    run_strategy_test("mean_reversion", "--period 15 --devfactor 1.8")
    
    # Test PnF strategy (if we fix the datetime issue)
    # run_strategy_test("pnf", "--box 2.0 --reversal 3")
    
    print("\n✅ Strategy testing completed!")
    print("\nAvailable strategies:")
    print("- mean_reversion: Simple Mean Reversion using Bollinger Bands on daily data")
    print("- pnf: Point & Figure Multi-Timeframe (needs datetime fix)")

if __name__ == "__main__":
    main()
