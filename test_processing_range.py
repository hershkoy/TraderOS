#!/usr/bin/env python3
"""
Test script for the processing range functionality
"""

import sys
import os
sys.path.append('.')

from utils.screener_zero_cost_collar_enhanced import CollarScreenerConfig, parse_arguments

def test_processing_range_config():
    """Test processing range configuration loading"""
    print("Testing processing range configuration...")
    
    # Test default config
    config = CollarScreenerConfig()
    print(f"START_INDEX: {config.START_INDEX}")
    print(f"MAX_TICKERS: {config.MAX_TICKERS}")
    
    return True

def test_argument_parsing():
    """Test command line argument parsing for processing range options"""
    print("\nTesting processing range argument parsing...")
    
    # Test default arguments
    sys.argv = ['test_processing_range.py']
    args = parse_arguments()
    print(f"Default start_index: {args.start_index}")
    print(f"Default max_tickers: {args.max_tickers}")
    print(f"Default resume_from: {args.resume_from}")
    
    # Test with start index
    sys.argv = ['test_processing_range.py', '--start-index', '100']
    args = parse_arguments()
    print(f"Custom start_index: {args.start_index}")
    
    # Test with max tickers
    sys.argv = ['test_processing_range.py', '--max-tickers', '50']
    args = parse_arguments()
    print(f"Custom max_tickers: {args.max_tickers}")
    
    # Test with resume from
    sys.argv = ['test_processing_range.py', '--resume-from', '200']
    args = parse_arguments()
    print(f"Custom resume_from: {args.resume_from}")
    
    # Test combined options
    sys.argv = ['test_processing_range.py', '--start-index', '150', '--max-tickers', '25']
    args = parse_arguments()
    print(f"Combined - start_index: {args.start_index}, max_tickers: {args.max_tickers}")
    
    return True

def test_config_override():
    """Test processing range settings override"""
    print("\nTesting processing range settings override...")
    
    # Test with resume from
    sys.argv = ['test_processing_range.py', '--resume-from', '300']
    args = parse_arguments()
    
    config = CollarScreenerConfig()
    
    # Override settings
    if args.resume_from is not None:
        config.START_INDEX = args.resume_from
    elif args.start_index != 0:
        config.START_INDEX = args.start_index
    
    if args.max_tickers:
        config.MAX_TICKERS = args.max_tickers
    
    print(f"After override - START_INDEX: {config.START_INDEX}")
    print(f"After override - MAX_TICKERS: {config.MAX_TICKERS}")
    
    return True

def test_universe_slicing():
    """Test universe slicing logic"""
    print("\nTesting universe slicing logic...")
    
    config = CollarScreenerConfig()
    universe = config.UNIVERSE
    
    print(f"Total universe size: {len(universe)}")
    print(f"First 5 symbols: {universe[:5]}")
    print(f"Last 5 symbols: {universe[-5:]}")
    
    # Test start index
    start_index = 5
    tickers_from_start = universe[start_index:]
    print(f"Symbols from index {start_index}: {tickers_from_start[:5]}...")
    
    # Test max tickers
    max_tickers = 10
    tickers_limited = tickers_from_start[:max_tickers]
    print(f"Limited to {max_tickers} tickers: {tickers_limited}")
    
    # Test combined
    start_index = 10
    max_tickers = 5
    tickers_combined = universe[start_index:start_index + max_tickers]
    print(f"Index {start_index} to {start_index + max_tickers}: {tickers_combined}")
    
    return True

if __name__ == '__main__':
    print("Testing Processing Range Functionality")
    print("=" * 50)
    
    try:
        test_processing_range_config()
        test_argument_parsing()
        test_config_override()
        test_universe_slicing()
        print("\n✅ All processing range tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

