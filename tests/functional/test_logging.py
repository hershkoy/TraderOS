#!/usr/bin/env python3
"""
Test script for the enhanced logging functionality
"""

import sys
import os
sys.path.append('.')

from utils.screener_zero_cost_collar_enhanced import CollarScreenerConfig, parse_arguments

def test_config_loading():
    """Test configuration loading with log file options"""
    print("Testing configuration loading...")
    
    # Test default config
    config = CollarScreenerConfig()
    print(f"LOG_TO_FILE: {config.LOG_TO_FILE}")
    print(f"LOG_FILENAME: {config.LOG_FILENAME}")
    print(f"LOG_LEVEL: {config.LOG_LEVEL}")
    
    return True

def test_argument_parsing():
    """Test command line argument parsing"""
    print("\nTesting argument parsing...")
    
    # Test default arguments
    sys.argv = ['test_logging.py']
    args = parse_arguments()
    print(f"Default log level: {args.log_level}")
    print(f"Default config: {args.config}")
    print(f"Verbose: {args.verbose}")
    print(f"Log file: {args.log_file}")
    print(f"No log file: {args.no_log_file}")
    
    return True

def test_log_file_override():
    """Test log file override functionality"""
    print("\nTesting log file override...")
    
    # Test with custom log file
    sys.argv = ['test_logging.py', '--log-file', 'test_log.txt']
    args = parse_arguments()
    print(f"Custom log file: {args.log_file}")
    
    # Test with no log file
    sys.argv = ['test_logging.py', '--no-log-file']
    args = parse_arguments()
    print(f"No log file flag: {args.no_log_file}")
    
    return True

if __name__ == '__main__':
    print("Testing Enhanced Zero-Cost Collar Screener Logging")
    print("=" * 50)
    
    try:
        test_config_loading()
        test_argument_parsing()
        test_log_file_override()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

