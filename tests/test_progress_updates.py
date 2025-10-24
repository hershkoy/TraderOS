#!/usr/bin/env python3
"""
Test script for the progress update functionality
"""

import sys
import os
sys.path.append('.')

from utils.screener_zero_cost_collar_enhanced import CollarScreenerConfig, parse_arguments

def test_progress_config():
    """Test progress configuration loading"""
    print("Testing progress configuration...")
    
    # Test default config
    config = CollarScreenerConfig()
    print(f"REAL_TIME_UPDATES: {config.REAL_TIME_UPDATES}")
    print(f"PROGRESS_UPDATE_FREQUENCY: {config.PROGRESS_UPDATE_FREQUENCY}")
    
    return True

def test_argument_parsing():
    """Test command line argument parsing for progress options"""
    print("\nTesting progress argument parsing...")
    
    # Test default arguments
    sys.argv = ['test_progress_updates.py']
    args = parse_arguments()
    print(f"Default progress frequency: {args.progress_frequency}")
    print(f"No real-time updates: {args.no_real_time_updates}")
    
    # Test with custom frequency
    sys.argv = ['test_progress_updates.py', '--progress-frequency', '10']
    args = parse_arguments()
    print(f"Custom progress frequency: {args.progress_frequency}")
    
    # Test with no real-time updates
    sys.argv = ['test_progress_updates.py', '--no-real-time-updates']
    args = parse_arguments()
    print(f"No real-time updates flag: {args.no_real_time_updates}")
    
    return True

def test_progress_override():
    """Test progress settings override"""
    print("\nTesting progress settings override...")
    
    # Test with no real-time updates
    sys.argv = ['test_progress_updates.py', '--no-real-time-updates', '--progress-frequency', '3']
    args = parse_arguments()
    
    config = CollarScreenerConfig()
    
    # Override settings
    if args.no_real_time_updates:
        config.REAL_TIME_UPDATES = False
    if args.progress_frequency:
        config.PROGRESS_UPDATE_FREQUENCY = args.progress_frequency
    
    print(f"After override - REAL_TIME_UPDATES: {config.REAL_TIME_UPDATES}")
    print(f"After override - PROGRESS_UPDATE_FREQUENCY: {config.PROGRESS_UPDATE_FREQUENCY}")
    
    return True

if __name__ == '__main__':
    print("Testing Progress Update Functionality")
    print("=" * 50)
    
    try:
        test_progress_config()
        test_argument_parsing()
        test_progress_override()
        print("\n✅ All progress update tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
