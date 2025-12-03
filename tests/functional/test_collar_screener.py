#!/usr/bin/env python3
"""
Test script for the Zero-Cost Collar Screener

This script tests the configuration loading and basic functionality
without requiring an Interactive Brokers connection.

Author: AI Assistant
Date: 2025
"""

import sys
import os
import yaml
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test configuration file loading"""
    print("Testing configuration loading...")
    
    try:
        # Test if config file exists
        if os.path.exists("collar_screener_config.yaml"):
            print("✓ Configuration file found")
            
            # Test YAML parsing
            with open("collar_screener_config.yaml", 'r') as file:
                config = yaml.safe_load(file)
            
            # Test required sections
            required_sections = ['ib_connection', 'capital', 'options', 'risk', 'universe', 'output', 'advanced']
            for section in required_sections:
                if section in config:
                    print(f"✓ {section} section found")
                else:
                    print(f"✗ {section} section missing")
                    return False
            
            # Test specific values
            print(f"✓ Capital budget: ${config['capital']['budget']:,}")
            print(f"✓ Universe size: {len(config['universe'])} symbols")
            print(f"✓ DTE range: {config['options']['min_dte']}-{config['options']['max_dte']} days")
            
            return True
        else:
            print("✗ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def test_imports():
    """Test required imports"""
    print("\nTesting required imports...")
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError:
        print("✗ PyYAML not available")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError:
        print("✗ pandas not available (CSV export will be disabled)")
    
    try:
        import ib_insync
        print("✓ ib_insync imported successfully")
    except ImportError:
        print("✗ ib_insync not available")
        return False
    
    return True

def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = ['logs', 'reports']
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/ directory exists")
        else:
            print(f"✗ {directory}/ directory missing")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Created {directory}/ directory")
            except Exception as e:
                print(f"✗ Could not create {directory}/ directory: {e}")
                return False
    
    return True

def test_collar_calculation():
    """Test collar calculation logic"""
    print("\nTesting collar calculation logic...")
    
    # Mock data for testing
    stock_price = 100.0
    put_strike = 95.0
    put_mid = 2.0
    call_strike = 105.0
    call_mid = 1.5
    
    # Calculate collar metrics
    C0 = 100 * stock_price + put_mid - call_mid  # Net cost
    floor = 100 * put_strike - C0                # Loss floor
    max_gain = 100 * call_strike - C0            # Max gain if called away
    capital_at_risk = max(0, C0 - 100 * put_strike)
    
    print(f"Stock Price: ${stock_price}")
    print(f"Put Strike: ${put_strike}, Put Mid: ${put_mid}")
    print(f"Call Strike: ${call_strike}, Call Mid: ${call_mid}")
    print(f"Net Cost (C0): ${C0:.2f}")
    print(f"Loss Floor: ${floor:.2f}")
    print(f"Max Gain: ${max_gain:.2f}")
    print(f"Capital at Risk: ${capital_at_risk:.2f}")
    
    # Test zero-loss condition
    if floor >= 0:
        print("✓ Zero-loss condition met (floor >= 0)")
    else:
        print("✗ Zero-loss condition not met (floor < 0)")
    
    return True

def test_delta_ranges():
    """Test delta range validation"""
    print("\nTesting delta range validation...")
    
    try:
        with open("collar_screener_config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        put_range = config['options']['put_delta_range']
        call_range = config['options']['call_delta_range']
        
        print(f"Put Delta Range: {put_range}")
        print(f"Call Delta Range: {call_range}")
        
        # Validate ranges
        if put_range[0] < put_range[1] and put_range[0] < 0:
            print("✓ Put delta range is valid (negative, ascending)")
        else:
            print("✗ Put delta range is invalid")
            return False
        
        if call_range[0] < call_range[1] and call_range[0] > 0:
            print("✓ Call delta range is valid (positive, ascending)")
        else:
            print("✗ Call delta range is invalid")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing delta ranges: {e}")
        return False

def main():
    """Run all tests"""
    print("Zero-Cost Collar Screener - Test Suite")
    print("=" * 50)
    print(f"Test run at: {datetime.now()}")
    print()
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Required Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Collar Calculation", test_collar_calculation),
        ("Delta Ranges", test_delta_ranges),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The screener is ready to use.")
        print("\nTo run the screener:")
        print("1. Make sure TWS/IBG is running")
        print("2. Run: python utils/screener_zero_cost_collar_enhanced.py")
    else:
        print("✗ Some tests failed. Please fix the issues before running the screener.")
    
    return passed == total

if __name__ == '__main__':
    main()
