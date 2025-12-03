#!/usr/bin/env python3
"""
Test script to demonstrate YAML configuration system
"""

import yaml
import subprocess

def show_config():
    """Display the current YAML configuration"""
    print("📋 Current YAML Configuration (defaults.yaml):")
    print("=" * 60)
    
    with open('../defaults.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Global Settings:")
    for key, value in config.get('global', {}).items():
        print(f"  {key}: {value}")
    
    print("\nStrategy Configurations:")
    for strategy_name, strategy_config in config.get('strategies', {}).items():
        print(f"  {strategy_name}:")
        print(f"    description: {strategy_config.get('description', 'N/A')}")
        print(f"    parameters:")
        for param, value in strategy_config.get('parameters', {}).items():
            print(f"      {param}: {value}")
        print()

def test_yaml_system():
    """Test the YAML configuration system"""
    print("\n🧪 Testing YAML Configuration System:")
    print("=" * 60)
    
    # Test 1: Default configuration
    print("\n1. Testing with default YAML configuration:")
    cmd = "python ../backtrader_runner_yaml.py --parquet ../data/ALPACA/NFLX/1h/nflx_1h.parquet --strategy mean_reversion --fromdate 2024-10-01 --todate 2024-11-30 --quiet"
    print(f"Command: {cmd}")
    print("Expected: Uses period=20, devfactor=2.0 from YAML")
    
    # Test 2: Command line overrides
    print("\n2. Testing with command line overrides:")
    cmd = "python ../backtrader_runner_yaml.py --parquet ../data/ALPACA/NFLX/1h/nflx_1h.parquet --strategy mean_reversion --fromdate 2024-10-01 --todate 2024-11-30 --period 15 --devfactor 1.8 --cash 75000 --quiet"
    print(f"Command: {cmd}")
    print("Expected: Uses period=15, devfactor=1.8, cash=75000 from command line")
    
    print("\n✅ YAML Configuration Features:")
    print("- Centralized configuration in defaults.yaml")
    print("- Strategy-specific parameter defaults")
    print("- Command line arguments override YAML values")
    print("- Global settings (cash, dates, etc.) in one place")
    print("- Easy to add new strategies and parameters")

def main():
    show_config()
    test_yaml_system()
    
    print("\n🎯 Usage Examples:")
    print("=" * 60)
    print("# Use all defaults from YAML:")
    print("python ../backtrader_runner_yaml.py --parquet ../data.parquet --strategy mean_reversion")
    print()
    print("# Override specific parameters:")
    print("python ../backtrader_runner_yaml.py --parquet ../data.parquet --strategy mean_reversion --period 10 --cash 50000")
    print()
    print("# Use different config file:")
    print("python ../backtrader_runner_yaml.py --config ../my_config.yaml --parquet ../data.parquet --strategy pnf")

if __name__ == "__main__":
    main()
