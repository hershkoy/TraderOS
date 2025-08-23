#!/usr/bin/env python3
"""
Test script for the new symbol mapping system
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_symbol_mapping_system():
    """Test the symbol mapping system"""
    try:
        from utils.update_universe_data import UniverseDataUpdater
        
        print("Testing Symbol Mapping System")
        print("=" * 50)
        
        # Initialize updater
        updater = UniverseDataUpdater('ib', '1h')
        
        # Test symbol conversion
        test_symbols = ['BF.B', 'BRK.B', 'AAPL', 'MSFT']
        
        print("\nTesting symbol conversions:")
        for symbol in test_symbols:
            converted = updater._convert_symbol_for_ib(symbol)
            print(f"{symbol} -> {converted}")
        
        # Show mapping stats
        print("\nSymbol mapping statistics:")
        stats = updater.get_symbol_mapping_stats()
        if 'error' not in stats:
            print(f"Total mappings: {stats['total_mappings']}")
            print(f"Valid mappings: {stats['valid_mappings']}")
            print(f"Invalid mappings: {stats['invalid_mappings']}")
            print(f"Success rate: {stats['success_rate']:.1f}%")
        else:
            print(f"Error getting stats: {stats['error']}")
        
        # Show recent mappings
        print("\nRecent mappings:")
        mappings = updater.view_symbol_mappings(limit=10)
        for mapping in mappings:
            print(f"{mapping['original_symbol']} -> {mapping['ib_symbol']} (valid: {mapping['is_valid']}, uses: {mapping['use_count']})")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error testing symbol mapping system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_symbol_mapping_system()
