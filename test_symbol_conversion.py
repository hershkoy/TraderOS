#!/usr/bin/env python3
"""
Test script for symbol conversion logic
"""

def convert_symbol_for_ib(symbol):
    """Convert symbol format for IB compatibility"""
    # Known problematic symbols that need format conversion
    symbol_mappings = {
        'BF.B': 'BF B',    # Brown-Forman Class B
        'BRK.B': 'BRK B',  # Berkshire Hathaway Class B
        'BF-B': 'BF B',    # Alternative format
        'BRK-B': 'BRK B',  # Alternative format
    }
    
    # Check if symbol is in our known mappings
    if symbol in symbol_mappings:
        converted = symbol_mappings[symbol]
        print(f"[CONVERT] Converting symbol {symbol} -> {converted} for IB compatibility")
        return converted
    
    # General rule: convert dots to spaces for Class B shares
    if '.' in symbol and symbol.endswith('.B'):
        converted = symbol.replace('.', ' ')
        print(f"[CONVERT] Converting Class B symbol {symbol} -> {converted} for IB compatibility")
        return converted
    
    # Return original if no conversion needed
    return symbol

def main():
    """Test the symbol conversion logic"""
    test_symbols = [
        'BF.B',
        'BF-B', 
        'BRK.B',
        'BRK-B',
        'AAPL',
        'MSFT',
        'GOOGL'
    ]
    
    print("Testing symbol conversions:")
    print("=" * 50)
    
    for symbol in test_symbols:
        converted = convert_symbol_for_ib(symbol)
        print(f"{symbol} -> {converted}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
