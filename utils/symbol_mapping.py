#!/usr/bin/env python3
"""
Symbol Mapping Utility

This script helps map between different symbol formats (Wikipedia, IB, etc.)
and find the correct symbols for Interactive Brokers.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.fetch_data import get_ib_connection
    from ib_insync import Stock, Contract
except ImportError:
    print("Failed to import required modules. Make sure ib_insync is installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_symbol_variations(symbol):
    """Test different symbol variations to find the correct IB symbol"""
    logger.info(f"Testing symbol variations for: {symbol}")
    
    # Common variations to try
    variations = [
        symbol,                    # Original: BF.B
        symbol.replace('.', ' '), # BF B (space - this is what IB actually uses!)
        symbol.replace('.', '-'), # BF-B
        symbol.replace('.', ''),  # BFB
        symbol.replace('.', '/'), # BF/B
        symbol.replace('.', '_'), # BF_B
        symbol.split('.')[0],     # BF (first part only)
        symbol.split('.')[1] if '.' in symbol else None,  # B (second part only)
    ]
    
    # Remove None values
    variations = [v for v in variations if v is not None]
    
    logger.info(f"Testing variations: {variations}")
    
    try:
        # Get IB connection
        ib = get_ib_connection()
        if not ib.isConnected():
            logger.error("Failed to connect to IB")
            return None
        
        # Test each variation
        for variation in variations:
            try:
                logger.info(f"Testing variation: {variation}")
                
                # Create contract
                contract = Stock(symbol=variation, exchange='SMART', currency='USD')
                
                # Request contract details
                ib.qualifyContracts(contract)
                
                # If we get here, the symbol is valid
                logger.info(f"[SUCCESS] Found valid IB symbol: {variation}")
                logger.info(f"Contract details: {contract}")
                
                # Get more details
                try:
                    # Try to get historical data to confirm it's tradeable
                    bars = ib.reqHistoricalData(
                        contract,
                        endDateTime='',
                        durationStr='1 D',
                        barSizeSetting='1 hour',
                        whatToShow='TRADES',
                        useRTH=True,
                        formatDate=1
                    )
                    
                    if bars:
                        logger.info(f"[CONFIRMED] Symbol {variation} is tradeable with {len(bars)} bars")
                    else:
                        logger.info(f"[WARNING] Symbol {variation} exists but has no data")
                        
                except Exception as data_error:
                    logger.warning(f"[WARNING] Symbol {variation} exists but data fetch failed: {data_error}")
                
                return variation
                
            except Exception as e:
                error_str = str(e).lower()
                if "no security definition" in error_str:
                    logger.info(f"[SKIP] {variation} - no security definition")
                elif "contract not found" in error_str:
                    logger.info(f"[SKIP] {variation} - contract not found")
                else:
                    logger.info(f"[SKIP] {variation} - error: {e}")
                continue
        
        logger.warning(f"[FAILED] No valid IB symbol found for {symbol}")
        return None
        
    except Exception as e:
        logger.error(f"Error testing symbol variations: {e}")
        return None
    finally:
        try:
            ib.disconnect()
        except:
            pass

def find_ib_symbols_for_universe():
    """Find IB symbols for common problematic symbols in the universe"""
    problematic_symbols = [
        "BF.B",    # Brown-Forman
        "BRK.B",   # Berkshire Hathaway
        "BF.B",    # Brown-Forman (duplicate)
        "BRK.B",   # Berkshire Hathaway (duplicate)
    ]
    
    results = {}
    
    for symbol in problematic_symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing symbol: {symbol}")
        logger.info(f"{'='*50}")
        
        ib_symbol = test_symbol_variations(symbol)
        results[symbol] = ib_symbol
        
        if ib_symbol:
            logger.info(f"[MAPPING] {symbol} -> {ib_symbol}")
        else:
            logger.info(f"[MAPPING] {symbol} -> NOT FOUND")
    
    return results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test symbol variations to find correct IB symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--symbol', 
        default=None,
        help='Specific symbol to test (e.g., BF.B)'
    )
    
    parser.add_argument(
        '--universe', 
        action='store_true',
        help='Test common problematic symbols from the universe'
    )
    
    args = parser.parse_args()
    
    if args.symbol:
        # Test specific symbol
        logger.info(f"Testing specific symbol: {args.symbol}")
        result = test_symbol_variations(args.symbol)
        if result:
            logger.info(f"[RESULT] {args.symbol} -> {result}")
        else:
            logger.info(f"[RESULT] {args.symbol} -> NOT FOUND")
    
    elif args.universe:
        # Test universe symbols
        logger.info("Testing common problematic symbols from universe")
        results = find_ib_symbols_for_universe()
        
        logger.info(f"\n{'='*50}")
        logger.info("SUMMARY OF RESULTS")
        logger.info(f"{'='*50}")
        for original, ib_symbol in results.items():
            if ib_symbol:
                logger.info(f"✅ {original} -> {ib_symbol}")
            else:
                logger.error(f"❌ {original} -> NOT FOUND")
    
    else:
        # Default: test BF.B
        logger.info("Testing default symbol: BF.B")
        result = test_symbol_variations("BF.B")
        if result:
            logger.info(f"[RESULT] BF.B -> {result}")
        else:
            logger.info(f"[RESULT] BF.B -> NOT FOUND")

if __name__ == "__main__":
    main()
