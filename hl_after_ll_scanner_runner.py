#!/usr/bin/env python3
"""
HL After LL Scanner Runner with Configuration
Scans for LL → HH → HL patterns using YAML configuration
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.scanning.hl_after_ll import scan_symbol_for_setup, scan_universe, load_from_timescaledb
from utils.data.ticker_universe import get_ticker_universe

class HLAfterLLScanner:
    """HL After LL Scanner with configuration support"""
    
    def __init__(self, config_path: str = "hl_after_ll_scanner_config.yaml"):
        """Initialize scanner with configuration"""
        self.config = self.load_config(config_path)
        self.logger = self.setup_logging()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'scanner': {
                'left_bars': 1,
                'right_bars': 2,
                'patterns': ['LL→HH→HL', 'LL→HH→HH→HL'],
                'monday_check': {'enabled': True},
                'data': {'timeframe': '1d', 'min_weeks': 52, 'max_weeks': 260}
            },
            'output': {
                'console': {'show_summary': True, 'show_details': True, 'show_monday_check': True},
                'files': {'save_csv': True, 'save_log': True}
            },
            'universe': {'source': 'database'},
            'database': {
                'host': 'localhost', 'port': 5432, 'database': 'backtrader',
                'user': 'backtrader_user', 'password': 'backtrader_password'
            },
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(message)s'}
        }
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        
        # Create logs directory
        log_dir = 'logs/scanners/hl_after_ll'
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup handlers
        handlers = [logging.StreamHandler()]
        
        if self.config.get('output', {}).get('files', {}).get('save_log', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            handlers.append(logging.FileHandler(f'{log_dir}/hl_after_ll_scanner_{timestamp}.log'))
        
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=handlers
        )
        
        return logging.getLogger(__name__)
    
    def get_symbols(self) -> List[str]:
        """Get symbols to scan based on configuration"""
        universe_config = self.config.get('universe', {})
        source = universe_config.get('source', 'database')
        
        if source == 'database':
            self.logger.info("Loading ticker universe from database...")
            universe = get_ticker_universe()
            symbols = list(universe.keys())
            self.logger.info(f"Loaded {len(symbols)} symbols from universe")
            return symbols
        
        elif source == 'file':
            file_path = universe_config.get('file_path', 'data/ticker_universe.csv')
            # TODO: Implement file loading
            self.logger.warning("File source not implemented yet")
            return []
        
        elif source == 'manual':
            symbols = universe_config.get('symbols', [])
            self.logger.info(f"Using {len(symbols)} manually specified symbols")
            return symbols
        
        else:
            self.logger.error(f"Unknown universe source: {source}")
            return []
    
    def scan_symbols(self, symbols: List[str]) -> List[Any]:
        """Scan symbols for LL → HH → HL patterns"""
        scanner_config = self.config.get('scanner', {})
        data_config = scanner_config.get('data', {})
        timeframe = data_config.get('timeframe', '1d')
        
        self.logger.info(f"Loading {timeframe} data from TimescaleDB for {len(symbols)} symbols...")
        
        try:
            # Load data from TimescaleDB
            ohlcv_data = load_from_timescaledb(symbols, timeframe)
            self.logger.info(f"Successfully loaded data for {len(ohlcv_data)} symbols")
            
            if not ohlcv_data:
                self.logger.warning("No data loaded from TimescaleDB")
                return []
            
            # Scan for patterns
            self.logger.info("Scanning for LL → HH → HL patterns...")
            matches = scan_universe(ohlcv_data)
            
            self.logger.info(f"Found {len(matches)} symbols with LL → HH → HL patterns")
            return matches
            
        except Exception as e:
            self.logger.error(f"Error during scanning: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def print_results(self, matches: List[Any]):
        """Print scan results"""
        output_config = self.config.get('output', {}).get('console', {})
        
        if not matches:
            print("\n❌ No LL → HH → HL patterns found")
            return
        
        if output_config.get('show_summary', True):
            print(f"\n✅ Found {len(matches)} symbols with LL → HH → HL patterns:")
            print("=" * 80)
        
        for i, match in enumerate(matches, 1):
            if output_config.get('show_details', True):
                print(f"\n{i}. {match.symbol}")
                print(f"   LL: {match.ll_date.strftime('%Y-%m-%d')} at ${match.ll_price:.2f}")
                
                # Print HH details
                for j, (hh_date, hh_price) in enumerate(zip(match.hh_dates, match.hh_prices)):
                    print(f"   HH{j+1}: {hh_date.strftime('%Y-%m-%d')} at ${hh_price:.2f}")
                
                print(f"   HL: {match.hl_date.strftime('%Y-%m-%d')} at ${match.hl_price:.2f}")
                print(f"   Current Price: ${match.last_price:.2f}")
                
                # Monday check
                if output_config.get('show_monday_check', True):
                    if match.last_price >= match.hl_price:
                        print(f"   ✅ Monday Check: Current price (${match.last_price:.2f}) >= HL (${match.hl_price:.2f})")
                    else:
                        print(f"   ❌ Monday Check: Current price (${match.last_price:.2f}) < HL (${match.hl_price:.2f}) - Pattern broken")
    
    def save_results(self, matches: List[Any]):
        """Save results to CSV file"""
        if not matches:
            print("No matches to save")
            return
        
        output_config = self.config.get('output', {}).get('files', {})
        if not output_config.get('save_csv', True):
            return
        
        import pandas as pd
        
        # Generate filename
        filename = output_config.get('csv_filename', 'hl_after_ll_scan_{timestamp}.csv')
        if '{timestamp}' in filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filename.format(timestamp=timestamp)
        
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
        filepath = Path("reports") / filename
        
        # Prepare data for CSV
        data = []
        for match in matches:
            # Create one row per HH
            for i, (hh_date, hh_price) in enumerate(zip(match.hh_dates, match.hh_prices)):
                data.append({
                    'symbol': match.symbol,
                    'll_date': match.ll_date.strftime('%Y-%m-%d'),
                    'll_price': match.ll_price,
                    'hh_number': i + 1,
                    'hh_date': hh_date.strftime('%Y-%m-%d'),
                    'hh_price': hh_price,
                    'hl_date': match.hl_date.strftime('%Y-%m-%d'),
                    'hl_price': match.hl_price,
                    'current_price': match.last_price,
                    'monday_check_pass': match.last_price >= match.hl_price,
                    'pattern_type': f"LL→HH{'→HH' * (i)}→HL" if i > 0 else "LL→HH→HL"
                })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"\n📊 Results saved to: {filepath}")
    
    def run(self):
        """Run the complete scan"""
        print("🔍 HL After LL Scanner")
        print("=" * 50)
        print("Scanning for LL → HH → HL patterns in weekly data")
        print("Configuration: 1 left bar, 2 right bar confirmation")
        print("Monday check: Current price must be >= HL price")
        print()
        
        # Check if it's Monday (optional)
        today = datetime.now()
        if today.weekday() == 0:  # Monday
            print("✅ Running on Monday - perfect timing!")
        else:
            print(f"⚠️  Running on {today.strftime('%A')} - consider running on Monday for best results")
        
        print()
        
        # Get symbols to scan
        symbols = self.get_symbols()
        if not symbols:
            self.logger.error("No symbols to scan")
            return
        
        # Run the scan
        matches = self.scan_symbols(symbols)
        
        # Display results
        self.print_results(matches)
        
        # Save to CSV
        self.save_results(matches)
        
        print(f"\n🎯 Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="HL After LL Scanner")
    parser.add_argument('--config', type=str, default='hl_after_ll_scanner_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to scan (overrides config)')
    parser.add_argument('--test', action='store_true',
                       help='Run test with sample data')
    
    args = parser.parse_args()
    
    if args.test:
        # Run test
        from tests.test_hl_after_ll_scanner import test_scanner
        test_scanner()
        return
    
    # Initialize scanner
    scanner = HLAfterLLScanner(args.config)
    
    # Override symbols if provided
    if args.symbols:
        scanner.config['universe'] = {'source': 'manual', 'symbols': args.symbols}
    
    # Run scanner
    scanner.run()

if __name__ == "__main__":
    main()
