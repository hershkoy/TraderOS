#!/usr/bin/env python3
"""
Unified Scanner Runner
Supports multiple scanner types with consistent interface
"""

import os
import sys
import yaml
import logging
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import scanner modules
from utils.hl_after_ll_scanner import scan_symbol_for_setup, scan_universe, load_from_timescaledb
from utils.ticker_universe import get_combined_universe
from utils.timescaledb_client import get_timescaledb_client

class ScannerRunner:
    """Unified scanner runner supporting multiple scanner types"""
    
    def __init__(self, config_path: str = "scanner_config.yaml"):
        """Initialize scanner runner with configuration"""
        self.config = self.load_config(config_path)
        self.logger = self.setup_logging()
        self.scanner_type = None
        
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
                'type': 'hl_after_ll',
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
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(message)s'},
            'auto_update': {
                'enabled': True,
                'max_days_old': 1,
                'provider': 'ALPACA',
                'timeframe': '1d',
                'days_back': 30
            }
        }
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup handlers
        handlers = [logging.StreamHandler()]
        
        if self.config.get('output', {}).get('files', {}).get('save_log', True):
            handlers.append(logging.FileHandler('logs/scanner_runner.log'))
        
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
            symbols = get_combined_universe()
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
    
    def check_data_freshness(self, symbols: List[str]) -> Dict[str, Any]:
        """Check how fresh the data is for given symbols"""
        auto_update_config = self.config.get('auto_update', {})
        if not auto_update_config.get('enabled', True):
            return {'needs_update': False, 'stale_symbols': [], 'missing_symbols': []}
        
        provider = auto_update_config.get('provider', 'ALPACA')
        timeframe = auto_update_config.get('timeframe', '1d')
        max_days_old = auto_update_config.get('max_days_old', 1)
        
        self.logger.info(f"Checking data freshness for {len(symbols)} symbols...")
        
        try:
            client = get_timescaledb_client()
            if not client.ensure_connection():
                self.logger.error("Cannot connect to TimescaleDB for data freshness check")
                return {'needs_update': False, 'stale_symbols': [], 'missing_symbols': []}
            
            stale_symbols = []
            missing_symbols = []
            fresh_symbols = []
            
            for symbol in symbols:
                try:
                    # Get latest data for this symbol using the client's method
                    df = client.get_market_data(symbol, timeframe, provider)
                    
                    if df is not None and not df.empty:
                        # Get the latest timestamp from the data
                        latest_date = df.index.max()
                        
                        # Calculate days since latest data
                        if hasattr(latest_date, 'tz') and latest_date.tz is not None:
                            latest_date = latest_date.tz_convert(None)
                        
                        # Ensure we have datetime objects for comparison
                        if isinstance(latest_date, pd.Timestamp):
                            latest_date = latest_date.to_pydatetime()
                        
                        days_old = (datetime.now() - latest_date).days
                        record_count = len(df)
                        
                        if days_old > max_days_old:
                            stale_symbols.append((symbol, latest_date, days_old, record_count))
                        else:
                            fresh_symbols.append((symbol, latest_date, days_old, record_count))
                    else:
                        missing_symbols.append(symbol)
                        
                except Exception as e:
                    self.logger.warning(f"Error checking data freshness for {symbol}: {e}")
                    missing_symbols.append(symbol)
            
            client.disconnect()
            
            needs_update = len(stale_symbols) > 0 or len(missing_symbols) > 0
            
            self.logger.info(f"Data freshness check complete:")
            self.logger.info(f"  Fresh: {len(fresh_symbols)} symbols")
            self.logger.info(f"  Stale: {len(stale_symbols)} symbols")
            self.logger.info(f"  Missing: {len(missing_symbols)} symbols")
            self.logger.info(f"  Needs update: {needs_update}")
            
            return {
                'needs_update': needs_update,
                'stale_symbols': stale_symbols,
                'missing_symbols': missing_symbols,
                'fresh_symbols': fresh_symbols
            }
            
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
            return {'needs_update': False, 'stale_symbols': [], 'missing_symbols': []}
    
    def update_stale_data(self, stale_symbols: List[tuple], missing_symbols: List[str]) -> bool:
        """Update stale and missing data"""
        auto_update_config = self.config.get('auto_update', {})
        provider = auto_update_config.get('provider', 'ALPACA')
        timeframe = auto_update_config.get('timeframe', '1d')
        days_back = auto_update_config.get('days_back', 30)
        
        symbols_to_update = [s[0] for s in stale_symbols] + missing_symbols
        
        if not symbols_to_update:
            return True
        
        self.logger.info(f"Updating data for {len(symbols_to_update)} symbols from {provider}")
        
        successful_updates = 0
        failed_updates = 0
        
        for i, symbol in enumerate(symbols_to_update, 1):
            try:
                self.logger.info(f"  [{i}/{len(symbols_to_update)}] Updating {symbol}...")
                
                # Calculate bars needed
                if timeframe == "1d":
                    bars_needed = days_back + 10  # Extra buffer
                else:  # 1h
                    bars_needed = days_back * 7 + 50  # Extra buffer for hourly
                
                # Fetch data directly and save to TimescaleDB
                try:
                    # Import the fetch function and save the data
                    import sys
                    sys.path.append('utils')
                    from fetch_data import fetch_from_alpaca, fetch_from_ib
                    
                    # Fetch the data directly
                    if provider.lower() == 'alpaca':
                        df = fetch_from_alpaca(symbol, bars_needed, timeframe)
                    else:
                        df = fetch_from_ib(symbol, bars_needed, timeframe)
                    
                    if df is not None and not df.empty:
                        # Save to TimescaleDB
                        client = get_timescaledb_client()
                        if client.ensure_connection():
                            success = client.insert_market_data(df, symbol, provider, timeframe)
                            client.disconnect()
                            
                            if success:
                                successful_updates += 1
                                self.logger.info(f"  ✅ {symbol} - Data updated and saved successfully")
                            else:
                                failed_updates += 1
                                self.logger.warning(f"  ❌ {symbol} - Data fetched but failed to save to database")
                        else:
                            failed_updates += 1
                            self.logger.warning(f"  ❌ {symbol} - Could not connect to database")
                    else:
                        failed_updates += 1
                        self.logger.warning(f"  ❌ {symbol} - No data returned from fetch")
                        
                except Exception as e:
                    failed_updates += 1
                    self.logger.warning(f"  ❌ {symbol} - Error fetching/saving data: {e}")
                    
            except Exception as e:
                failed_updates += 1
                self.logger.warning(f"  ❌ {symbol} - Error: {e}")
        
        self.logger.info(f"Data update summary:")
        self.logger.info(f"  ✅ Successful: {successful_updates}")
        self.logger.info(f"  ❌ Failed: {failed_updates}")
        self.logger.info(f"  📈 Success Rate: {successful_updates/(successful_updates+failed_updates)*100:.1f}%")
        
        return successful_updates > 0
    
    def run_hl_after_ll_scanner(self, symbols: List[str]) -> List[Any]:
        """Run HL After LL scanner"""
        scanner_config = self.config.get('scanner', {})
        data_config = scanner_config.get('data', {})
        timeframe = data_config.get('timeframe', '1d')
        
        self.logger.info(f"Running HL After LL scanner on {len(symbols)} symbols")
        self.logger.info(f"Loading {timeframe} data from TimescaleDB...")
        
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
            self.logger.error(f"Error during HL After LL scanning: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def run_vcp_scanner(self, symbols: List[str]) -> List[Any]:
        """Run VCP scanner (placeholder for future implementation)"""
        self.logger.info("VCP scanner not implemented yet")
        return []
    
    def run_liquidity_sweep_scanner(self, symbols: List[str]) -> List[Any]:
        """Run Liquidity Sweep scanner (placeholder for future implementation)"""
        self.logger.info("Liquidity Sweep scanner not implemented yet")
        return []
    
    def run_scanner(self, scanner_type: str, symbols: List[str]) -> List[Any]:
        """Run specified scanner type"""
        self.scanner_type = scanner_type
        
        if scanner_type == 'hl_after_ll':
            return self.run_hl_after_ll_scanner(symbols)
        elif scanner_type == 'vcp':
            return self.run_vcp_scanner(symbols)
        elif scanner_type == 'liquidity_sweep':
            return self.run_liquidity_sweep_scanner(symbols)
        else:
            self.logger.error(f"Unknown scanner type: {scanner_type}")
            return []
    
    def print_hl_after_ll_results(self, matches: List[Any]):
        """Print HL After LL scan results"""
        output_config = self.config.get('output', {}).get('console', {})
        
        if not matches:
            print("\nNo LL -> HH -> HL patterns found")
            return
        
        if output_config.get('show_summary', True):
            print(f"\nFound {len(matches)} symbols with LL -> HH -> HL patterns:")
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
                        print(f"   PASS: Monday Check: Current price (${match.last_price:.2f}) >= HL (${match.hl_price:.2f})")
                    else:
                        print(f"   FAIL: Monday Check: Current price (${match.last_price:.2f}) < HL (${match.hl_price:.2f}) - Pattern broken")
    
    def print_results(self, matches: List[Any]):
        """Print scan results based on scanner type"""
        if self.scanner_type == 'hl_after_ll':
            self.print_hl_after_ll_results(matches)
        else:
            print(f"\nResults for {self.scanner_type} scanner:")
            print(f"Found {len(matches)} matches")
    
    def save_hl_after_ll_results(self, matches: List[Any]):
        """Save HL After LL results to CSV"""
        if not matches:
            print("No matches to save")
            return
        
        output_config = self.config.get('output', {}).get('files', {})
        if not output_config.get('save_csv', True):
            return
        
        import pandas as pd
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/hl_after_ll_scan_{timestamp}.csv"
        
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
        
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
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
    
    def save_results(self, matches: List[Any]):
        """Save results based on scanner type"""
        if self.scanner_type == 'hl_after_ll':
            self.save_hl_after_ll_results(matches)
        else:
            print(f"CSV export not implemented for {self.scanner_type} scanner yet")
    
    def run(self, scanner_type: str, symbols: Optional[List[str]] = None, skip_auto_update: bool = False):
        """Run the complete scan with optional auto-update"""
        print(f"{scanner_type.upper()} Scanner")
        print("=" * 50)
        
        # Get symbols to scan
        if symbols is None:
            symbols = self.get_symbols()
        
        if not symbols:
            self.logger.error("No symbols to scan")
            return
        
        # Auto-update data if enabled and not skipped
        auto_update_config = self.config.get('auto_update', {})
        if not skip_auto_update and auto_update_config.get('enabled', True):
            print(f"\nAuto-Update Data Check")
            print("-" * 30)
            
            # Check data freshness
            freshness = self.check_data_freshness(symbols)
            
            if freshness['needs_update']:
                print(f"WARNING: Data needs updating:")
                if freshness['stale_symbols']:
                    print(f"   Stale data: {len(freshness['stale_symbols'])} symbols")
                if freshness['missing_symbols']:
                    print(f"   Missing data: {len(freshness['missing_symbols'])} symbols")
                
                print(f"\nUpdating data...")
                update_success = self.update_stale_data(
                    freshness['stale_symbols'], 
                    freshness['missing_symbols']
                )
                
                if update_success:
                    print(f"SUCCESS: Data update completed successfully")
                else:
                    print(f"WARNING: Data update had some failures, but continuing with scan...")
            else:
                print(f"SUCCESS: All data is fresh, no update needed")
        elif skip_auto_update:
            print(f"SKIP: Skipping auto-update (--skip-update specified)")
        else:
            print(f"SKIP: Auto-update disabled in configuration")
        
        # Run the scanner
        print(f"\nRunning {scanner_type.upper()} scanner...")
        matches = self.run_scanner(scanner_type, symbols)
        
        # Display results
        self.print_results(matches)
        
        # Save results
        if matches:
            self.save_results(matches)
        
        print(f"\n{scanner_type.upper()} scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Unified Scanner Runner")
    parser.add_argument('--config', type=str, default='scanner_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--scanner', type=str, required=True,
                       choices=['hl_after_ll', 'vcp', 'liquidity_sweep'],
                       help='Scanner type to run')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to scan (overrides config)')
    parser.add_argument('--provider', type=str, choices=['ALPACA', 'IB'], default='ALPACA',
                       help='Data provider (for future use)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    parser.add_argument('--skip-update', action='store_true',
                       help='Skip automatic data update, scan existing data only')
    parser.add_argument('--force-update', action='store_true',
                       help='Force data update even if data appears fresh')
    
    args = parser.parse_args()
    
    # Initialize scanner runner
    runner = ScannerRunner(args.config)
    
    # Override logging level if specified
    if args.log_level:
        runner.config['logging']['level'] = args.log_level
        runner.setup_logging()
    
    # Override symbols if provided
    if args.symbols:
        runner.config['universe'] = {'source': 'manual', 'symbols': args.symbols}
    
    # Override auto-update settings if specified
    if args.skip_update:
        runner.config['auto_update']['enabled'] = False
    elif args.force_update:
        runner.config['auto_update']['max_days_old'] = 0  # Force update even fresh data
    
    # Run scanner
    runner.run(args.scanner, args.symbols, skip_auto_update=args.skip_update)

if __name__ == "__main__":
    main()
