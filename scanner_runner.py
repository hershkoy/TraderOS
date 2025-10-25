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
from utils.squeeze_scanner import (
    scan_symbol_for_squeeze as squeeze_scan_symbol,
    load_from_timescaledb as squeeze_load,
)
from utils.ticker_universe import get_combined_universe
from utils.timescaledb_client import get_timescaledb_client

class ScannerRunner:
    """Unified scanner runner supporting multiple scanner types"""
    
    def __init__(self, config_path: str = "scanner_config.yaml", scanner_type: str = None):
        """Initialize scanner runner with configuration"""
        self.config = self.load_config(config_path)
        self.scanner_type = scanner_type
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
        
        # Add file logging if enabled
        output_config = self.config.get('output', {})
        files_config = output_config.get('files', {})
        
        if files_config.get('save_log', True):
            # Generate timestamped log filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"logs/scanner_runner_{timestamp}.log"
            
            # Use custom filename if specified in config
            if 'log_filename' in files_config:
                log_filename = files_config['log_filename'].format(
                    timestamp=timestamp,
                    scanner_type=self.scanner_type or 'unknown'
                )
            
            # Add file handler
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(format_str))
            handlers.append(file_handler)
            
            # Log the file location
            print(f"Logging to file: {log_filename}")
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=handlers,
            force=True  # Force reconfiguration
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
                                self.logger.info(f"  [SUCCESS] {symbol} - Data updated and saved successfully")
                            else:
                                failed_updates += 1
                                self.logger.warning(f"  [FAILED] {symbol} - Data fetched but failed to save to database")
                        else:
                            failed_updates += 1
                            self.logger.warning(f"  [FAILED] {symbol} - Could not connect to database")
                    else:
                        failed_updates += 1
                        self.logger.warning(f"  [FAILED] {symbol} - No data returned from fetch")
                        
                except Exception as e:
                    failed_updates += 1
                    self.logger.warning(f"  [FAILED] {symbol} - Error fetching/saving data: {e}")
                    
            except Exception as e:
                failed_updates += 1
                self.logger.warning(f"  [FAILED] {symbol} - Error: {e}")
        
        self.logger.info(f"Data update summary:")
        self.logger.info(f"  [SUCCESS] Successful: {successful_updates}")
        self.logger.info(f"  [FAILED] Failed: {failed_updates}")
        self.logger.info(f"  [STATS] Success Rate: {successful_updates/(successful_updates+failed_updates)*100:.1f}%")
        
        return successful_updates > 0
    
    def _log_pivot_signals(self, symbol: str, df: pd.DataFrame):
        """Log the last 5 pivot signals (LL, HL, HH, LH) for debugging"""
        try:
            # Import the scanner functions
            from utils.hl_after_ll_scanner import to_weekly, classify_pivots_weekly
            
            # Log data info
            try:
                self.logger.info(f"  {symbol}: DataFrame shape: {df.shape}")
                self.logger.info(f"  {symbol}: DataFrame index type: {type(df.index)}")
                self.logger.info(f"  {symbol}: DataFrame index sample: {df.index[:3].tolist()}")
                
                min_date = df.index.min()
                max_date = df.index.max()
                if pd.isna(min_date) or pd.isna(max_date):
                    self.logger.info(f"  {symbol}: Data has invalid timestamps (NaT values)")
                    self.logger.info(f"  {symbol}: Min date: {min_date}, Max date: {max_date}")
                else:
                    self.logger.info(f"  {symbol}: Data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({len(df)} bars)")
            except Exception as e:
                self.logger.info(f"  {symbol}: Error getting data range: {e}")
            
            # Convert to weekly data
            df_w = to_weekly(df)
            try:
                min_date_w = df_w.index.min()
                max_date_w = df_w.index.max()
                if pd.isna(min_date_w) or pd.isna(max_date_w):
                    self.logger.info(f"  {symbol}: Weekly data has invalid timestamps (NaT values)")
                else:
                    self.logger.info(f"  {symbol}: Weekly data: {min_date_w.strftime('%Y-%m-%d')} to {max_date_w.strftime('%Y-%m-%d')} ({len(df_w)} weeks)")
            except Exception as e:
                self.logger.info(f"  {symbol}: Error getting weekly data range: {e}")
            
            # Get pivot signals
            swings = classify_pivots_weekly(df_w)
            
            if not swings:
                self.logger.info(f"  {symbol}: No pivot signals detected")
                return
            
            # Get the last 5 signals
            last_signals = swings[-5:] if len(swings) >= 5 else swings
            
            self.logger.info(f"  {symbol}: Last {len(last_signals)} pivot signals:")
            for signal in last_signals:
                date_str = signal.ts.strftime('%Y-%m-%d')
                self.logger.info(f"    {date_str}: {signal.label} at ${signal.price:.2f}")
                
        except Exception as e:
            self.logger.info(f"  {symbol}: Error getting pivot signals: {e}")
            import traceback
            self.logger.info(f"  {symbol}: Traceback: {traceback.format_exc()}")
    
    def check_and_fetch_insufficient_data(self, symbols: List[str]) -> List[str]:
        """Check for insufficient data and fetch more historical data if needed"""
        scanner_config = self.config.get('scanner', {})
        data_config = scanner_config.get('data', {})
        auto_fetch_enabled = data_config.get('auto_fetch_enabled', True)
        auto_fetch_weeks = data_config.get('auto_fetch_weeks', 120)
        min_weeks = data_config.get('min_weeks', 52)
        timeframe = data_config.get('timeframe', '1d')
        
        if not auto_fetch_enabled:
            return symbols
        
        self.logger.info(f"Checking data sufficiency for {len(symbols)} symbols...")
        
        # Check each symbol for sufficient data
        symbols_needing_data = []
        client = get_timescaledb_client()
        
        try:
            if client.ensure_connection():
                for symbol in symbols:
                    # Get data count for this symbol
                    query = """
                    SELECT COUNT(*) as count, 
                           MIN(ts) as earliest, 
                           MAX(ts) as latest
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s
                    """
                    
                    result = client.execute_query(query, (symbol, timeframe))
                    if result and len(result) > 0:
                        count, earliest, latest = result[0]
                        
                        if count > 0:
                            # Calculate weeks of data
                            if earliest and latest:
                                weeks_of_data = (latest - earliest).days / 7
                                self.logger.info(f"  {symbol}: {count} bars, {weeks_of_data:.1f} weeks of data")
                                
                                if weeks_of_data < min_weeks:
                                    self.logger.info(f"  {symbol}: Insufficient data ({weeks_of_data:.1f} weeks < {min_weeks} weeks)")
                                    symbols_needing_data.append(symbol)
                                else:
                                    self.logger.info(f"  {symbol}: Sufficient data ({weeks_of_data:.1f} weeks >= {min_weeks} weeks)")
                            else:
                                self.logger.warning(f"  {symbol}: Could not determine data range")
                                symbols_needing_data.append(symbol)
                        else:
                            self.logger.info(f"  {symbol}: No data found")
                            symbols_needing_data.append(symbol)
                    else:
                        self.logger.warning(f"  {symbol}: Could not check data count")
                        symbols_needing_data.append(symbol)
        
        except Exception as e:
            self.logger.error(f"Error checking data sufficiency: {e}")
            return symbols
        
        finally:
            client.disconnect()
        
        # Fetch more data for symbols that need it
        if symbols_needing_data:
            self.logger.info(f"Fetching additional historical data for {len(symbols_needing_data)} symbols...")
            
            # Get auto-update config for provider
            auto_update_config = self.config.get('auto_update', {})
            provider = auto_update_config.get('provider', 'ALPACA')
            
            # Calculate bars needed (convert weeks to days, then to bars)
            if timeframe == "1d":
                bars_needed = auto_fetch_weeks * 7 + 50  # Extra buffer
            else:  # 1h
                bars_needed = auto_fetch_weeks * 7 * 24 + 200  # Extra buffer for hourly
            
            successful_fetches = 0
            
            for i, symbol in enumerate(symbols_needing_data, 1):
                try:
                    self.logger.info(f"  [{i}/{len(symbols_needing_data)}] Fetching {auto_fetch_weeks} weeks of data for {symbol}...")
                    
                    # Import the fetch function
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
                        if client.ensure_connection():
                            success = client.insert_market_data(df, symbol, provider, timeframe)
                            client.disconnect()
                            
                            if success:
                                successful_fetches += 1
                                self.logger.info(f"  [SUCCESS] {symbol} - Historical data fetched and saved successfully")
                            else:
                                self.logger.error(f"  [FAILED] {symbol} - Failed to save historical data")
                        else:
                            self.logger.error(f"  [FAILED] {symbol} - Could not connect to database")
                    else:
                        self.logger.error(f"  [FAILED] {symbol} - No data received from {provider}")
                        
                except Exception as e:
                    self.logger.error(f"  [FAILED] {symbol} - Error fetching historical data: {e}")
                    continue
            
            self.logger.info(f"Historical data fetch summary:")
            self.logger.info(f"  [SUCCESS] Successful: {successful_fetches}")
            self.logger.info(f"  [FAILED] Failed: {len(symbols_needing_data) - successful_fetches}")
            self.logger.info(f"  [STATS] Success Rate: {successful_fetches/len(symbols_needing_data)*100:.1f}%")
        
        return symbols

    def _check_and_fetch_insufficient_data_single(self, symbol: str) -> bool:
        """Check and fetch insufficient data for a single stock"""
        scanner_config = self.config.get('scanner', {})
        data_config = scanner_config.get('data', {})
        auto_fetch_enabled = data_config.get('auto_fetch_enabled', True)
        auto_fetch_weeks = data_config.get('auto_fetch_weeks', 120)
        min_weeks = data_config.get('min_weeks', 52)
        timeframe = data_config.get('timeframe', '1d')
        
        if not auto_fetch_enabled:
            return True
        
        client = get_timescaledb_client()
        
        try:
            if client.ensure_connection():
                # Get data count for this symbol
                query = """
                SELECT COUNT(*) as count, 
                       MIN(ts) as earliest, 
                       MAX(ts) as latest
                FROM market_data 
                WHERE symbol = %s AND timeframe = %s
                """
                
                result = client.execute_query(query, (symbol, timeframe))
                if result and len(result) > 0:
                    count, earliest, latest = result[0]
                    
                    if count > 0:
                        # Calculate weeks of data
                        if earliest and latest:
                            weeks_of_data = (latest - earliest).days / 7
                            self.logger.info(f"  {symbol}: {count} bars, {weeks_of_data:.1f} weeks of data")
                            
                            if weeks_of_data < min_weeks:
                                self.logger.info(f"  {symbol}: Insufficient data ({weeks_of_data:.1f} weeks < {min_weeks} weeks)")
                                
                                # Fetch more data for this symbol
                                self.logger.info(f"  {symbol}: Fetching {auto_fetch_weeks} weeks of historical data...")
                                
                                # Get auto-update config for provider
                                auto_update_config = self.config.get('auto_update', {})
                                provider = auto_update_config.get('provider', 'ALPACA')
                                
                                # Calculate bars needed
                                if timeframe == "1d":
                                    bars_needed = auto_fetch_weeks * 7 + 50  # Extra buffer
                                else:  # 1h
                                    bars_needed = auto_fetch_weeks * 7 * 24 + 200  # Extra buffer for hourly
                                
                                # Import the fetch function
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
                                    if client.ensure_connection():
                                        success = client.insert_market_data(df, symbol, provider, timeframe)
                                        client.disconnect()
                                        
                                        if success:
                                            self.logger.info(f"  {symbol}: [SUCCESS] Historical data fetched and saved successfully")
                                            return True
                                        else:
                                            self.logger.error(f"  {symbol}: [FAILED] Failed to save historical data")
                                            return False
                                    else:
                                        self.logger.error(f"  {symbol}: [FAILED] Could not connect to database")
                                        return False
                                else:
                                    self.logger.error(f"  {symbol}: [FAILED] No data received from {provider}")
                                    return False
                            else:
                                self.logger.info(f"  {symbol}: Sufficient data ({weeks_of_data:.1f} weeks >= {min_weeks} weeks)")
                                return True
                        else:
                            self.logger.warning(f"  {symbol}: Could not determine data range")
                            return False
                    else:
                        self.logger.info(f"  {symbol}: No data found")
                        return False
                else:
                    self.logger.warning(f"  {symbol}: Could not check data count")
                    return False
        
        except Exception as e:
            self.logger.error(f"  {symbol}: Error checking data sufficiency: {e}")
            return False
        
        finally:
            client.disconnect()
        
        return True

    def _check_data_completeness_single(self, symbol: str) -> dict:
        """Check data completeness for a single stock"""
        auto_update_config = self.config.get('auto_update', {})
        max_days_old = auto_update_config.get('max_days_old', 1)
        timeframe = auto_update_config.get('timeframe', '1d')
        
        client = get_timescaledb_client()
        
        try:
            if client.ensure_connection():
                # Get latest data for this symbol
                query = """
                SELECT MAX(ts) as latest_date, COUNT(*) as count
                FROM market_data 
                WHERE symbol = %s AND timeframe = %s
                """
                
                result = client.execute_query(query, (symbol, timeframe))
                if result and len(result) > 0:
                    latest_date, count = result[0]
                    
                    if count == 0:
                        return {
                            'needs_update': True,
                            'reason': 'No data found',
                            'latest_date': None,
                            'days_old': None
                        }
                    
                    # Calculate how old the data is
                    if latest_date:
                        from datetime import datetime, timezone
                        now = datetime.now(timezone.utc)
                        if latest_date.tzinfo is None:
                            latest_date = latest_date.replace(tzinfo=timezone.utc)
                        
                        days_old = (now - latest_date).days
                        
                        if days_old > max_days_old:
                            return {
                                'needs_update': True,
                                'reason': f'Data is {days_old} days old (max: {max_days_old})',
                                'latest_date': latest_date,
                                'days_old': days_old
                            }
                        else:
                            return {
                                'needs_update': False,
                                'reason': f'Data is fresh ({days_old} days old)',
                                'latest_date': latest_date,
                                'days_old': days_old
                            }
                    else:
                        return {
                            'needs_update': True,
                            'reason': 'Could not determine data age',
                            'latest_date': None,
                            'days_old': None
                        }
                else:
                    return {
                        'needs_update': True,
                        'reason': 'Could not query data',
                        'latest_date': None,
                        'days_old': None
                    }
        
        except Exception as e:
            self.logger.error(f"  {symbol}: Error checking data completeness: {e}")
            return {
                'needs_update': True,
                'reason': f'Error: {e}',
                'latest_date': None,
                'days_old': None
            }
        
        finally:
            client.disconnect()

    def _update_data_single(self, symbol: str) -> bool:
        """Update data for a single stock"""
        auto_update_config = self.config.get('auto_update', {})
        provider = auto_update_config.get('provider', 'ALPACA')
        timeframe = auto_update_config.get('timeframe', '1d')
        days_back = auto_update_config.get('days_back', 30)
        
        try:
            # Calculate bars needed
            if timeframe == "1d":
                bars_needed = days_back + 10  # Extra buffer
            else:  # 1h
                bars_needed = days_back * 7 + 50  # Extra buffer for hourly
            
            # Import the fetch function
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
                        self.logger.info(f"  {symbol}: [SUCCESS] Data updated and saved successfully")
                        return True
                    else:
                        self.logger.error(f"  {symbol}: [FAILED] Failed to save data")
                        return False
                else:
                    self.logger.error(f"  {symbol}: [FAILED] Could not connect to database")
                    return False
            else:
                self.logger.error(f"  {symbol}: [FAILED] No data received from {provider}")
                return False
                
        except Exception as e:
            self.logger.error(f"  {symbol}: [FAILED] Error updating data: {e}")
            return False

    def run_hl_after_ll_scanner(self, symbols: List[str]) -> List[Any]:
        """Run HL After LL scanner - process each stock individually"""
        scanner_config = self.config.get('scanner', {})
        data_config = scanner_config.get('data', {})
        timeframe = data_config.get('timeframe', '1d')
        
        self.logger.info(f"Running HL After LL scanner on {len(symbols)} symbols")
        self.logger.info("Processing each stock individually: check completeness -> update data -> check criteria")
        
        matches = []
        successful_scans = 0
        failed_scans = 0
        
        for i, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
                
                # Step 1: Check data completeness for this specific stock
                self.logger.info(f"  {symbol}: Checking data completeness...")
                data_status = self._check_data_completeness_single(symbol)
                
                # Step 2: Update data if needed
                if data_status['needs_update']:
                    self.logger.info(f"  {symbol}: Updating data...")
                    update_success = self._update_data_single(symbol)
                    if not update_success:
                        self.logger.warning(f"  {symbol}: Data update failed, skipping...")
                        failed_scans += 1
                        continue
                else:
                    self.logger.info(f"  {symbol}: Data is up to date")
                
                # Step 3: Check for insufficient historical data and fetch more if needed
                self.logger.info(f"  {symbol}: Checking data sufficiency...")
                self._check_and_fetch_insufficient_data_single(symbol)
                
                # Step 4: Load data for this stock
                self.logger.info(f"  {symbol}: Loading data from TimescaleDB...")
                ohlcv_data = load_from_timescaledb([symbol], timeframe)
                
                if not ohlcv_data or symbol not in ohlcv_data:
                    self.logger.warning(f"  {symbol}: No data available after update")
                    failed_scans += 1
                    continue
                
                df = ohlcv_data[symbol]
                
                # Step 5: Get debug info for this symbol
                self._log_pivot_signals(symbol, df)
                
                # Step 6: Scan for patterns
                self.logger.info(f"  {symbol}: Scanning for LL -> HH -> HL patterns...")
                match = scan_symbol_for_setup(symbol, df)
                
                if match:
                    matches.append(match)
                    self.logger.info(f"  {symbol}: [SUCCESS] Pattern found!")
                else:
                    self.logger.info(f"  {symbol}: No pattern detected")
                
                successful_scans += 1
                
            except Exception as e:
                self.logger.error(f"  {symbol}: Error processing: {e}")
                failed_scans += 1
                continue
        
        # Summary
        self.logger.info(f"Scanning complete:")
        self.logger.info(f"  [SUCCESS] Successful: {successful_scans}")
        self.logger.info(f"  [FAILED] Failed: {failed_scans}")
        self.logger.info(f"  [STATS] Success Rate: {successful_scans/len(symbols)*100:.1f}%")
        self.logger.info(f"  [RESULTS] Patterns Found: {len(matches)}")
        
        return matches
    
    def run_vcp_scanner(self, symbols: List[str]) -> List[Any]:
        """Run VCP scanner (placeholder for future implementation)"""
        self.logger.info("VCP scanner not implemented yet")
        return []
    
    def run_squeeze_scanner(self, symbols: List[str]) -> List[Any]:
        """Run Squeeze Zero-Cross-Up scanner - per-stock completeness + update like HL scanner."""
        scanner_cfg = self.config.get('scanner', {})
        data_cfg = scanner_cfg.get('data', {})
        timeframe = data_cfg.get('timeframe', '1d')
        lengthKC = scanner_cfg.get('lengthKC', 20)
        confirm_on_close = scanner_cfg.get('confirm_on_close', True)

        self.logger.info(f"Running SQUEEZE scanner on {len(symbols)} symbols")

        matches = []
        ok, fail = 0, 0
        for i, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"[{i}/{len(symbols)}] {symbol}...")

                # completeness
                status = self._check_data_completeness_single(symbol)
                if status['needs_update']:
                    self.logger.info(f"  {symbol}: Updating data...")
                    if not self._update_data_single(symbol):
                        self.logger.warning(f"  {symbol}: Update failed, skipping")
                        fail += 1
                        continue

                # ensure enough history
                self._check_and_fetch_insufficient_data_single(symbol)

                # load and scan
                ohlcv = squeeze_load([symbol], timeframe)
                if symbol not in ohlcv or ohlcv[symbol].empty:
                    self.logger.warning(f"  {symbol}: No data after update")
                    fail += 1
                    continue

                m = squeeze_scan_symbol(
                    symbol, ohlcv[symbol],
                    lengthKC=lengthKC, confirm_on_close=confirm_on_close,
                    timeframe_label=timeframe
                )
                if m:
                    matches.append(m)
                    self.logger.info(f"  {symbol}: [HIT] Squeeze zero-cross up on {m.cross_date.date()}")
                else:
                    self.logger.info(f"  {symbol}: No cross-up")
                ok += 1

            except Exception as e:
                self.logger.error(f"  {symbol}: Error: {e}")
                fail += 1
                continue

        self.logger.info(f"Scanning complete. Success {ok}, Failed {fail}, Hits {len(matches)}")
        return matches

    def run_liquidity_sweep_scanner(self, symbols: List[str]) -> List[Any]:
        """Run Liquidity Sweep scanner (placeholder for future implementation)"""
        self.logger.info("Liquidity Sweep scanner not implemented yet")
        return []
    
    def run_multi_scanner(self, scanner_types: List[str], symbols: List[str]) -> Dict[str, List[Any]]:
        """Run multiple scanners on each stock - more efficient approach"""
        self.logger.info(f"Running MULTI-SCANNER on {len(symbols)} symbols with scanners: {scanner_types}")
        self.logger.info("Processing each stock once: check data -> update if needed -> run all scanners")
        
        # Initialize results for each scanner
        all_results = {scanner_type: [] for scanner_type in scanner_types}
        
        successful_scans = 0
        failed_scans = 0
        
        for i, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
                
                # Step 1: Check data completeness for this specific stock
                self.logger.info(f"  {symbol}: Checking data completeness...")
                data_status = self._check_data_completeness_single(symbol)
                
                # Step 2: Update data if needed
                if data_status['needs_update']:
                    self.logger.info(f"  {symbol}: Updating data...")
                    update_success = self._update_data_single(symbol)
                    if not update_success:
                        self.logger.warning(f"  {symbol}: Data update failed, skipping...")
                        failed_scans += 1
                        continue
                else:
                    self.logger.info(f"  {symbol}: Data is up to date")
                
                # Step 3: Check for insufficient historical data and fetch more if needed
                self.logger.info(f"  {symbol}: Checking data sufficiency...")
                self._check_and_fetch_insufficient_data_single(symbol)
                
                # Step 4: Load data for this stock once
                scanner_config = self.config.get('scanner', {})
                data_config = scanner_config.get('data', {})
                timeframe = data_config.get('timeframe', '1d')
                
                self.logger.info(f"  {symbol}: Loading data from TimescaleDB...")
                ohlcv_data = load_from_timescaledb([symbol], timeframe)
                
                if not ohlcv_data or symbol not in ohlcv_data:
                    self.logger.warning(f"  {symbol}: No data available after update")
                    failed_scans += 1
                    continue
                
                df = ohlcv_data[symbol]
                
                # Step 5: Run all enabled scanners on this stock
                symbol_results = {}
                for scanner_type in scanner_types:
                    try:
                        self.logger.info(f"  {symbol}: Running {scanner_type.upper()} scanner...")
                        
                        if scanner_type == 'hl_after_ll':
                            # Get debug info for HL scanner
                            self._log_pivot_signals(symbol, df)
                            match = scan_symbol_for_setup(symbol, df)
                            
                        elif scanner_type == 'squeeze':
                            lengthKC = scanner_config.get('lengthKC', 20)
                            confirm_on_close = scanner_config.get('confirm_on_close', True)
                            match = squeeze_scan_symbol(
                                symbol, df, 
                                lengthKC=lengthKC, 
                                confirm_on_close=confirm_on_close,
                                timeframe_label=timeframe
                            )
                            
                        elif scanner_type == 'vcp':
                            match = None  # Placeholder
                            self.logger.info(f"  {symbol}: VCP scanner not implemented yet")
                            
                        elif scanner_type == 'liquidity_sweep':
                            match = None  # Placeholder
                            self.logger.info(f"  {symbol}: Liquidity Sweep scanner not implemented yet")
                            
                        else:
                            self.logger.warning(f"  {symbol}: Unknown scanner type: {scanner_type}")
                            match = None
                        
                        if match:
                            all_results[scanner_type].append(match)
                            symbol_results[scanner_type] = match
                            self.logger.info(f"  {symbol}: [HIT] {scanner_type.upper()} pattern found!")
                        else:
                            self.logger.info(f"  {symbol}: No {scanner_type.upper()} pattern detected")
                            
                    except Exception as e:
                        self.logger.error(f"  {symbol}: Error in {scanner_type} scanner: {e}")
                        continue
                
                # Log summary for this symbol
                hits = [st for st, result in symbol_results.items() if result is not None]
                if hits:
                    self.logger.info(f"  {symbol}: [SUMMARY] Found patterns: {', '.join(hits)}")
                else:
                    self.logger.info(f"  {symbol}: [SUMMARY] No patterns found")
                
                successful_scans += 1
                
            except Exception as e:
                self.logger.error(f"  {symbol}: Error processing: {e}")
                failed_scans += 1
                continue
        
        # Summary
        self.logger.info(f"Multi-scanner complete:")
        self.logger.info(f"  [SUCCESS] Successful: {successful_scans}")
        self.logger.info(f"  [FAILED] Failed: {failed_scans}")
        self.logger.info(f"  [STATS] Success Rate: {successful_scans/len(symbols)*100:.1f}%")
        
        for scanner_type in scanner_types:
            count = len(all_results[scanner_type])
            self.logger.info(f"  [RESULTS] {scanner_type.upper()} patterns found: {count}")
        
        return all_results

    def run_scanner(self, scanner_type: str, symbols: List[str]) -> List[Any]:
        """Run specified scanner type (legacy single-scanner method)"""
        self.scanner_type = scanner_type
        
        if scanner_type == 'hl_after_ll':
            return self.run_hl_after_ll_scanner(symbols)
        elif scanner_type == 'squeeze':
            return self.run_squeeze_scanner(symbols)
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
    
    def print_squeeze_results(self, matches: List[Any]):
        """Print Squeeze scan results"""
        if not matches:
            print("\nNo Squeeze zero-cross up signals found")
            return
        print(f"\nFound {len(matches)} symbols with Squeeze zero-cross up (val: <=0 → >0):")
        print("=" * 80)
        for i, m in enumerate(matches, 1):
            print(f"\n{i}. {m.symbol}")
            print(f"   Cross Week : {m.cross_date.strftime('%Y-%m-%d')}")
            print(f"   Prev val   : {m.prev_value:.4f}")
            print(f"   Curr val   : {m.cross_value:.4f}")
            print(f"   Last Price : {m.last_price:.2f}")

    def print_multi_scanner_results(self, all_results: Dict[str, List[Any]]):
        """Print results for multiple scanners"""
        print("\n" + "=" * 80)
        print("MULTI-SCANNER RESULTS")
        print("=" * 80)
        
        total_matches = 0
        for scanner_type, matches in all_results.items():
            total_matches += len(matches)
            print(f"\n{scanner_type.upper()} Scanner: {len(matches)} matches")
            print("-" * 40)
            
            if scanner_type == 'hl_after_ll':
                self.print_hl_after_ll_results(matches)
            elif scanner_type == 'squeeze':
                self.print_squeeze_results(matches)
            else:
                print(f"Found {len(matches)} matches")
        
        print(f"\n" + "=" * 80)
        print(f"TOTAL MATCHES ACROSS ALL SCANNERS: {total_matches}")
        print("=" * 80)

    def print_results(self, matches: List[Any]):
        """Print scan results based on scanner type"""
        if self.scanner_type == 'hl_after_ll':
            self.print_hl_after_ll_results(matches)
        elif self.scanner_type == 'squeeze':
            self.print_squeeze_results(matches)
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
    
    def save_squeeze_results(self, matches: List[Any]):
        """Save Squeeze results to CSV"""
        if not matches:
            print("No matches to save")
            return
        files_cfg = self.config.get('output', {}).get('files', {})
        if not files_cfg.get('save_csv', True):
            return
        from pathlib import Path
        import pandas as pd
        Path("reports").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"reports/squeeze_zero_cross_{ts}.csv"
        rows = [{
            "symbol": m.symbol,
            "cross_week": m.cross_date.strftime("%Y-%m-%d"),
            "prev_val": m.prev_value,
            "curr_val": m.cross_value,
            "last_price": m.last_price,
            "timeframe": m.timeframe,
        } for m in matches]
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(f"\nResults saved to: {fn}")

    def save_multi_scanner_results(self, all_results: Dict[str, List[Any]]):
        """Save results for multiple scanners to separate CSV files"""
        files_cfg = self.config.get('output', {}).get('files', {})
        if not files_cfg.get('save_csv', True):
            return
        
        from pathlib import Path
        import pandas as pd
        
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        for scanner_type, matches in all_results.items():
            if not matches:
                continue
                
            if scanner_type == 'hl_after_ll':
                filename = f"reports/hl_after_ll_scan_{timestamp}.csv"
                self.save_hl_after_ll_results(matches)
                saved_files.append(filename)
                
            elif scanner_type == 'squeeze':
                filename = f"reports/squeeze_zero_cross_{timestamp}.csv"
                self.save_squeeze_results(matches)
                saved_files.append(filename)
                
            else:
                print(f"CSV export not implemented for {scanner_type} scanner yet")
        
        if saved_files:
            print(f"\nMulti-scanner results saved to:")
            for file in saved_files:
                print(f"  - {file}")

    def save_results(self, matches: List[Any]):
        """Save results based on scanner type"""
        if self.scanner_type == 'hl_after_ll':
            self.save_hl_after_ll_results(matches)
        elif self.scanner_type == 'squeeze':
            self.save_squeeze_results(matches)
        else:
            print(f"CSV export not implemented for {self.scanner_type} scanner yet")
    
    def run_multi(self, scanner_types: List[str], symbols: Optional[List[str]] = None, skip_auto_update: bool = False):
        """Run multiple scanners on each stock - efficient approach"""
        print(f"MULTI-SCANNER: {', '.join(scanner_types).upper()}")
        print("=" * 60)
        
        # Get symbols to scan
        if symbols is None:
            symbols = self.get_symbols()
        
        if not symbols:
            self.logger.error("No symbols to scan")
            return
        
        # Show auto-update status
        auto_update_config = self.config.get('auto_update', {})
        if skip_auto_update:
            print(f"SKIP: Skipping auto-update (--skip-update specified)")
        elif not auto_update_config.get('enabled', True):
            print(f"SKIP: Auto-update disabled in configuration")
        else:
            print(f"ENABLED: Per-stock auto-update will check and update data for each symbol individually")
        
        # Run the multi-scanner
        print(f"\nRunning MULTI-SCANNER with: {', '.join(scanner_types)}...")
        all_results = self.run_multi_scanner(scanner_types, symbols)
        
        # Display results
        self.print_multi_scanner_results(all_results)
        
        # Save results
        self.save_multi_scanner_results(all_results)
        
        print(f"\nMULTI-SCANNER completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def run(self, scanner_type: str, symbols: Optional[List[str]] = None, skip_auto_update: bool = False):
        """Run the complete scan with per-stock auto-update"""
        print(f"{scanner_type.upper()} Scanner")
        print("=" * 50)
        
        # Get symbols to scan
        if symbols is None:
            symbols = self.get_symbols()
        
        if not symbols:
            self.logger.error("No symbols to scan")
            return
        
        # Show auto-update status
        auto_update_config = self.config.get('auto_update', {})
        if skip_auto_update:
            print(f"SKIP: Skipping auto-update (--skip-update specified)")
        elif not auto_update_config.get('enabled', True):
            print(f"SKIP: Auto-update disabled in configuration")
        else:
            print(f"ENABLED: Per-stock auto-update will check and update data for each symbol individually")
        
        # Run the scanner (which now handles per-stock updates internally)
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
    parser.add_argument('--scanner', type=str, nargs='+',
                       choices=['hl_after_ll', 'squeeze', 'vcp', 'liquidity_sweep'],
                       help='Scanner type(s) to run. Use multiple for multi-scanner mode.')
    parser.add_argument('--multi', action='store_true',
                       help='Run multiple scanners on each stock (efficient mode)')
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
    
    # Validate arguments
    if not args.scanner:
        parser.error("--scanner is required")
    
    # Initialize scanner runner
    runner = ScannerRunner(args.config, args.scanner[0] if len(args.scanner) == 1 else 'multi')
    
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
    
    # Run scanner(s)
    if len(args.scanner) == 1 and not args.multi:
        # Single scanner mode
        runner.run(args.scanner[0], args.symbols, skip_auto_update=args.skip_update)
    else:
        # Multi-scanner mode
        runner.run_multi(args.scanner, args.symbols, skip_auto_update=args.skip_update)

if __name__ == "__main__":
    main()
