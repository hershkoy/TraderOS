#!/usr/bin/env python3
"""
Enhanced Zero-Cost Collar Screener for Interactive Brokers

This enhanced screener reads configuration from YAML and provides additional features:
- Configurable parameters via YAML file
- Better error handling and logging
- CSV export with detailed metrics
- Integration with existing backtrader project structure
- CLI parameter support for log levels
- Detailed debug logging for opportunity analysis

Author: AI Assistant
Date: 2025
"""

import sys
import os
import math
import yaml
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ib_insync import *
except ImportError:
    print("Error: ib_insync not found. Please install with: pip install ib_insync")
    sys.exit(1)

def qualify_with_timeout(ib: IB, contract: Contract, timeout: float = 8.0):
    """Qualify contract with real timeout that cancels the request"""
    try:
        # Run the async qualification inside ib_insync's loop with timeout
        qualified = ib.run(asyncio.wait_for(ib.qualifyContractsAsync(contract), timeout=timeout))
        return qualified[0] if qualified else None
    except asyncio.TimeoutError:
        return None  # signal: no qualification in time
    except Exception as e:
        return None  # other errors also return None

def qualify_multiple_with_timeout(ib: IB, contracts: List[Contract], timeout: float = 10.0):
    """Qualify multiple contracts together with timeout"""
    try:
        # Run the async qualification inside ib_insync's loop with timeout
        qualified = ib.run(asyncio.wait_for(ib.qualifyContractsAsync(*contracts), timeout=timeout))
        return qualified if qualified else []
    except asyncio.TimeoutError:
        return []  # signal: no qualification in time
    except Exception as e:
        return []  # other errors also return empty list

def qualify_option_variants(ib, symbol, expiry_ib, K, right, secdef_list, timeout=8.0):
    """Build & qualify options with smart fallbacks (don't force tradingClass)"""
    variants = []
    # build candidate exchanges in good order (SMART first)
    seen = set()
    for pp in secdef_list:
        ex = getattr(pp, 'exchange', 'SMART') or 'SMART'
        if ex in seen: 
            continue
        seen.add(ex)
        tclass = getattr(pp, 'tradingClass', None)

        # no tradingClass first
        variants.append(Option(symbol, expiry_ib, K, right, ex, multiplier='100'))
        # then with tradingClass if present
        if tclass:
            variants.append(Option(symbol, expiry_ib, K, right, ex, multiplier='100', tradingClass=tclass))

    # also add a pure SMART/no-tclass fallback just in case
    variants.insert(0, Option(symbol, expiry_ib, K, right, 'SMART', multiplier='100'))

    for opt in variants:
        q = qualify_with_timeout(ib, opt, timeout=timeout)
        if q:
            return q
    return None

def resolve_option_smart(ib, symbol, expiry_ib, K, right, timeout=10.0):
    """
    Resolve an option by asking IB for ContractDetails on SMART only,
    then qualify the exact returned contract. Avoids trying many venues.
    """
    # Start with the minimal SMART contract
    base = Option(symbol, expiry_ib, K, right, 'SMART', multiplier='100')
    try:
        # Ask IB to resolve it
        cds = ib.run(asyncio.wait_for(ib.reqContractDetailsAsync(base), timeout=timeout))
    except Exception:
        cds = []
    if not cds:
        return None

    # Take the first concrete contract IB returns (has conId + exchange)
    concrete = cds[0].contract
    # Now qualify that exact contract (fast)
    qc = qualify_with_timeout(ib, concrete, timeout=min(8.0, timeout))
    return qc

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Zero-Cost Collar Screener')
    parser.add_argument('--log-level', '-l', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set logging level (default: INFO)')
    parser.add_argument('--config', '-c',
                       default='collar_screener_config.yaml',
                       help='Configuration file path (default: collar_screener_config.yaml)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging (same as --log-level DEBUG)')
    parser.add_argument('--log-file', '-f',
                       help='Log file path (overrides config file setting)')
    parser.add_argument('--no-log-file',
                       action='store_true',
                       help='Disable logging to file (console only)')
    parser.add_argument('--no-real-time-updates',
                       action='store_true',
                       help='Disable real-time CSV updates (update only at end)')
    parser.add_argument('--progress-frequency',
                       type=int,
                       default=5,
                       help='Print progress summary every N symbols (default: 5)')
    return parser.parse_args()

class CollarScreenerConfig:
    """Configuration class for the collar screener"""
    
    def __init__(self, config_file: str = "collar_screener_config.yaml", log_level: str = "INFO"):
        self.config_file = config_file
        self.log_level = log_level
        self.load_config()
        self.setup_logging()
        # Load ticker universe from database
        self.load_ticker_universe()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
            
            # IB Connection
            self.IB_HOST = config['ib_connection']['host']
            self.IB_PORT = config['ib_connection']['port']
            self.IB_CLIENT_ID = config['ib_connection']['client_id']
            self.IB_READONLY = config['ib_connection']['readonly']
            
            # Capital and Budget
            self.CAPITAL_BUDGET = config['capital']['budget']
            self.MIN_CAPITAL_USED = config['capital']['min_position_size']
            
            # Option Parameters
            self.MIN_DTE = config['options']['min_dte']
            self.MAX_DTE = config['options']['max_dte']
            self.PUT_DELTA_RANGE = tuple(config['options']['put_delta_range'])
            self.CALL_DELTA_RANGE = tuple(config['options']['call_delta_range'])
            
            # Risk Tolerance
            self.FLOOR_TOLERANCE = config['risk']['floor_tolerance']
            self.MAX_SPREAD_PCT = config['risk']['max_spread_percent']
            
            # Universe (will be overridden by database)
            self.UNIVERSE = config['universe']
            
            # Output Settings
            self.MAX_RESULTS = config['output']['max_results']
            self.SAVE_TO_CSV = config['output']['save_to_csv']
            self.CSV_FILENAME = config['output']['csv_filename']
            self.LOG_TO_FILE = config['output'].get('log_to_file', True)
            self.LOG_FILENAME = config['output'].get('log_filename', 'logs/zero_cost_collar_screener.log')
            self.PROGRESS_UPDATE_FREQUENCY = config['output'].get('progress_update_frequency', 5)
            self.REAL_TIME_UPDATES = config['output'].get('real_time_updates', True)
            # Override config log level with CLI parameter
            self.LOG_LEVEL = self.log_level
            
            # Advanced Settings
            self.MAX_STRIKES_PER_EXPIRY = config['advanced']['max_strikes_per_expiry']
            self.REQUIRE_MODEL_GREEKS = config['advanced']['require_model_greeks']
            self.MIN_VOLUME = config['advanced']['min_volume']
            self.REQUEST_TIMEOUT = config['advanced']['request_timeout']
            self.SLEEP_BETWEEN_REQUESTS = config['advanced']['sleep_between_requests']
            self.HAS_OPRA = config['advanced'].get('has_opra', False)  # OPRA subscription for Greeks
            
            # Black-Scholes Parameters
            self.RISK_FREE_RATE = config['options'].get('risk_free_rate', 0.045)
            self.DIVIDEND_YIELD_DEFAULT = config['options'].get('dividend_yield_default', 0.0)
            self.ZERO_COST_TOLERANCE = config['options'].get('zero_cost_tolerance', -25.0)
            self.DELAYED_SNAPSHOT_PATIENCE = config['options'].get('delayed_snapshot_patience_s', 3.0)
            
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found. Using default configuration.")
            self._set_defaults()
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            self._set_defaults()
    
    def load_ticker_universe(self):
        """Load ticker universe from database, fallback to config file"""
        try:
            # Try different import paths
            try:
                from utils.ticker_universe import TickerUniverseManager
            except ImportError:
                try:
                    from ticker_universe import TickerUniverseManager
                except ImportError:
                    # Add parent directory to path
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from utils.ticker_universe import TickerUniverseManager
            
            manager = TickerUniverseManager()
            db_universe = manager.get_combined_universe()
            
            if db_universe and len(db_universe) > 0:
                self.UNIVERSE = db_universe
                print(f"Loaded {len(self.UNIVERSE)} symbols from database")
            else:
                print(f"Database returned empty universe, using config file with {len(self.UNIVERSE)} symbols")
                
        except Exception as e:
            print(f"Failed to load ticker universe from database: {e}")
            print(f"Using config file universe with {len(self.UNIVERSE)} symbols")
    
    def _set_defaults(self):
        """Set default configuration values"""
        self.IB_HOST = '127.0.0.1'
        self.IB_PORT = 7497
        self.IB_CLIENT_ID = 19
        self.IB_READONLY = True
        
        self.CAPITAL_BUDGET = 30_000
        self.MIN_CAPITAL_USED = 5_000
        
        self.MIN_DTE = 30
        self.MAX_DTE = 60
        self.PUT_DELTA_RANGE = (-0.35, -0.20)
        self.CALL_DELTA_RANGE = (0.20, 0.35)
        
        self.FLOOR_TOLERANCE = -25.0
        self.MAX_SPREAD_PCT = 0.15
        
        self.UNIVERSE = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'SPY', 'QQQ']
        
        self.MAX_RESULTS = 25
        self.SAVE_TO_CSV = True
        self.CSV_FILENAME = 'reports/collar_opportunities.csv'
        self.LOG_TO_FILE = True
        self.LOG_FILENAME = 'logs/zero_cost_collar_screener.log'
        self.PROGRESS_UPDATE_FREQUENCY = 5
        self.REAL_TIME_UPDATES = True
        self.LOG_LEVEL = self.log_level
        
        self.MAX_STRIKES_PER_EXPIRY = 16
        self.REQUIRE_MODEL_GREEKS = True
        self.MIN_VOLUME = 0
        self.REQUEST_TIMEOUT = 5.0
        self.SLEEP_BETWEEN_REQUESTS = 0.25
        self.HAS_OPRA = False
        
        # Black-Scholes Parameters
        self.RISK_FREE_RATE = 0.045
        self.DIVIDEND_YIELD_DEFAULT = 0.0
        self.ZERO_COST_TOLERANCE = -25.0
        self.DELAYED_SNAPSHOT_PATIENCE = 3.0
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Convert string log level to logging constant
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        
        # Create handlers list
        handlers = [logging.StreamHandler()]  # Always include console output
        
        # Add file handler if logging to file is enabled
        if self.LOG_TO_FILE:
            # Ensure the log file directory exists
            log_dir = os.path.dirname(self.LOG_FILENAME)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # Create file handler with timestamp in filename if it doesn't already have one
            log_filename = self.LOG_FILENAME
            if not any(char in log_filename for char in ['%', '{', '}']):
                # Add timestamp to filename if it doesn't have placeholders
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                name, ext = os.path.splitext(log_filename)
                log_filename = f"{name}_{timestamp}{ext}"
            
            handlers.append(logging.FileHandler(log_filename, mode='w'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        # Log the logging configuration
        if self.LOG_TO_FILE:
            print(f"Logging to file: {log_filename}")
        else:
            print("Logging to console only")
        
        # Filter out account data logging from ib_insync
        logging.getLogger('ib_insync.client').setLevel(logging.WARNING)
        logging.getLogger('ib_insync.wrapper').setLevel(logging.WARNING)
        
        # Filter out noisy IB subscription errors (10089, 10091)
        class IbErrorFilter(logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                return not ('Error 10089' in msg or 'Error 10091' in msg)
        
        wr = logging.getLogger('ib_insync.wrapper')
        wr.addFilter(IbErrorFilter())
        
        self.logger = logging.getLogger(__name__)

class CollarScreener:
    """Main collar screener class"""
    
    def __init__(self, config: CollarScreenerConfig):
        self.config = config
        self.logger = config.logger
        self.ib = None
    
    def connect_ib(self) -> bool:
        """Connect to Interactive Brokers using the same connection management as update_universe_data.py"""
        try:
            from fetch_data import get_ib_connection
            self.ib = get_ib_connection()
            if self.ib.isConnected():
                # Use delayed-frozen quotes for more consistent option snapshots
                self.ib.reqMarketDataType(4)  # 4 = DELAYED_FROZEN
                self.logger.info("Connected to Interactive Brokers using shared connection with delayed-frozen quotes")
                return True
            else:
                self.logger.error("Failed to connect to IB using shared connection")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to IB: {e}")
            print("Error: Could not connect to Interactive Brokers. Make sure TWS/IBG is running.")
            return False
    
    def disconnect_ib(self):
        """Disconnect from Interactive Brokers using shared cleanup"""
        try:
            from fetch_data import cleanup_ib_connection
            cleanup_ib_connection()
            self.logger.info("Disconnected from Interactive Brokers using shared cleanup")
        except Exception as e:
            self.logger.warning(f"Error during IB cleanup: {e}")
    
    def mid(self, bid: float, ask: float) -> Optional[float]:
        """Calculate mid price from bid/ask"""
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None
        return (bid + ask) / 2.0
    
    def in_range(self, x: float, lo: float, hi: float) -> bool:
        """Check if value is within range"""
        return x is not None and lo <= x <= hi
    
    def dte(self, expiry_str: str) -> int:
        """Calculate days to expiry"""
        try:
            if '-' not in expiry_str:
                d = datetime.strptime(expiry_str, '%Y%m%d')
            else:
                d = datetime.strptime(expiry_str, '%Y-%m-%d')
            return (d.date() - datetime.now().date()).days
        except ValueError as e:
            self.logger.warning(f"Invalid expiry format: {expiry_str}, error: {e}")
            return 0
    
    def option_mid_price(self, opt: Contract) -> Tuple[Optional[float], Optional[object]]:
        """
        Get option mid price using delayed snapshots.
        Be patient: delayed OPRA can take >1s to populate.
        Returns: (mid_price, ticker_object) - ticker can be reused for spread calculations
        """
        try:
            max_tries = 6          # ~ up to ~3–5 seconds total
            base_wait = getattr(self.config, 'DELAYED_SNAPSHOT_PATIENCE', 3.0) / 6.0  # distribute patience over tries
            # Request a snapshot (quotes only)
            ticker = self.ib.reqMktData(opt, '', True, False)

            for i in range(max_tries):
                # wait progressively longer
                self.ib.sleep(base_wait + 0.3 * i)  # progressive wait
                bid = getattr(ticker, 'bid', None)
                ask = getattr(ticker, 'ask', None)
                last = getattr(ticker, 'last', None)
                close = getattr(ticker, 'close', None)

                # Prefer bid/ask mid
                if bid and ask and bid > 0 and ask > 0:
                    m = (bid + ask) / 2.0
                    if m > 0:
                        return m, ticker

                # fallback to last then close
                if last and last > 0:
                    return last, ticker
                if close and close > 0:
                    return close, ticker

            # If nothing populated after patience window:
            self.logger.debug(f"No price data for {opt.symbol} {opt.strike} {opt.right}")
            return None, ticker

        except Exception as e:
            self.logger.debug(f"Error getting option price for {opt.symbol}: {e}")
            return None, None
    
    def stock_mid_price(self, stk: Contract) -> Optional[float]:
        """Get stock mid price with improved handling for delayed quotes"""
        max_retries = 5  # More retries for delayed quotes
        wait_time = 2.0  # Wait longer for delayed market data
        
        for attempt in range(max_retries):
            try:
                ticker = self.ib.reqMktData(stk, '', True, False)
                self.ib.sleep(wait_time)  # Wait longer for delayed market data
                
                # Try bid/ask first
                if ticker.bid is not None and ticker.ask is not None and ticker.bid > 0 and ticker.ask > 0:
                    mid_price = self.mid(ticker.bid, ticker.ask)
                    if mid_price and mid_price > 0 and not math.isnan(mid_price):
                        return mid_price
                
                # Fallback to last price
                if ticker.last is not None and ticker.last > 0 and not math.isnan(ticker.last):
                    return ticker.last
                
                # Fallback to close price
                if ticker.close is not None and ticker.close > 0 and not math.isnan(ticker.close):
                    return ticker.close
                
                # Try to get a reasonable price from available data
                prices = [p for p in [ticker.bid, ticker.ask, ticker.last, ticker.close] 
                         if p is not None and p > 0 and not math.isnan(p)]
                
                if prices:
                    return sum(prices) / len(prices)
                
                # For delayed quotes, try to get historical data as fallback
                if attempt >= 2:  # Only try historical data after a few attempts
                    try:
                        bars = self.ib.reqHistoricalData(
                            stk, 
                            '', 
                            f'{max_retries - attempt + 1} D', 
                            '1 day', 
                            'TRADES', 
                            useRTH=True,
                            formatDate=1
                        )
                        if bars and len(bars) > 0:
                            latest_bar = bars[-1]
                            if latest_bar.close and latest_bar.close > 0:
                                self.logger.debug(f"Using historical close price for {stk.symbol}: ${latest_bar.close:.2f}")
                                return latest_bar.close
                    except Exception as e:
                        self.logger.debug(f"Historical data fallback failed for {stk.symbol}: {e}")
                
                if attempt < max_retries - 1:
                    self.logger.debug(f"Attempt {attempt + 1} failed for {stk.symbol}, retrying... (bid={ticker.bid}, ask={ticker.ask}, last={ticker.last}, close={ticker.close})")
                    self.ib.sleep(1.0)  # Wait longer before retry for delayed quotes
                    continue
                
                self.logger.warning(f"No valid price data for stock {stk.symbol} after {max_retries} attempts (bid={ticker.bid}, ask={ticker.ask}, last={ticker.last}, close={ticker.close})")
                return None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.debug(f"Error getting stock price for {stk.symbol} (attempt {attempt + 1}): {e}, retrying...")
                    self.ib.sleep(1.0)
                    continue
                else:
                    self.logger.error(f"Error getting stock price for {stk.symbol} after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    def option_model_delta(self, opt: Contract) -> Optional[float]:
        """
        Only try Greeks if OPRA is available.
        - If you *do* have OPRA: request 106 with snapshot=False.
        - If you *don't*: return None immediately.
        """
        try:
            if not getattr(self.config, 'HAS_OPRA', False):
                return None  # no subscription → no Greeks

            # OPRA present: 106 must be streaming (snapshot=False)
            ticker = self.ib.reqMktData(opt, '106', False, False)
            self.ib.sleep(0.5)
            
            if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
                return ticker.modelGreeks.delta
            else:
                self.logger.debug(f"No delta data for {opt.symbol} {opt.strike} {opt.right} (likely no OPRA subscription)")
                return None
        except Exception as e:
            # Expected for delayed quotes without OPRA subscription
            self.logger.debug(f"Delta not available for {opt.symbol} {opt.strike} {opt.right}: {e}")
            return None
    
    def pick_by_moneyness(self, strikes: List[Tuple], stock_price: float, 
                          target_moneyness_range: Tuple[float, float], 
                          option_type: str = 'put') -> List[Tuple]:
        """Pick options by moneyness (K/spot) when delta is not available - ADAPTIVE VERSION"""
        candidates = []
        for contract, delta, mid_price, strike in strikes:
            if mid_price is None:
                continue
                
            # Calculate moneyness (K/spot)
            moneyness = strike / stock_price
            
            if option_type == 'put':
                # For puts: target 0.65-0.95 moneyness (15-35% OTM)
                if self.in_range(moneyness, target_moneyness_range[0], target_moneyness_range[1]):
                    candidates.append((contract, delta, mid_price, strike, moneyness))
            else:  # call
                # For calls: target 1.05-1.35 moneyness (15-35% OTM)
                if self.in_range(moneyness, target_moneyness_range[0], target_moneyness_range[1]):
                    candidates.append((contract, delta, mid_price, strike, moneyness))
        
        # Sort by distance from center of target moneyness range
        center = sum(target_moneyness_range) / 2
        candidates.sort(key=lambda x: abs(x[4] - center))  # x[4] is moneyness
        
        return [(c[0], c[1], c[2], c[3]) for c in candidates[:4]]  # Remove moneyness from result

    def pick_by_moneyness_adaptive(self, strikes: List[Tuple], stock_price: float, 
                                  option_type: str = 'put', min_candidates: int = 2) -> List[Tuple]:
        """
        ADAPTIVE moneyness-based selection with tiered widening to ensure we get candidates.
        Returns at least min_candidates if available, with progressive widening of moneyness bands.
        """
        candidates = []
        
        # Define tiered moneyness ranges for adaptive selection
        if option_type == 'put':
            # Tier 1: Preferred OTM range (15-35% OTM)
            tiers = [
                (0.65, 0.95),   # Tier 1: 15-35% OTM
                (0.80, 1.02),   # Tier 2: Allow ATM/near-ITM
                (0.75, 1.10),   # Tier 3: Wider range
                (0.70, 1.15),   # Tier 4: Very wide
            ]
        else:  # call
            # Tier 1: Preferred OTM range (15-35% OTM)
            tiers = [
                (1.05, 1.35),   # Tier 1: 15-35% OTM
                (0.98, 1.10),   # Tier 2: Allow ATM/near-ITM
                (0.95, 1.20),   # Tier 3: Wider range
                (0.90, 1.25),   # Tier 4: Very wide
            ]
        
        # Try each tier until we have enough candidates
        for tier_idx, (min_moneyness, max_moneyness) in enumerate(tiers):
            tier_candidates = []
            
            for contract, delta, mid_price, strike in strikes:
                if mid_price is None:
                    continue
                    
                moneyness = strike / stock_price
                if self.in_range(moneyness, min_moneyness, max_moneyness):
                    tier_candidates.append((contract, delta, mid_price, strike, moneyness))
            
            # Sort by distance from center of this tier's range
            center = (min_moneyness + max_moneyness) / 2
            tier_candidates.sort(key=lambda x: abs(x[4] - center))
            
            # Add candidates from this tier
            for candidate in tier_candidates:
                if len(candidates) >= min_candidates * 2:  # Get more than minimum to have choices
                    break
                candidates.append(candidate)
            
            # If we have enough candidates, stop
            if len(candidates) >= min_candidates:
                break
        
        # If we still don't have enough, take the nearest strikes to spot
        if len(candidates) < min_candidates:
            self.logger.debug(f"Adaptive moneyness selection for {option_type}s: only {len(candidates)} candidates, taking nearest to spot")
            
            # Get all strikes with valid mid prices
            all_valid = [(contract, delta, mid_price, strike, abs(strike - stock_price)) 
                        for contract, delta, mid_price, strike in strikes 
                        if mid_price is not None]
            
            # Sort by distance from spot
            all_valid.sort(key=lambda x: x[4])  # x[4] is distance from spot
            
            # Add the nearest ones
            for candidate in all_valid:
                if len(candidates) >= min_candidates * 2:
                    break
                # Convert back to (contract, delta, mid_price, strike, moneyness) format
                contract, delta, mid_price, strike, _ = candidate
                moneyness = strike / stock_price
                candidates.append((contract, delta, mid_price, strike, moneyness))
        
        # Remove moneyness from result and return
        return [(c[0], c[1], c[2], c[3]) for c in candidates[:min_candidates * 2]]
    
    def pick_by_delta(self, strikes: List[Tuple], target_range: Tuple[float, float], 
                      option_type: str = 'put', stock_price: float = None) -> List[Tuple]:
        """Pick options closest to target delta range, with adaptive fallback to moneyness"""
        # First try to pick by delta if available
        delta_candidates = []
        no_delta_strikes = []
        
        for contract, delta, mid_price, strike in strikes:
            if mid_price is None:
                continue
                
            if delta is not None:
                if option_type == 'put' and self.in_range(delta, target_range[0], target_range[1]):
                    delta_candidates.append((contract, delta, mid_price, strike))
                elif option_type == 'call' and self.in_range(delta, target_range[0], target_range[1]):
                    delta_candidates.append((contract, delta, mid_price, strike))
            else:
                no_delta_strikes.append((contract, delta, mid_price, strike))
        
        # If we have delta-based candidates, return them
        if delta_candidates:
            # Sort by distance from center of target range
            center = sum(target_range) / 2
            delta_candidates.sort(key=lambda x: abs(x[1] - center))
            return delta_candidates[:4]
        
        # Fallback to adaptive moneyness-based selection if no delta candidates
        if no_delta_strikes and stock_price:
            self.logger.debug(f"Falling back to adaptive moneyness-based selection for {option_type}s (no delta data)")
            return self.pick_by_moneyness_adaptive(no_delta_strikes, stock_price, option_type, min_candidates=2)
        
        return []
    
    def calculate_spread_percentage(self, bid: float, ask: float) -> float:
        """Calculate bid-ask spread as percentage of mid price"""
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return float('inf')
        mid_price = (bid + ask) / 2
        return (ask - bid) / mid_price
    
    def safe_round(self, x, n=3):
        """Safely round a value, handling None values"""
        return round(x, n) if isinstance(x, (int, float)) else None
    
    def scan_ticker(self, symbol: str) -> List[Dict]:
        """Scan a single ticker for collar opportunities with timeout protection"""
        import threading
        import time
        
        # Timeout mechanism using threading
        timeout_occurred = False
        
        def timeout_handler():
            nonlocal timeout_occurred
            timeout_occurred = True
        
        # Set timeout for the entire scan (5 minutes)
        timer = threading.Timer(300.0, timeout_handler)  # 5 minutes timeout
        timer.start()
        
        try:
            self.logger.info(f"Scanning {symbol} for collar opportunities...")
            
            # Create stock contract with primary exchange for ambiguous tickers
            if symbol == 'A':
                stk = Stock(symbol, 'SMART', 'USD', primaryExchange='NYSE')
            elif symbol in ['M', 'T', 'F', 'C', 'BAC', 'GE', 'GM', 'JPM', 'WMT']:
                # Other commonly ambiguous tickers
                stk = Stock(symbol, 'SMART', 'USD', primaryExchange='NYSE')
            else:
                stk = Stock(symbol, 'SMART', 'USD')
            
            # Qualify stock contract with timeout
            qualified_stk = qualify_with_timeout(self.ib, stk, timeout=8.0)
            if not qualified_stk:
                self.logger.warning(f"Could not qualify stock contract for {symbol} within timeout")
                return []
            
            stk = qualified_stk
            self.logger.debug(f"Qualified stock contract for {symbol}: {stk}")
            
            # Check timeout before proceeding
            if timeout_occurred:
                raise TimeoutError(f"Scan timeout for {symbol}")
            
            # Get stock price
            stock_price = self.stock_mid_price(stk)
            if not stock_price:
                self.logger.warning(f"Could not get price for {symbol}")
                return []
            
            self.logger.debug(f"Stock {symbol} price: ${stock_price:.2f}")
            
            # Check timeout before proceeding
            if timeout_occurred:
                raise TimeoutError(f"Scan timeout for {symbol}")
            
            # Get option parameters
            params = self.ib.reqSecDefOptParams(stk.symbol, '', stk.secType, stk.conId)
            if not params:
                self.logger.warning(f"No option parameters for {symbol}")
                return []
            
            # prefer SMART; otherwise keep all for later fallbacks
            p_smart = next((pp for pp in params if getattr(pp, 'exchange', '') == 'SMART'), None)
            p_list = [p_smart] + [pp for pp in params if pp is not p_smart] if p_smart else params
            
            # Use the first record for basic info (expirations, strikes)
            p = p_list[0]
            
            self.logger.debug(f"Option parameters for {symbol}: {len(params)} SecDef records, {len(p.expirations)} expirations, {len(p.strikes)} strikes")
            
            # Process expirations
            expiries = []
            for e in p.expirations:
                if '-' not in e:
                    expiries.append(f"{e[:4]}-{e[4:6]}-{e[6:]}")
                else:
                    expiries.append(e)
            
            # Filter by DTE
            expiries = [e for e in expiries if self.in_range(self.dte(e), self.config.MIN_DTE, self.config.MAX_DTE)]
            
            if not expiries:
                self.logger.info(f"No suitable expirations for {symbol} (DTE range: {self.config.MIN_DTE}-{self.config.MAX_DTE})")
                return []
            
            self.logger.debug(f"Filtered to {len(expiries)} suitable expirations for {symbol}")
            
            results = []
            
            for expiry in sorted(expiries, key=lambda e: self.dte(e)):
                # Check timeout before processing each expiry
                if timeout_occurred:
                    raise TimeoutError(f"Scan timeout for {symbol}")
                
                dte = self.dte(expiry)
                self.logger.debug(f"Processing {symbol} expiry {expiry} (DTE: {dte})")
                
                # Get strikes around current stock price - INCREASED TO 6-8 STRIKES
                approx_strikes = sorted([float(s) for s in p.strikes if s and s != '0'])
                
                # If we have a valid stock price, filter strikes around it
                if stock_price and stock_price > 0 and not math.isnan(stock_price):
                    band = [K for K in approx_strikes if 0.5 * stock_price <= K <= 1.5 * stock_price]
                    # Sample strikes nearest to current price - increased to 6-8 strikes
                    band = sorted(band, key=lambda K: abs(K - stock_price))[:8]  # Increased from 4 to 8
                else:
                    # Fallback: use strikes around the middle of the available range
                    if approx_strikes:
                        mid_strike = (min(approx_strikes) + max(approx_strikes)) / 2
                        band = sorted(approx_strikes, key=lambda K: abs(K - mid_strike))[:8]  # Increased from 4 to 8
                        self.logger.debug(f"Using fallback strike selection for {symbol} {expiry} (mid_strike=${mid_strike:.2f})")
                    else:
                        band = []
                
                self.logger.debug(f"Selected {len(band)} strikes for {symbol} {expiry}: {[f'${k:.2f}' for k in band]}")
                
                # Fetch option data
                temp_puts, temp_calls = [], []
                
                self.logger.debug(f"Starting to fetch option data for {symbol} {expiry} - processing {len(band)} strikes...")
                
                for i, K in enumerate(band):
                    # Check timeout before processing each strike
                    if timeout_occurred:
                        raise TimeoutError(f"Scan timeout for {symbol}")
                    
                    self.logger.debug(f"Processing strike {i+1}/{len(band)}: ${K:.2f} for {symbol} {expiry}")
                    
                    # Qualify contracts with timeout using async helper
                    self.logger.debug(f"  Qualifying contracts for {symbol} {K} {expiry}...")
                    
                    # Create both contracts with correct expiry format and smart fallbacks
                    expiry_ib = expiry.replace('-', '')  # '2025-10-17' -> '20251017'
                    
                    # Use fallback strike resolution to handle non-existent strikes
                    put = self.resolve_strike_with_fallback(symbol, expiry_ib, K, 'P', timeout=12.0)
                    call = self.resolve_strike_with_fallback(symbol, expiry_ib, K, 'C', timeout=12.0)
                    
                    if not put or not call:
                        self.logger.debug(f"    Could not resolve/qualify {symbol} {K} {expiry} on SMART (with fallback)")
                        continue
                    
                    self.logger.debug(f"    Qualified {put.symbol} {K} {put.right} and {call.symbol} {K} {call.right}")
                    
                    # Get put data with timeout
                    self.logger.debug(f"  Getting put data for {symbol} {K} {expiry}...")
                    
                    # Timeout for put data
                    put_timeout = False
                    def put_timeout_handler():
                        nonlocal put_timeout
                        put_timeout = True
                    
                    put_timer = threading.Timer(15.0, put_timeout_handler)  # 15 second timeout
                    put_timer.start()
                    
                    try:
                        p_mid, p_ticker = self.option_mid_price(put)
                        if put_timeout:
                            self.logger.debug(f"    Put data timeout for {symbol} {K} {expiry}")
                            p_mid = None
                        self.logger.debug(f"    Put mid price: {p_mid}")
                    finally:
                        put_timer.cancel()
                    
                    # Timeout for put delta
                    put_delta_timeout = False
                    def put_delta_timeout_handler():
                        nonlocal put_delta_timeout
                        put_delta_timeout = True
                    
                    put_delta_timer = threading.Timer(15.0, put_delta_timeout_handler)  # 15 second timeout
                    put_delta_timer.start()
                    
                    try:
                        self.logger.debug(f"  Getting put delta for {symbol} {K} {expiry}...")
                        p_delta = self.option_model_delta(put)
                        if put_delta_timeout:
                            self.logger.debug(f"    Put delta timeout for {symbol} {K} {expiry}")
                            p_delta = None
                        
                        # If no OPRA delta, try to calculate implied delta using professional solver
                        if p_delta is None and p_mid is not None:
                            self.logger.debug(f"    Calculating implied delta for put {symbol} {K} {expiry}")
                            p_delta = self.calculate_implied_delta_professional(
                                stock_price, K, expiry_ib, p_mid, 'put',
                                risk_free_rate=getattr(self.config, 'RISK_FREE_RATE', 0.045),
                                dividend_yield=getattr(self.config, 'DIVIDEND_YIELD_DEFAULT', 0.0)
                            )
                            if p_delta is not None:
                                self.logger.debug(f"    Implied put delta: {p_delta:.3f}")
                        
                        self.logger.debug(f"    Put delta: {p_delta}")
                    finally:
                        put_delta_timer.cancel()
                    
                    # keep when mid exists; delta may be None
                    if p_mid is not None:
                        temp_puts.append((put, p_delta, p_mid, K))
                        self.logger.debug(f"    Added put {symbol} {K} {expiry}: mid=${p_mid:.4f}, delta={p_delta}")
                    else:
                        self.logger.debug(f"    Put {symbol} {K} {expiry}: no mid")
                    
                    # Get call data with timeout
                    self.logger.debug(f"  Getting call data for {symbol} {K} {expiry}...")
                    
                    # Timeout for call data
                    call_timeout = False
                    def call_timeout_handler():
                        nonlocal call_timeout
                        call_timeout = True
                    
                    call_timer = threading.Timer(15.0, call_timeout_handler)  # 15 second timeout
                    call_timer.start()
                    
                    try:
                        c_mid, c_ticker = self.option_mid_price(call)
                        if call_timeout:
                            self.logger.debug(f"    Call data timeout for {symbol} {K} {expiry}")
                            c_mid = None
                        self.logger.debug(f"    Call mid price: {c_mid}")
                    finally:
                        call_timer.cancel()
                    
                    # Timeout for call delta
                    call_delta_timeout = False
                    def call_delta_timeout_handler():
                        nonlocal call_delta_timeout
                        call_delta_timeout = True
                    
                    call_delta_timer = threading.Timer(15.0, call_delta_timeout_handler)  # 15 second timeout
                    call_delta_timer.start()
                    
                    try:
                        self.logger.debug(f"  Getting call delta for {symbol} {K} {expiry}...")
                        c_delta = self.option_model_delta(call)
                        if call_delta_timeout:
                            self.logger.debug(f"    Call delta timeout for {symbol} {K} {expiry}")
                            c_delta = None
                        
                        # If no OPRA delta, try to calculate implied delta using professional solver
                        if c_delta is None and c_mid is not None:
                            self.logger.debug(f"    Calculating implied delta for call {symbol} {K} {expiry}")
                            c_delta = self.calculate_implied_delta_professional(
                                stock_price, K, expiry_ib, c_mid, 'call',
                                risk_free_rate=getattr(self.config, 'RISK_FREE_RATE', 0.045),
                                dividend_yield=getattr(self.config, 'DIVIDEND_YIELD_DEFAULT', 0.0)
                            )
                            if c_delta is not None:
                                self.logger.debug(f"    Implied call delta: {c_delta:.3f}")
                        
                        self.logger.debug(f"    Call delta: {c_delta}")
                    finally:
                        call_delta_timer.cancel()
                    
                    # keep when mid exists; delta may be None
                    if c_mid is not None:
                        temp_calls.append((call, c_delta, c_mid, K))
                        self.logger.debug(f"    Added call {symbol} {K} {expiry}: mid=${c_mid:.4f}, delta={c_delta}")
                    else:
                        self.logger.debug(f"    Call {symbol} {K} {expiry}: no mid")
                    
                    self.logger.debug(f"  Completed strike {i+1}/{len(band)} for {symbol} {expiry}")
                    
                    # Small backoff between strikes to avoid overwhelming IB API
                    import random
                    backoff = random.uniform(0.05, 0.15)  # 50-150ms
                    self.ib.sleep(backoff)
                
                self.logger.debug(f"Collected {len(temp_puts)} puts and {len(temp_calls)} calls for {symbol} {expiry}")
                
                # Skip this expiry if we got no data after processing all strikes
                if len(temp_puts) == 0 and len(temp_calls) == 0:
                    self.logger.debug(f"Skipping {symbol} {expiry} - no option data collected")
                    continue
                
                # Pick candidates by delta
                put_candidates = self.pick_by_delta(temp_puts, self.config.PUT_DELTA_RANGE, 'put', stock_price)
                call_candidates = self.pick_by_delta(temp_calls, self.config.CALL_DELTA_RANGE, 'call', stock_price)
                
                self.logger.debug(f"Selected {len(put_candidates)} put candidates and {len(call_candidates)} call candidates for {symbol} {expiry}")
                
                # Always form at least one pairing if we have candidates on both sides
                if len(put_candidates) == 0 or len(call_candidates) == 0:
                    self.logger.debug(f"Skipping {symbol} {expiry} - insufficient candidates (puts: {len(put_candidates)}, calls: {len(call_candidates)})")
                    continue
                
                # If we have very few candidates, ensure we use them all
                if len(put_candidates) < 2:
                    self.logger.debug(f"Using all {len(put_candidates)} put candidates for {symbol} {expiry}")
                if len(call_candidates) < 2:
                    self.logger.debug(f"Using all {len(call_candidates)} call candidates for {symbol} {expiry}")
                
                # Pair puts and calls
                for pc in put_candidates:
                    put, p_delta, p_mid, Kp = pc
                    
                    for cc in call_candidates:
                        call, c_delta, c_mid, Kc = cc
                        
                        # Only meaningful if call strike >= put strike
                        if Kc < Kp:
                            self.logger.debug(f"Skipping {symbol} {expiry}: call strike ${Kc:.2f} < put strike ${Kp:.2f}")
                            continue
                        
                        # Calculate collar metrics
                        C0 = 100 * stock_price + p_mid - c_mid  # Net cost
                        floor = 100 * Kp - C0                   # Loss floor
                        max_gain = 100 * Kc - C0                # Max gain if called away
                        capital_used = C0
                        
                        if capital_used is None or max_gain is None:
                            self.logger.debug(f"Skipping {symbol} {expiry}: invalid calculations (C0={C0}, max_gain={max_gain})")
                            continue
                        
                        # Check spread quality - reuse existing ticker objects
                        try:
                            # Use the tickers we already have from option_mid_price
                            put_spread = self.calculate_spread_percentage(p_ticker.bid, p_ticker.ask) if p_ticker and p_ticker.bid and p_ticker.ask else 0
                            call_spread = self.calculate_spread_percentage(c_ticker.bid, c_ticker.ask) if c_ticker and c_ticker.bid and c_ticker.ask else 0
                        except Exception as e:
                            self.logger.debug(f"Could not get spread data for {symbol} {expiry}: {e}")
                            put_spread = 0
                            call_spread = 0
                        
                        # If we don't have spread data, skip spread filtering but continue with other filters
                        if put_spread == 0 and call_spread == 0:
                            self.logger.debug(f"  No spread data available for {symbol} {expiry}, skipping spread filters")
                        
                        # Use safe_round to handle None deltas
                        pd = self.safe_round(p_delta, 3)
                        cd = self.safe_round(c_delta, 3)
                        self.logger.debug(f"Analyzing {symbol} {expiry} collar: put ${Kp:.2f} (delta={pd}, spread={put_spread*100:.1f}%), call ${Kc:.2f} (delta={cd}, spread={call_spread*100:.1f}%)")
                        self.logger.debug(f"  Net cost: ${C0:.2f}, Floor: ${floor:.2f}, Max gain: ${max_gain:.2f}")
                        
                        # Apply filters
                        filters_passed = []
                        filters_failed = []
                        
                        if capital_used <= self.config.CAPITAL_BUDGET:
                            filters_passed.append(f"capital_used (${capital_used:.2f}) <= budget (${self.config.CAPITAL_BUDGET})")
                        else:
                            filters_failed.append(f"capital_used (${capital_used:.2f}) > budget (${self.config.CAPITAL_BUDGET})")
                        
                        if capital_used >= self.config.MIN_CAPITAL_USED:
                            filters_passed.append(f"capital_used (${capital_used:.2f}) >= min (${self.config.MIN_CAPITAL_USED})")
                        else:
                            filters_failed.append(f"capital_used (${capital_used:.2f}) < min (${self.config.MIN_CAPITAL_USED})")
                        
                        # More lenient floor tolerance for near-zero-loss opportunities
                        floor_tolerance = max(self.config.FLOOR_TOLERANCE, self.config.ZERO_COST_TOLERANCE)
                        if floor >= floor_tolerance:
                            filters_passed.append(f"floor (${floor:.2f}) >= tolerance (${floor_tolerance})")
                        else:
                            filters_failed.append(f"floor (${floor:.2f}) < tolerance (${floor_tolerance})")
                        
                        if put_spread <= self.config.MAX_SPREAD_PCT:
                            filters_passed.append(f"put_spread ({put_spread*100:.1f}%) <= max ({self.config.MAX_SPREAD_PCT*100:.1f}%)")
                        else:
                            filters_failed.append(f"put_spread ({put_spread*100:.1f}%) > max ({self.config.MAX_SPREAD_PCT*100:.1f}%)")
                        
                        if call_spread <= self.config.MAX_SPREAD_PCT:
                            filters_passed.append(f"call_spread ({call_spread*100:.1f}%) <= max ({self.config.MAX_SPREAD_PCT*100:.1f}%)")
                        else:
                            filters_failed.append(f"call_spread ({call_spread*100:.1f}%) > max ({self.config.MAX_SPREAD_PCT*100:.1f}%)")
                        
                        # If we don't have spread data, don't fail on spread filters
                        if put_spread == 0 and call_spread == 0:
                            self.logger.debug(f"  Skipping spread filters due to missing data")
                            # Remove any spread filter failures
                            filters_failed = [f for f in filters_failed if 'spread' not in f]
                        
                        if len(filters_failed) == 0:
                            self.logger.debug(f"PASSED {symbol} {expiry} - All filters passed:")
                            for filter_msg in filters_passed:
                                self.logger.debug(f"  PASS: {filter_msg}")
                            
                            capital_at_risk = max(0, C0 - 100 * Kp)
                            
                            # Scoring: prioritize zero-risk opportunities with spread penalty
                            score_risk_reward = max_gain / max(1.0, capital_at_risk + 1e-6)
                            score_gain_per_capital = max_gain / max(1.0, capital_used)
                            
                            # Apply spread penalty for illiquid options
                            spread_penalty = 1.0
                            if put_spread > 0.25 or call_spread > 0.25:  # >25% spread
                                spread_penalty = 0.5  # Reduce score by 50%
                                self.logger.debug(f"  Applying spread penalty (put: {put_spread*100:.1f}%, call: {call_spread*100:.1f}%)")
                            
                            score_risk_reward *= spread_penalty
                            score_gain_per_capital *= spread_penalty
                            
                            self.logger.debug(f"  Final scores: risk_reward={score_risk_reward:.2f}, gain_per_capital={score_gain_per_capital:.2f}")
                            
                            # Determine if this is true zero-loss or near-zero-loss
                            is_true_zero_loss = floor >= 0
                            loss_tolerance = "ZERO" if is_true_zero_loss else "NEAR_ZERO"
                            
                            results.append({
                                'symbol': symbol,
                                'expiry': expiry,
                                'stock_mid': round(stock_price, 4),
                                'put_K': Kp,
                                'put_mid': round(p_mid, 4),
                                'put_delta': self.safe_round(p_delta, 3),
                                'put_spread_pct': round(put_spread * 100, 2),
                                'call_K': Kc,
                                'call_mid': round(c_mid, 4),
                                'call_delta': self.safe_round(c_delta, 3),
                                'call_spread_pct': round(call_spread * 100, 2),
                                'net_cost_C0': round(C0, 2),
                                'floor_$': round(floor, 2),
                                'max_gain_$': round(max_gain, 2),
                                'capital_used_$': round(capital_used, 2),
                                'capital_at_risk_$': round(capital_at_risk, 2),
                                'dte': self.dte(expiry),
                                'score_risk_reward': round(score_risk_reward, 2),
                                'score_gain_per_capital': round(score_gain_per_capital, 2),
                                'loss_tolerance': loss_tolerance,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            self.logger.debug(f"FAILED {symbol} {expiry} - Some filters failed:")
                            for filter_msg in filters_failed:
                                self.logger.debug(f"  FAIL: {filter_msg}")
                            for filter_msg in filters_passed:
                                self.logger.debug(f"  PASS: {filter_msg}")
            
            # Rank results: prioritize true zero-loss, then by score
            results.sort(key=lambda r: (r['loss_tolerance'] != 'ZERO', -r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
            
            if results:
                self.logger.info(f"Found {len(results)} collar opportunities for {symbol}")
            else:
                self.logger.debug(f"No collar opportunities found for {symbol}")
            
            return results
            
        except TimeoutError as e:
            self.logger.error(f"Scan timeout for {symbol}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error scanning {symbol}: {e}")
            return []
        finally:
            # Cancel the timer
            timer.cancel()
    
    def resolve_strike_with_fallback(self, symbol: str, expiry_ib: str, target_strike: float, 
                                    right: str, timeout: float = 8.0) -> Optional[Contract]:
        """
        Resolve a strike with fallback to nearby strikes if the target doesn't exist.
        Handles cases where half-dollar strikes don't exist for certain expiries.
        """
        # Try the target strike first
        contract = resolve_option_smart(self.ib, symbol, expiry_ib, target_strike, right, timeout)
        if contract:
            return contract
        
        # If target strike fails, try nearby strikes
        # Determine strike increment based on price level
        if target_strike >= 200:
            increment = 5.0  # $5 increments for high-priced stocks
        elif target_strike >= 100:
            increment = 2.5  # $2.5 increments for mid-priced stocks
        else:
            increment = 1.0  # $1 increments for low-priced stocks
        
        # Try strikes above and below the target
        fallback_strikes = [
            target_strike + increment,
            target_strike - increment,
            target_strike + 2 * increment,
            target_strike - 2 * increment,
        ]
        
        for fallback_strike in fallback_strikes:
            if fallback_strike <= 0:
                continue
                
            self.logger.debug(f"  Trying fallback strike ${fallback_strike:.2f} for {symbol} {right} {expiry_ib}")
            contract = resolve_option_smart(self.ib, symbol, expiry_ib, fallback_strike, right, timeout)
            if contract:
                self.logger.debug(f"  Successfully resolved fallback strike ${fallback_strike:.2f}")
                return contract
        
        return None
    
    def print_results(self, results: List[Dict]):
        """Print results in a formatted table"""
        if not results:
            print("\nNo collar opportunities found matching criteria.")
            return
        
        print(f"\n{'='*130}")
        print(f"ZERO-COST COLLAR OPPORTUNITIES (Budget: ${self.config.CAPITAL_BUDGET:,})")
        print(f"{'='*130}")
        print(f"{'SYM':<6} {'EXP':<10} {'PX':>7} {'P_K':>6} {'P_mid':>7} {'P_delta':>7} {'C_K':>6} {'C_mid':>7} {'C_delta':>7} {'C0':>9} {'FLOOR':>8} {'MAXG':>8} {'DTE':>4} {'TYPE':>8} {'SCORE':>8}")
        print(f"{'-'*130}")
        
        for r in results[:self.config.MAX_RESULTS]:
            put_delta_str = f"{r['put_delta']:.2f}" if r['put_delta'] is not None else "N/A"
            call_delta_str = f"{r['call_delta']:.2f}" if r['call_delta'] is not None else "N/A"
            
            print(f"{r['symbol']:<6} {r['expiry']:<10} {r['stock_mid']:>7.2f} "
                  f"{r['put_K']:>6.2f} {r['put_mid']:>7.2f} {put_delta_str:>5} "
                  f"{r['call_K']:>6.2f} {r['call_mid']:>7.2f} {call_delta_str:>5} "
                  f"{r['net_cost_C0']:>9.2f} {r['floor_$']:>8.2f} {r['max_gain_$']:>8.2f} "
                  f"{r['dte']:>4} {r['loss_tolerance']:>8} {r['score_risk_reward']:>8.2f}")
        
        print(f"\nFound {len(results)} opportunities, showing top {min(self.config.MAX_RESULTS, len(results))}")
        print(f"Legend: PX=Stock Price, P_K/C_K=Put/Call Strike, P_mid/C_mid=Put/Call Mid Price")
        print(f"        P_delta/C_delta=Put/Call Delta, C0=Net Cost, FLOOR=Loss Floor, MAXG=Max Gain")
        print(f"        DTE=Days to Expiry, TYPE=ZERO/NEAR_ZERO, SCORE=Risk-Reward Score")
    
    def save_to_csv(self, results: List[Dict], append: bool = False):
        """Save results to CSV file with option to append or overwrite"""
        if not self.config.SAVE_TO_CSV or not results:
            return
        
        try:
            import pandas as pd
            
            # Ensure reports directory exists
            os.makedirs('reports', exist_ok=True)
            
            df = pd.DataFrame(results)
            
            if append and os.path.exists(self.config.CSV_FILENAME):
                # Append to existing file
                existing_df = pd.read_csv(self.config.CSV_FILENAME)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(self.config.CSV_FILENAME, index=False)
                self.logger.info(f"Appended {len(results)} results to {self.config.CSV_FILENAME}")
            else:
                # Create new file or overwrite
                df.to_csv(self.config.CSV_FILENAME, index=False)
                self.logger.info(f"Results saved to {self.config.CSV_FILENAME}")
            
        except ImportError:
            self.logger.warning("pandas not available, skipping CSV export")
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
    
    def update_report_after_symbol(self, symbol: str, symbol_results: List[Dict], all_results: List[Dict]):
        """Update the report file after each symbol is processed"""
        if not self.config.SAVE_TO_CSV:
            return
        
        try:
            import pandas as pd
            
            # Ensure reports directory exists
            os.makedirs('reports', exist_ok=True)
            
            # Create a summary of current progress
            total_symbols = len(self.config.UNIVERSE)
            processed_symbols = len([r for r in all_results if r.get('symbol')])
            unique_symbols = len(set(r.get('symbol') for r in all_results))
            
            # Create progress summary
            progress_summary = {
                'timestamp': datetime.now().isoformat(),
                'symbol_processed': symbol,
                'symbols_found': len(symbol_results),
                'total_symbols_processed': processed_symbols,
                'unique_symbols_with_opportunities': unique_symbols,
                'total_opportunities_found': len(all_results),
                'progress_percent': round((processed_symbols / total_symbols) * 100, 1)
            }
            
            # Save current results
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(self.config.CSV_FILENAME, index=False)
            
            # Create or update progress file
            progress_file = self.config.CSV_FILENAME.replace('.csv', '_progress.csv')
            progress_df = pd.DataFrame([progress_summary])
            
            if os.path.exists(progress_file):
                # Append to existing progress file
                existing_progress = pd.read_csv(progress_file)
                combined_progress = pd.concat([existing_progress, progress_df], ignore_index=True)
                combined_progress.to_csv(progress_file, index=False)
            else:
                # Create new progress file
                progress_df.to_csv(progress_file, index=False)
            
            # Log progress
            self.logger.info(f"Progress: {progress_summary['progress_percent']}% ({processed_symbols}/{total_symbols}) - {symbol}: {len(symbol_results)} opportunities")
            
        except ImportError:
            self.logger.warning("pandas not available, skipping progress update")
        except Exception as e:
            self.logger.error(f"Error updating progress: {e}")
    
    def print_progress_summary(self, all_results: List[Dict], current_symbol: str = None):
        """Print a summary of current progress"""
        if not all_results:
            return
        
        total_symbols = len(self.config.UNIVERSE)
        unique_symbols = len(set(r.get('symbol') for r in all_results))
        total_opportunities = len(all_results)
        
        # Calculate progress
        if current_symbol:
            # Estimate progress based on symbol position in universe
            try:
                symbol_index = self.config.UNIVERSE.index(current_symbol)
                progress_percent = round(((symbol_index + 1) / total_symbols) * 100, 1)
            except ValueError:
                progress_percent = 0
        else:
            progress_percent = 100
        
        print(f"\n{'='*60}")
        print(f"PROGRESS SUMMARY")
        print(f"{'='*60}")
        print(f"Progress: {progress_percent}% ({unique_symbols}/{total_symbols} symbols with opportunities)")
        print(f"Total Opportunities Found: {total_opportunities}")
        print(f"Current Symbol: {current_symbol if current_symbol else 'Completed'}")
        print(f"Report File: {self.config.CSV_FILENAME}")
        print(f"Progress File: {self.config.CSV_FILENAME.replace('.csv', '_progress.csv')}")
        print(f"{'='*60}")
        
        # Show top 5 opportunities so far
        if all_results:
            print(f"\nTop 5 Opportunities Found So Far:")
            print(f"{'SYM':<6} {'EXP':<10} {'C0':>9} {'FLOOR':>8} {'MAXG':>8} {'TYPE':>8}")
            print(f"{'-'*50}")
            
            # Sort by score and show top 5
            sorted_results = sorted(all_results, key=lambda r: (r['loss_tolerance'] != 'ZERO', -r['score_risk_reward']))
            for r in sorted_results[:5]:
                print(f"{r['symbol']:<6} {r['expiry']:<10} {r['net_cost_C0']:>9.2f} {r['floor_$']:>8.2f} {r['max_gain_$']:>8.2f} {r['loss_tolerance']:>8}")
        
        print()
    
    def run(self):
        """Run the complete screener"""
        print("Enhanced Zero-Cost Collar Screener")
        print("=" * 50)
        print(f"Budget: ${self.config.CAPITAL_BUDGET:,}")
        print(f"DTE Range: {self.config.MIN_DTE}-{self.config.MAX_DTE} days")
        print(f"Put Delta: {self.config.PUT_DELTA_RANGE}")
        print(f"Call Delta: {self.config.CALL_DELTA_RANGE}")
        print(f"Universe: {len(self.config.UNIVERSE)} symbols")
        print(f"Log Level: {self.config.LOG_LEVEL}")
        if self.config.LOG_TO_FILE:
            print(f"Log File: {self.config.LOG_FILENAME}")
        else:
            print("Log File: Console only")
        print(f"Real-time Updates: {'Enabled' if self.config.REAL_TIME_UPDATES else 'Disabled'}")
        print(f"Progress Frequency: Every {self.config.PROGRESS_UPDATE_FREQUENCY} symbols")
        print()
        
        # Connect to IB
        if not self.connect_ib():
            return
        
        try:
            # Scan all symbols
            all_results = []
            total_symbols = len(self.config.UNIVERSE)
            
            self.logger.info(f"Starting scan of {total_symbols} symbols...")
            
            for i, symbol in enumerate(self.config.UNIVERSE, 1):
                try:
                    self.logger.info(f"Progress: {i}/{total_symbols} ({i/total_symbols*100:.1f}%) - Scanning {symbol}")
                    
                    results = self.scan_ticker(symbol)
                    if results:
                        self.logger.info(f"FOUND {symbol}: Found {len(results)} opportunities")
                        all_results.extend(results)
                    else:
                        self.logger.debug(f"NO_OPPORTUNITIES {symbol}: No opportunities found")
                    
                    # Update report after each symbol if real-time updates are enabled
                    if self.config.REAL_TIME_UPDATES:
                        self.update_report_after_symbol(symbol, results, all_results)
                    
                    # Print progress summary based on frequency or when opportunities are found
                    if i % self.config.PROGRESS_UPDATE_FREQUENCY == 0 or results:
                        self.print_progress_summary(all_results, symbol)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    # Still update progress even if there was an error
                    self.update_report_after_symbol(symbol, [], all_results)
                    continue
            
            # Sort all results: prioritize true zero-loss, then by score
            all_results.sort(key=lambda r: (r['loss_tolerance'] != 'ZERO', -r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
            
            self.logger.info(f"Scan complete! Found {len(all_results)} total opportunities across {total_symbols} symbols")
            
            # Print final progress summary
            self.print_progress_summary(all_results)
            
            # Print and save results
            self.print_results(all_results)
            self.save_to_csv(all_results)
            
        finally:
            # Always disconnect
            self.disconnect_ib()

    def calculate_implied_delta(self, stock_price: float, strike: float, time_to_expiry: float, 
                               option_price: float, option_type: str = 'call', 
                               risk_free_rate: float = 0.045) -> Optional[float]:
        """
        Calculate implied delta using Black-Scholes approximation when OPRA is not available.
        Uses bisection to find implied volatility, then calculates delta.
        """
        try:
            import math
            
            def normal_cdf(x):
                """Approximate normal CDF"""
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            def normal_pdf(x):
                """Normal PDF"""
                return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
            
            def black_scholes_call(S, K, T, r, sigma):
                """Black-Scholes call option price"""
                if T <= 0 or sigma <= 0:
                    return None
                    
                d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
                d2 = d1 - sigma * math.sqrt(T)
                
                call_price = S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
                return call_price
            
            def black_scholes_put(S, K, T, r, sigma):
                """Black-Scholes put option price"""
                if T <= 0 or sigma <= 0:
                    return None
                    
                d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
                d2 = d1 - sigma * math.sqrt(T)
                
                put_price = K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
                return put_price
            
            def black_scholes_delta(S, K, T, r, sigma, option_type):
                """Black-Scholes delta"""
                if T <= 0 or sigma <= 0:
                    return None
                    
                d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
                
                if option_type == 'call':
                    return normal_cdf(d1)
                else:  # put
                    return normal_cdf(d1) - 1
            
            # Convert time to years
            T = time_to_expiry / 365.0
            
            if T <= 0:
                return None
            
            # Use bisection to find implied volatility
            sigma_min, sigma_max = 0.1, 1.2  # 10% to 120% volatility range
            tolerance = 0.001
            max_iterations = 50
            
            for i in range(max_iterations):
                sigma = (sigma_min + sigma_max) / 2
                
                if option_type == 'call':
                    theoretical_price = black_scholes_call(stock_price, strike, T, risk_free_rate, sigma)
                else:
                    theoretical_price = black_scholes_put(stock_price, strike, T, risk_free_rate, sigma)
                
                if theoretical_price is None:
                    return None
                
                price_diff = theoretical_price - option_price
                
                if abs(price_diff) < tolerance:
                    # Found implied volatility, calculate delta
                    delta = black_scholes_delta(stock_price, strike, T, risk_free_rate, sigma, option_type)
                    return delta
                
                if price_diff > 0:
                    sigma_max = sigma
                else:
                    sigma_min = sigma
            
            # If bisection didn't converge, return None
            return None
            
        except Exception as e:
            self.logger.debug(f"Error calculating implied delta: {e}")
            return None

    # ---- Professional Black-Scholes IV Solver ----
    def _phi(self, x):
        """Standard normal CDF (erf-based; good enough)"""
        import math
        SQRT2 = math.sqrt(2.0)
        return 0.5 * (1.0 + math.erf(x / SQRT2))

    def _d1(self, S, K, T, r, q, iv):
        """Calculate d1 for Black-Scholes"""
        import math
        if iv <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None
        return (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))

    def bs_price(self, S, K, T, r=0.045, q=0.0, iv=0.25, right='C'):
        """European BS price with continuous dividend yield q."""
        import math
        if T <= 0 or S <= 0 or K <= 0 or iv <= 0:
            return None
        d1 = self._d1(S, K, T, r, q, iv)
        if d1 is None:
            return None
        d2 = d1 - iv * math.sqrt(T)
        df_r = math.exp(-r * T)
        df_q = math.exp(-q * T)
        if right == 'C':
            return df_q * S * self._phi(d1) - df_r * K * self._phi(d2)
        else:
            return df_r * K * self._phi(-d2) - df_q * S * self._phi(-d1)

    def bs_delta(self, S, K, T, r=0.045, q=0.0, iv=0.25, right='C'):
        """BS delta (∂Price/∂S)."""
        d1 = self._d1(S, K, T, r, q, iv)
        if d1 is None:
            return None
        import math
        df_q = math.exp(-q * T)
        if right == 'C':
            return df_q * self._phi(d1)
        else:
            return -df_q * self._phi(-d1)

    def implied_vol_bisection(self, target_price, S, K, T, r=0.045, q=0.0,
                              right='C', iv_low=1e-4, iv_high=5.0, tol=1e-4, max_iter=80):
        """
        Solve for IV using bisection. Returns (iv, converged:bool).
        If price is out of arbitrage bounds, returns (None, False).
        """
        import math
        if target_price is None or target_price <= 0 or S <= 0 or K <= 0 or T <= 0:
            return (None, False)

        # crude no-arb bounds for L1 sanity (European with dividends)
        df_r = math.exp(-r * T); df_q = math.exp(-q * T)
        if right == 'C':
            lower = max(0.0, df_q * S - df_r * K)
            upper = df_q * S
        else:
            lower = max(0.0, df_r * K - df_q * S)
            upper = df_r * K
        if not (lower - 1e-6 <= target_price <= upper + 1e-6):
            return (None, False)

        lo, hi = iv_low, iv_high
        f_lo = self.bs_price(S, K, T, r, q, lo, right)
        f_hi = self.bs_price(S, K, T, r, q, hi, right)
        if f_lo is None or f_hi is None:
            return (None, False)

        # Ensure target lies between f(lo) and f(hi); otherwise expand hi
        if f_lo > target_price:
            return (None, False)
        tries = 0
        while f_hi < target_price and hi < 10.0 and tries < 6:
            hi *= 1.5
            f_hi = self.bs_price(S, K, T, r, q, hi, right)
            tries += 1
            if f_hi is None:
                return (None, False)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = self.bs_price(S, K, T, r, q, mid, right)
            if f_mid is None:
                return (None, False)
            if abs(f_mid - target_price) < tol:
                return (mid, True)
            if f_mid < target_price:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi), False)

    def year_fraction(self, expiry_yyyymmdd: str, now=None):
        """
        Act/365F from 'YYYYMMDD' (IB format) using current UTC as 'now'.
        """
        from datetime import datetime, timezone
        now = now or datetime.now(timezone.utc)
        y, m, d = int(expiry_yyyymmdd[0:4]), int(expiry_yyyymmdd[4:6]), int(expiry_yyyymmdd[6:8])
        exp = datetime(y, m, d, 20, 0, 0, tzinfo=timezone.utc)  # ~after close cushion
        T = (exp - now).total_seconds() / (365.0 * 24 * 3600.0)
        return max(T, 1e-6)

    def calculate_implied_delta_professional(self, stock_price: float, strike: float, 
                                           expiry_ib: str, option_price: float, 
                                           option_type: str = 'call', 
                                           risk_free_rate: float = 0.045,
                                           dividend_yield: float = 0.0) -> Optional[float]:
        """
        Professional-grade implied delta calculation using robust IV solver.
        Returns (delta, iv) or (None, None) if calculation fails.
        """
        try:
            # Convert option type to right
            right = 'C' if option_type == 'call' else 'P'
            
            # Calculate time to expiry
            T = self.year_fraction(expiry_ib)
            
            # Solve for implied volatility
            iv, converged = self.implied_vol_bisection(
                target_price=option_price,
                S=stock_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                q=dividend_yield,
                right=right
            )
            
            if iv is None or not converged:
                return None
            
            # Calculate delta using the implied volatility
            delta = self.bs_delta(
                S=stock_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                q=dividend_yield,
                iv=iv,
                right=right
            )
            
            return delta
            
        except Exception as e:
            self.logger.debug(f"Error in professional implied delta calculation: {e}")
            return None

def main():
    """Main function"""
    args = parse_arguments()
    
    # Handle verbose flag
    if args.verbose:
        args.log_level = 'DEBUG'
    
    config = CollarScreenerConfig(args.config, args.log_level)
    
    # Override log file settings from command line arguments
    if args.log_file:
        config.LOG_FILENAME = args.log_file
        config.LOG_TO_FILE = True
    elif args.no_log_file:
        config.LOG_TO_FILE = False
    
    # Override progress settings from command line arguments
    if args.no_real_time_updates:
        config.REAL_TIME_UPDATES = False
    if args.progress_frequency:
        config.PROGRESS_UPDATE_FREQUENCY = args.progress_frequency
    
    screener = CollarScreener(config)
    screener.run()

if __name__ == '__main__':
    main()
