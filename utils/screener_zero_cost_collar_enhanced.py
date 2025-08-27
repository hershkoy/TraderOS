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
            # Override config log level with CLI parameter
            self.LOG_LEVEL = self.log_level
            
            # Advanced Settings
            self.MAX_STRIKES_PER_EXPIRY = config['advanced']['max_strikes_per_expiry']
            self.REQUIRE_MODEL_GREEKS = config['advanced']['require_model_greeks']
            self.MIN_VOLUME = config['advanced']['min_volume']
            self.REQUEST_TIMEOUT = config['advanced']['request_timeout']
            self.SLEEP_BETWEEN_REQUESTS = config['advanced']['sleep_between_requests']
            self.HAS_OPRA = config['advanced'].get('has_opra', False)  # OPRA subscription for Greeks
            
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
        self.LOG_LEVEL = self.log_level
        
        self.MAX_STRIKES_PER_EXPIRY = 16
        self.REQUIRE_MODEL_GREEKS = True
        self.MIN_VOLUME = 0
        self.REQUEST_TIMEOUT = 5.0
        self.SLEEP_BETWEEN_REQUESTS = 0.25
        self.HAS_OPRA = False
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Convert string log level to logging constant
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/collar_screener.log'),
                logging.StreamHandler()
            ]
        )
        
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
            base_wait = 0.6        # start at 0.6s
            # Request a snapshot (quotes only)
            ticker = self.ib.reqMktData(opt, '', True, False)

            for i in range(max_tries):
                # wait progressively longer
                self.ib.sleep(base_wait + 0.3 * i)  # 0.6, 0.9, 1.2, ...
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
        """Pick options by moneyness (K/spot) when delta is not available"""
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
    
    def pick_by_delta(self, strikes: List[Tuple], target_range: Tuple[float, float], 
                      option_type: str = 'put', stock_price: float = None) -> List[Tuple]:
        """Pick options closest to target delta range, with fallback to moneyness"""
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
        
        # Fallback to moneyness-based selection if no delta candidates
        if no_delta_strikes and stock_price:
            self.logger.debug(f"Falling back to moneyness-based selection for {option_type}s (no delta data)")
            if option_type == 'put':
                # Target 0.65-0.95 moneyness for puts (15-35% OTM)
                return self.pick_by_moneyness(no_delta_strikes, stock_price, (0.65, 0.95), option_type)
            else:
                # Target 1.05-1.35 moneyness for calls (15-35% OTM)
                return self.pick_by_moneyness(no_delta_strikes, stock_price, (1.05, 1.35), option_type)
        
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
                
                # Get strikes around current stock price
                approx_strikes = sorted([float(s) for s in p.strikes if s and s != '0'])
                
                # If we have a valid stock price, filter strikes around it
                if stock_price and stock_price > 0 and not math.isnan(stock_price):
                    band = [K for K in approx_strikes if 0.5 * stock_price <= K <= 1.5 * stock_price]
                    # Sample strikes nearest to current price - reduce to speed up
                    band = sorted(band, key=lambda K: abs(K - stock_price))[:4]  # Reduced from 8 to 4
                else:
                    # Fallback: use strikes around the middle of the available range
                    if approx_strikes:
                        mid_strike = (min(approx_strikes) + max(approx_strikes)) / 2
                        band = sorted(approx_strikes, key=lambda K: abs(K - mid_strike))[:4]  # Reduced from 8 to 4
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
                    
                    put = resolve_option_smart(self.ib, symbol, expiry_ib, K, 'P', timeout=12.0)
                    call = resolve_option_smart(self.ib, symbol, expiry_ib, K, 'C', timeout=12.0)
                    
                    if not put or not call:
                        self.logger.debug(f"    Could not resolve/qualify {symbol} {K} {expiry} on SMART")
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
                
                # Skip this expiry if we have no candidates
                if len(put_candidates) == 0 or len(call_candidates) == 0:
                    self.logger.debug(f"Skipping {symbol} {expiry} - insufficient candidates (puts: {len(put_candidates)}, calls: {len(call_candidates)})")
                    continue
                
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
                        
                        if floor >= self.config.FLOOR_TOLERANCE:
                            filters_passed.append(f"floor (${floor:.2f}) >= tolerance (${self.config.FLOOR_TOLERANCE})")
                        else:
                            filters_failed.append(f"floor (${floor:.2f}) < tolerance (${self.config.FLOOR_TOLERANCE})")
                        
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
                            
                            # Scoring: prioritize zero-risk opportunities
                            score_risk_reward = max_gain / max(1.0, capital_at_risk + 1e-6)
                            score_gain_per_capital = max_gain / max(1.0, capital_used)
                            
                            self.logger.debug(f"  Final scores: risk_reward={score_risk_reward:.2f}, gain_per_capital={score_gain_per_capital:.2f}")
                            
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
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            self.logger.debug(f"FAILED {symbol} {expiry} - Some filters failed:")
                            for filter_msg in filters_failed:
                                self.logger.debug(f"  FAIL: {filter_msg}")
                            for filter_msg in filters_passed:
                                self.logger.debug(f"  PASS: {filter_msg}")
            
            # Rank results
            results.sort(key=lambda r: (-r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
            
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
    
    def print_results(self, results: List[Dict]):
        """Print results in a formatted table"""
        if not results:
            print("\nNo collar opportunities found matching criteria.")
            return
        
        print(f"\n{'='*120}")
        print(f"ZERO-COST COLLAR OPPORTUNITIES (Budget: ${self.config.CAPITAL_BUDGET:,})")
        print(f"{'='*120}")
        print(f"{'SYM':<6} {'EXP':<10} {'PX':>7} {'P_K':>6} {'P_mid':>7} {'P_delta':>7} {'C_K':>6} {'C_mid':>7} {'C_delta':>7} {'C0':>9} {'FLOOR':>8} {'MAXG':>8} {'DTE':>4} {'SCORE':>8}")
        print(f"{'-'*120}")
        
        for r in results[:self.config.MAX_RESULTS]:
            put_delta_str = f"{r['put_delta']:.2f}" if r['put_delta'] is not None else "N/A"
            call_delta_str = f"{r['call_delta']:.2f}" if r['call_delta'] is not None else "N/A"
            
            print(f"{r['symbol']:<6} {r['expiry']:<10} {r['stock_mid']:>7.2f} "
                  f"{r['put_K']:>6.2f} {r['put_mid']:>7.2f} {put_delta_str:>5} "
                  f"{r['call_K']:>6.2f} {r['call_mid']:>7.2f} {call_delta_str:>5} "
                  f"{r['net_cost_C0']:>9.2f} {r['floor_$']:>8.2f} {r['max_gain_$']:>8.2f} "
                  f"{r['dte']:>4} {r['score_risk_reward']:>8.2f}")
        
        print(f"\nFound {len(results)} opportunities, showing top {min(self.config.MAX_RESULTS, len(results))}")
        print(f"Legend: PX=Stock Price, P_K/C_K=Put/Call Strike, P_mid/C_mid=Put/Call Mid Price")
        print(f"        P_delta/C_delta=Put/Call Delta, C0=Net Cost, FLOOR=Loss Floor, MAXG=Max Gain")
        print(f"        DTE=Days to Expiry, SCORE=Risk-Reward Score")
    
    def save_to_csv(self, results: List[Dict]):
        """Save results to CSV file"""
        if not self.config.SAVE_TO_CSV or not results:
            return
        
        try:
            import pandas as pd
            
            # Ensure reports directory exists
            os.makedirs('reports', exist_ok=True)
            
            df = pd.DataFrame(results)
            df.to_csv(self.config.CSV_FILENAME, index=False)
            self.logger.info(f"Results saved to {self.config.CSV_FILENAME}")
            
        except ImportError:
            self.logger.warning("pandas not available, skipping CSV export")
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
    
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
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Sort all results
            all_results.sort(key=lambda r: (-r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
            
            self.logger.info(f"Scan complete! Found {len(all_results)} total opportunities across {total_symbols} symbols")
            
            # Print and save results
            self.print_results(all_results)
            self.save_to_csv(all_results)
            
        finally:
            # Always disconnect
            self.disconnect_ib()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Handle verbose flag
    if args.verbose:
        args.log_level = 'DEBUG'
    
    config = CollarScreenerConfig(args.config, args.log_level)
    screener = CollarScreener(config)
    screener.run()

if __name__ == '__main__':
    main()
