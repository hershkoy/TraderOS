#!/usr/bin/env python3
"""
Enhanced Zero-Cost Collar Screener for Interactive Brokers

This enhanced screener reads configuration from YAML and provides additional features:
- Configurable parameters via YAML file
- Better error handling and logging
- CSV export with detailed metrics
- Integration with existing backtrader project structure

Author: AI Assistant
Date: 2025
"""

import sys
import os
import math
import yaml
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

class CollarScreenerConfig:
    """Configuration class for the collar screener"""
    
    def __init__(self, config_file: str = "collar_screener_config.yaml"):
        self.config_file = config_file
        self.load_config()
        self.setup_logging()
    
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
            
            # Universe
            self.UNIVERSE = config['universe']
            
            # Output Settings
            self.MAX_RESULTS = config['output']['max_results']
            self.SAVE_TO_CSV = config['output']['save_to_csv']
            self.CSV_FILENAME = config['output']['csv_filename']
            self.LOG_LEVEL = config['output']['log_level']
            
            # Advanced Settings
            self.MAX_STRIKES_PER_EXPIRY = config['advanced']['max_strikes_per_expiry']
            self.REQUIRE_MODEL_GREEKS = config['advanced']['require_model_greeks']
            self.MIN_VOLUME = config['advanced']['min_volume']
            self.REQUEST_TIMEOUT = config['advanced']['request_timeout']
            self.SLEEP_BETWEEN_REQUESTS = config['advanced']['sleep_between_requests']
            
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found. Using default configuration.")
            self._set_defaults()
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            self._set_defaults()
    
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
        self.LOG_LEVEL = 'INFO'
        
        self.MAX_STRIKES_PER_EXPIRY = 16
        self.REQUIRE_MODEL_GREEKS = True
        self.MIN_VOLUME = 0
        self.REQUEST_TIMEOUT = 5.0
        self.SLEEP_BETWEEN_REQUESTS = 0.25
    
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
                self.logger.info("Connected to Interactive Brokers using shared connection")
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
    
    def option_mid_price(self, opt: Contract) -> Optional[float]:
        """Get option mid price with retry logic"""
        try:
            ticker = self.ib.reqMktData(opt, '', True, False)
            self.ib.sleep(self.config.SLEEP_BETWEEN_REQUESTS)
            
            if ticker.bid and ticker.ask:
                return self.mid(ticker.bid, ticker.ask)
            elif ticker.last:
                return ticker.last
            else:
                self.logger.warning(f"No price data for {opt.symbol} {opt.strike} {opt.right}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting option price for {opt.symbol}: {e}")
            return None
    
    def stock_mid_price(self, stk: Contract) -> Optional[float]:
        """Get stock mid price"""
        try:
            ticker = self.ib.reqMktData(stk, '', True, False)
            self.ib.sleep(self.config.SLEEP_BETWEEN_REQUESTS)
            
            if ticker.bid and ticker.ask:
                return self.mid(ticker.bid, ticker.ask)
            elif ticker.last:
                return ticker.last
            else:
                self.logger.warning(f"No price data for stock {stk.symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting stock price for {stk.symbol}: {e}")
            return None
    
    def option_model_delta(self, opt: Contract) -> Optional[float]:
        """Get option delta using IB's model"""
        try:
            ticker = self.ib.reqMktData(opt, '106', True, False)
            self.ib.sleep(self.config.SLEEP_BETWEEN_REQUESTS)
            
            if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
                return ticker.modelGreeks.delta
            else:
                self.logger.warning(f"No delta data for {opt.symbol} {opt.strike} {opt.right}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting delta for {opt.symbol}: {e}")
            return None
    
    def pick_by_delta(self, strikes: List[Tuple], target_range: Tuple[float, float], 
                     option_type: str = 'put') -> List[Tuple]:
        """Pick options closest to target delta range"""
        candidates = []
        for contract, delta, mid_price, strike in strikes:
            if delta is None or mid_price is None:
                continue
                
            if option_type == 'put' and self.in_range(delta, target_range[0], target_range[1]):
                candidates.append((contract, delta, mid_price, strike))
            elif option_type == 'call' and self.in_range(delta, target_range[0], target_range[1]):
                candidates.append((contract, delta, mid_price, strike))
        
        # Sort by distance from center of target range
        center = sum(target_range) / 2
        candidates.sort(key=lambda x: abs(x[1] - center))
        
        return candidates[:4]
    
    def calculate_spread_percentage(self, bid: float, ask: float) -> float:
        """Calculate bid-ask spread as percentage of mid price"""
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return float('inf')
        mid_price = (bid + ask) / 2
        return (ask - bid) / mid_price
    
    def scan_ticker(self, symbol: str) -> List[Dict]:
        """Scan a single ticker for collar opportunities"""
        self.logger.info(f"Scanning {symbol} for collar opportunities...")
        
        try:
            # Create stock contract
            stk = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(stk)
            
            # Get stock price
            stock_price = self.stock_mid_price(stk)
            if not stock_price:
                self.logger.warning(f"Could not get price for {symbol}")
                return []
            
            # Get option parameters
            params = self.ib.reqSecDefOptParams(stk.symbol, '', stk.secType, stk.conId)
            if not params:
                self.logger.warning(f"No option parameters for {symbol}")
                return []
            
            p = params[0]
            
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
                self.logger.info(f"No suitable expirations for {symbol}")
                return []
            
            results = []
            
            for expiry in sorted(expiries, key=lambda e: self.dte(e)):
                self.logger.debug(f"Processing {symbol} expiry {expiry}")
                
                # Get strikes around current stock price
                approx_strikes = sorted([float(s) for s in p.strikes if s and s != '0'])
                band = [K for K in approx_strikes if 0.5 * stock_price <= K <= 1.5 * stock_price]
                
                # Sample strikes nearest to current price
                band = sorted(band, key=lambda K: abs(K - stock_price))[:self.config.MAX_STRIKES_PER_EXPIRY]
                
                # Fetch option data
                temp_puts, temp_calls = [], []
                
                for K in band:
                    put = Option(symbol, expiry, K, 'P', 'SMART', tradingClass=p.tradingClass)
                    call = Option(symbol, expiry, K, 'C', 'SMART', tradingClass=p.tradingClass)
                    
                    for contract in (put, call):
                        try:
                            self.ib.qualifyContracts(contract)
                        except Exception as e:
                            self.logger.debug(f"Could not qualify {contract.symbol} {K} {contract.right}: {e}")
                            continue
                    
                    # Get put data
                    p_mid = self.option_mid_price(put)
                    p_delta = self.option_model_delta(put)
                    if p_mid and p_delta:
                        temp_puts.append((put, p_delta, p_mid, K))
                    
                    # Get call data
                    c_mid = self.option_mid_price(call)
                    c_delta = self.option_model_delta(call)
                    if c_mid and c_delta:
                        temp_calls.append((call, c_delta, c_mid, K))
                
                # Pick candidates by delta
                put_candidates = self.pick_by_delta(temp_puts, self.config.PUT_DELTA_RANGE, 'put')
                call_candidates = self.pick_by_delta(temp_calls, self.config.CALL_DELTA_RANGE, 'call')
                
                # Pair puts and calls
                for pc in put_candidates:
                    put, p_delta, p_mid, Kp = pc
                    
                    for cc in call_candidates:
                        call, c_delta, c_mid, Kc = cc
                        
                        # Only meaningful if call strike >= put strike
                        if Kc < Kp:
                            continue
                        
                        # Calculate collar metrics
                        C0 = 100 * stock_price + p_mid - c_mid  # Net cost
                        floor = 100 * Kp - C0                   # Loss floor
                        max_gain = 100 * Kc - C0                # Max gain if called away
                        capital_used = C0
                        
                        if capital_used is None or max_gain is None:
                            continue
                        
                        # Check spread quality
                        put_spread = self.calculate_spread_percentage(put.bid, put.ask) if hasattr(put, 'bid') else 0
                        call_spread = self.calculate_spread_percentage(call.bid, call.ask) if hasattr(call, 'bid') else 0
                        
                        # Apply filters
                        if (capital_used <= self.config.CAPITAL_BUDGET and 
                            capital_used >= self.config.MIN_CAPITAL_USED and
                            floor >= self.config.FLOOR_TOLERANCE and
                            put_spread <= self.config.MAX_SPREAD_PCT and
                            call_spread <= self.config.MAX_SPREAD_PCT):
                            
                            capital_at_risk = max(0, C0 - 100 * Kp)
                            
                            # Scoring: prioritize zero-risk opportunities
                            score_risk_reward = max_gain / max(1.0, capital_at_risk + 1e-6)
                            score_gain_per_capital = max_gain / max(1.0, capital_used)
                            
                            results.append({
                                'symbol': symbol,
                                'expiry': expiry,
                                'stock_mid': round(stock_price, 4),
                                'put_K': Kp,
                                'put_mid': round(p_mid, 4),
                                'put_delta': round(p_delta, 3),
                                'put_spread_pct': round(put_spread * 100, 2),
                                'call_K': Kc,
                                'call_mid': round(c_mid, 4),
                                'call_delta': round(c_delta, 3),
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
            
            # Rank results
            results.sort(key=lambda r: (-r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
            return results
            
        except Exception as e:
            self.logger.error(f"Error scanning {symbol}: {e}")
            return []
    
    def print_results(self, results: List[Dict]):
        """Print results in a formatted table"""
        if not results:
            print("\nNo collar opportunities found matching criteria.")
            return
        
        print(f"\n{'='*120}")
        print(f"ZERO-COST COLLAR OPPORTUNITIES (Budget: ${self.config.CAPITAL_BUDGET:,})")
        print(f"{'='*120}")
        print(f"{'SYM':<6} {'EXP':<10} {'PX':>7} {'P_K':>6} {'P_mid':>7} {'P_δ':>5} {'C_K':>6} {'C_mid':>7} {'C_δ':>5} {'C0':>9} {'FLOOR':>8} {'MAXG':>8} {'DTE':>4} {'SCORE':>8}")
        print(f"{'-'*120}")
        
        for r in results[:self.config.MAX_RESULTS]:
            print(f"{r['symbol']:<6} {r['expiry']:<10} {r['stock_mid']:>7.2f} "
                  f"{r['put_K']:>6.2f} {r['put_mid']:>7.2f} {r['put_delta']:>5.2f} "
                  f"{r['call_K']:>6.2f} {r['call_mid']:>7.2f} {r['call_delta']:>5.2f} "
                  f"{r['net_cost_C0']:>9.2f} {r['floor_$']:>8.2f} {r['max_gain_$']:>8.2f} "
                  f"{r['dte']:>4} {r['score_risk_reward']:>8.2f}")
        
        print(f"\nFound {len(results)} opportunities, showing top {min(self.config.MAX_RESULTS, len(results))}")
        print(f"Legend: PX=Stock Price, P_K/C_K=Put/Call Strike, P_mid/C_mid=Put/Call Mid Price")
        print(f"        P_δ/C_δ=Put/Call Delta, C0=Net Cost, FLOOR=Loss Floor, MAXG=Max Gain")
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
        print()
        
        # Connect to IB
        if not self.connect_ib():
            return
        
        try:
            # Scan all symbols
            all_results = []
            for symbol in self.config.UNIVERSE:
                try:
                    results = self.scan_ticker(symbol)
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Sort all results
            all_results.sort(key=lambda r: (-r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
            
            # Print and save results
            self.print_results(all_results)
            self.save_to_csv(all_results)
            
        finally:
            # Always disconnect
            self.disconnect_ib()

def main():
    """Main function"""
    config = CollarScreenerConfig()
    screener = CollarScreener(config)
    screener.run()

if __name__ == '__main__':
    main()
