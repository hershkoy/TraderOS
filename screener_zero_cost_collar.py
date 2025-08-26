#!/usr/bin/env python3
"""
Zero-Cost Collar Screener for Interactive Brokers

This screener finds near-zero-cost collar opportunities by:
1. Buying 100 shares of stock
2. Buying a protective put (delta ~-0.35 to -0.20)
3. Selling a covered call (delta ~+0.20 to +0.35)

The goal is to find setups where: Net Cost (shares + put - call) ≤ 100 × put strike
This creates a "zero-loss" scenario where max loss at expiry is ≤ $0.

Author: AI Assistant
Date: 2025
"""

import sys
import os
import math
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collar_screener.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
class ScreenerConfig:
    # IB Connection
    IB_HOST = '127.0.0.1'
    IB_PORT = 7497  # 7497 paper / 7496 live
    IB_CLIENT_ID = 19
    
    # Capital and Budget
    CAPITAL_BUDGET = 30_000  # Default budget in USD
    MIN_CAPITAL_USED = 5_000  # Minimum position size to consider
    
    # Option Parameters
    MIN_DTE, MAX_DTE = 30, 60  # Days to expiry range
    PUT_DELTA_RANGE = (-0.35, -0.20)  # Target put delta range
    CALL_DELTA_RANGE = (0.20, 0.35)   # Target call delta range
    
    # Risk Tolerance
    FLOOR_TOLERANCE = -25.0  # Allow small negative floor (e.g., -$25)
    MAX_SPREAD_PCT = 0.15    # Maximum bid-ask spread as % of mid price
    
    # Universe (configurable)
    UNIVERSE = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'NFLX',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'
    ]
    
    # Output Settings
    MAX_RESULTS = 25
    SAVE_TO_CSV = True
    CSV_FILENAME = 'reports/collar_opportunities.csv'

def mid(bid: float, ask: float) -> Optional[float]:
    """Calculate mid price from bid/ask"""
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0

def in_range(x: float, lo: float, hi: float) -> bool:
    """Check if value is within range"""
    return x is not None and lo <= x <= hi

def dte(expiry_str: str) -> int:
    """Calculate days to expiry"""
    try:
        if '-' not in expiry_str:
            # Format: '20251017'
            d = datetime.strptime(expiry_str, '%Y%m%d')
        else:
            # Format: '2025-10-17'
            d = datetime.strptime(expiry_str, '%Y-%m-%d')
        return (d.date() - datetime.now().date()).days
    except ValueError as e:
        logger.warning(f"Invalid expiry format: {expiry_str}, error: {e}")
        return 0

def option_mid_price(ib: IB, opt: Contract) -> Optional[float]:
    """Get option mid price with retry logic"""
    try:
        ticker = ib.reqMktData(opt, '', True, False)
        ib.sleep(0.25)  # Wait for data
        
        if ticker.bid and ticker.ask:
            return mid(ticker.bid, ticker.ask)
        elif ticker.last:
            return ticker.last
        else:
            logger.warning(f"No price data for {opt.symbol} {opt.strike} {opt.right}")
            return None
    except Exception as e:
        logger.error(f"Error getting option price for {opt.symbol}: {e}")
        return None

def stock_mid_price(ib: IB, stk: Contract) -> Optional[float]:
    """Get stock mid price"""
    try:
        ticker = ib.reqMktData(stk, '', True, False)
        ib.sleep(0.15)
        
        if ticker.bid and ticker.ask:
            return mid(ticker.bid, ticker.ask)
        elif ticker.last:
            return ticker.last
        else:
            logger.warning(f"No price data for stock {stk.symbol}")
            return None
    except Exception as e:
        logger.error(f"Error getting stock price for {stk.symbol}: {e}")
        return None

def option_model_delta(ib: IB, opt: Contract) -> Optional[float]:
    """Get option delta using IB's model"""
    try:
        ticker = ib.reqMktData(opt, '106', True, False)  # 106 = option computation
        ib.sleep(0.25)
        
        if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
            return ticker.modelGreeks.delta
        else:
            logger.warning(f"No delta data for {opt.symbol} {opt.strike} {opt.right}")
            return None
    except Exception as e:
        logger.error(f"Error getting delta for {opt.symbol}: {e}")
        return None

def pick_by_delta(strikes: List[Tuple], target_range: Tuple[float, float], 
                  option_type: str = 'put') -> List[Tuple]:
    """Pick options closest to target delta range"""
    candidates = []
    for contract, delta, mid_price, strike in strikes:
        if delta is None or mid_price is None:
            continue
            
        # For puts, deltas are negative; for calls, positive
        if option_type == 'put' and in_range(delta, target_range[0], target_range[1]):
            candidates.append((contract, delta, mid_price, strike))
        elif option_type == 'call' and in_range(delta, target_range[0], target_range[1]):
            candidates.append((contract, delta, mid_price, strike))
    
    # Sort by distance from center of target range
    center = sum(target_range) / 2
    candidates.sort(key=lambda x: abs(x[1] - center))
    
    return candidates[:4]  # Keep top 4 candidates

def calculate_spread_percentage(bid: float, ask: float) -> float:
    """Calculate bid-ask spread as percentage of mid price"""
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return float('inf')
    mid_price = (bid + ask) / 2
    return (ask - bid) / mid_price

def scan_ticker(ib: IB, symbol: str, config: ScreenerConfig) -> List[Dict]:
    """Scan a single ticker for collar opportunities"""
    logger.info(f"Scanning {symbol} for collar opportunities...")
    
    try:
        # Create stock contract
        stk = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(stk)
        
        # Get stock price
        stock_price = stock_mid_price(ib, stk)
        if not stock_price:
            logger.warning(f"Could not get price for {symbol}")
            return []
        
        # Get option parameters
        params = ib.reqSecDefOptParams(stk.symbol, '', stk.secType, stk.conId)
        if not params:
            logger.warning(f"No option parameters for {symbol}")
            return []
        
        p = params[0]
        
        # Process expirations
        expiries = []
        for e in p.expirations:
            if '-' not in e:
                # Convert '20251017' to '2025-10-17'
                expiries.append(f"{e[:4]}-{e[4:6]}-{e[6:]}")
            else:
                expiries.append(e)
        
        # Filter by DTE
        expiries = [e for e in expiries if in_range(dte(e), config.MIN_DTE, config.MAX_DTE)]
        
        if not expiries:
            logger.info(f"No suitable expirations for {symbol}")
            return []
        
        results = []
        
        for expiry in sorted(expiries, key=lambda e: dte(e)):
            logger.debug(f"Processing {symbol} expiry {expiry}")
            
            # Get strikes around current stock price
            approx_strikes = sorted([float(s) for s in p.strikes if s and s != '0'])
            band = [K for K in approx_strikes if 0.5 * stock_price <= K <= 1.5 * stock_price]
            
            # Sample strikes nearest to current price for efficiency
            band = sorted(band, key=lambda K: abs(K - stock_price))[:16]
            
            # Fetch option data
            temp_puts, temp_calls = [], []
            
            for K in band:
                put = Option(symbol, expiry, K, 'P', 'SMART', tradingClass=p.tradingClass)
                call = Option(symbol, expiry, K, 'C', 'SMART', tradingClass=p.tradingClass)
                
                for contract in (put, call):
                    try:
                        ib.qualifyContracts(contract)
                    except Exception as e:
                        logger.debug(f"Could not qualify {contract.symbol} {K} {contract.right}: {e}")
                        continue
                
                # Get put data
                p_mid = option_mid_price(ib, put)
                p_delta = option_model_delta(ib, put)
                if p_mid and p_delta:
                    temp_puts.append((put, p_delta, p_mid, K))
                
                # Get call data
                c_mid = option_mid_price(ib, call)
                c_delta = option_model_delta(ib, call)
                if c_mid and c_delta:
                    temp_calls.append((call, c_delta, c_mid, K))
            
            # Pick candidates by delta
            put_candidates = pick_by_delta(temp_puts, config.PUT_DELTA_RANGE, 'put')
            call_candidates = pick_by_delta(temp_calls, config.CALL_DELTA_RANGE, 'call')
            
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
                    put_spread = calculate_spread_percentage(put.bid, put.ask) if hasattr(put, 'bid') else 0
                    call_spread = calculate_spread_percentage(call.bid, call.ask) if hasattr(call, 'bid') else 0
                    
                    # Apply filters
                    if (capital_used <= config.CAPITAL_BUDGET and 
                        capital_used >= config.MIN_CAPITAL_USED and
                        floor >= config.FLOOR_TOLERANCE and
                        put_spread <= config.MAX_SPREAD_PCT and
                        call_spread <= config.MAX_SPREAD_PCT):
                        
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
                            'dte': dte(expiry),
                            'score_risk_reward': round(score_risk_reward, 2),
                            'score_gain_per_capital': round(score_gain_per_capital, 2),
                            'timestamp': datetime.now().isoformat()
                        })
        
        # Rank results
        results.sort(key=lambda r: (-r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
        return results
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return []

def print_results(results: List[Dict], config: ScreenerConfig):
    """Print results in a formatted table"""
    if not results:
        print("\nNo collar opportunities found matching criteria.")
        return
    
    print(f"\n{'='*120}")
    print(f"ZERO-COST COLLAR OPPORTUNITIES (Budget: ${config.CAPITAL_BUDGET:,})")
    print(f"{'='*120}")
    print(f"{'SYM':<6} {'EXP':<10} {'PX':>7} {'P_K':>6} {'P_mid':>7} {'P_δ':>5} {'C_K':>6} {'C_mid':>7} {'C_δ':>5} {'C0':>9} {'FLOOR':>8} {'MAXG':>8} {'DTE':>4} {'SCORE':>8}")
    print(f"{'-'*120}")
    
    for r in results[:config.MAX_RESULTS]:
        print(f"{r['symbol']:<6} {r['expiry']:<10} {r['stock_mid']:>7.2f} "
              f"{r['put_K']:>6.2f} {r['put_mid']:>7.2f} {r['put_delta']:>5.2f} "
              f"{r['call_K']:>6.2f} {r['call_mid']:>7.2f} {r['call_delta']:>5.2f} "
              f"{r['net_cost_C0']:>9.2f} {r['floor_$']:>8.2f} {r['max_gain_$']:>8.2f} "
              f"{r['dte']:>4} {r['score_risk_reward']:>8.2f}")
    
    print(f"\nFound {len(results)} opportunities, showing top {min(config.MAX_RESULTS, len(results))}")
    print(f"Legend: PX=Stock Price, P_K/C_K=Put/Call Strike, P_mid/C_mid=Put/Call Mid Price")
    print(f"        P_δ/C_δ=Put/Call Delta, C0=Net Cost, FLOOR=Loss Floor, MAXG=Max Gain")
    print(f"        DTE=Days to Expiry, SCORE=Risk-Reward Score")

def save_to_csv(results: List[Dict], config: ScreenerConfig):
    """Save results to CSV file"""
    if not config.SAVE_TO_CSV or not results:
        return
    
    try:
        import pandas as pd
        
        # Ensure reports directory exists
        os.makedirs('reports', exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(config.CSV_FILENAME, index=False)
        logger.info(f"Results saved to {config.CSV_FILENAME}")
        
    except ImportError:
        logger.warning("pandas not available, skipping CSV export")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")

def main():
    """Main screener function"""
    config = ScreenerConfig()
    
    print("Zero-Cost Collar Screener")
    print("=" * 50)
    print(f"Budget: ${config.CAPITAL_BUDGET:,}")
    print(f"DTE Range: {config.MIN_DTE}-{config.MAX_DTE} days")
    print(f"Put Delta: {config.PUT_DELTA_RANGE}")
    print(f"Call Delta: {config.CALL_DELTA_RANGE}")
    print(f"Universe: {len(config.UNIVERSE)} symbols")
    print()
    
    # Connect to IB
    ib = IB()
    try:
        ib.connect(config.IB_HOST, config.IB_PORT, clientId=config.IB_CLIENT_ID, readonly=True)
        logger.info("Connected to Interactive Brokers")
    except Exception as e:
        logger.error(f"Failed to connect to IB: {e}")
        print("Error: Could not connect to Interactive Brokers. Make sure TWS/IBG is running.")
        return
    
    # Scan all symbols
    all_results = []
    for symbol in config.UNIVERSE:
        try:
            results = scan_ticker(ib, symbol, config)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    # Sort all results
    all_results.sort(key=lambda r: (-r['score_risk_reward'], -r['score_gain_per_capital'], r['dte']))
    
    # Print and save results
    print_results(all_results, config)
    save_to_csv(all_results, config)
    
    # Disconnect
    ib.disconnect()
    logger.info("Disconnected from Interactive Brokers")

if __name__ == '__main__':
    main()
