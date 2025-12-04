#!/usr/bin/env python3
"""
Generate mock options data for testing LEAPS strategy
Creates realistic simulated options data without requiring paid API access
"""

import os
import sys
import logging
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.options.option_utils import build_option_id
from utils.db.pg_copy import copy_rows_with_upsert

# Configure logging
os.makedirs('logs/data/mock', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data/mock/mock_options_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MockOptionsDataGenerator:
    """Generates realistic mock options data for testing"""
    
    def __init__(self, db_connection_string: str = None):
        """
        Initialize the mock data generator
        
        Args:
            db_connection_string: PostgreSQL connection string
        """
        self.db_connection_string = db_connection_string or self._get_default_connection_string()
        
        # Base parameters for realistic data
        self.base_price = 350.0  # QQQ base price
        self.volatility = 0.25   # 25% volatility
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
    def _get_default_connection_string(self) -> str:
        """Get default database connection string"""
        host = os.getenv('TIMESCALEDB_HOST', 'localhost')
        port = os.getenv('TIMESCALEDB_PORT', '5432')
        database = os.getenv('TIMESCALEDB_DATABASE', 'backtrader')
        user = os.getenv('TIMESCALEDB_USER', 'backtrader_user')
        password = os.getenv('TIMESCALEDB_PASSWORD', 'backtrader_password')
        
        return f"host={host} port={port} dbname={database} user={user} password={password}"
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_connection_string)
    
    def generate_strike_prices(self, base_price: float, num_strikes: int = 20) -> List[float]:
        """Generate realistic strike prices around the base price"""
        strikes = []
        # Generate strikes from -20% to +20% of base price
        min_strike = base_price * 0.8
        max_strike = base_price * 1.2
        
        for i in range(num_strikes):
            strike = min_strike + (max_strike - min_strike) * i / (num_strikes - 1)
            strikes.append(round(strike, 2))
        
        return strikes
    
    def calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'C') -> Dict[str, float]:
        """
        Calculate Black-Scholes option prices and Greeks
        Simplified implementation for mock data
        """
        import math
        
        # Simplified Black-Scholes calculation
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'C':
            # Call option
            price = S * self._normal_cdf(d1) - K * math.exp(-r * T) * self._normal_cdf(d2)
            delta = self._normal_cdf(d1)
            gamma = self._normal_pdf(d1) / (S * sigma * math.sqrt(T))
            theta = (-S * self._normal_pdf(d1) * sigma / (2 * math.sqrt(T)) - 
                    r * K * math.exp(-r * T) * self._normal_cdf(d2))
            vega = S * math.sqrt(T) * self._normal_pdf(d1)
        else:
            # Put option
            price = K * math.exp(-r * T) * self._normal_cdf(-d2) - S * self._normal_cdf(-d1)
            delta = self._normal_cdf(d1) - 1
            gamma = self._normal_pdf(d1) / (S * sigma * math.sqrt(T))
            theta = (-S * self._normal_pdf(d1) * sigma / (2 * math.sqrt(T)) + 
                    r * K * math.exp(-r * T) * self._normal_cdf(-d2))
            vega = S * math.sqrt(T) * self._normal_pdf(d1)
        
        return {
            'price': max(price, 0.01),  # Minimum price of $0.01
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'iv': sigma
        }
    
    def _normal_cdf(self, x: float) -> float:
        """Normal cumulative distribution function (simplified)"""
        import math
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Normal probability density function"""
        import math
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    
    def generate_contracts(self, underlying: str, expiration_date: date, option_right: str = 'C') -> List[Dict[str, Any]]:
        """
        Generate mock option contracts for a specific expiration
        
        Args:
            underlying: Underlying symbol
            expiration_date: Expiration date
            option_right: Option type ('C' for call, 'P' for put)
        
        Returns:
            List of contract dictionaries
        """
        contracts = []
        current_date = date.today()
        days_to_expiry = (expiration_date - current_date).days
        
        if days_to_expiry <= 0:
            return contracts
        
        # Generate strike prices
        strikes = self.generate_strike_prices(self.base_price)
        
        for strike in strikes:
            # Calculate time to expiry in years
            T = days_to_expiry / 365.0
            
            # Calculate Greeks using Black-Scholes
            greeks = self.calculate_black_scholes_greeks(
                S=self.base_price,
                K=strike,
                T=T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                option_type=option_right
            )
            
            # Build option ID
            option_id = build_option_id(underlying, expiration_date, strike, option_right)
            
            # Create contract data
            contract = {
                'option_id': option_id,
                'underlying': underlying,
                'expiration': expiration_date.strftime('%Y-%m-%d'),
                'strike_cents': int(strike * 100),
                'option_right': option_right,
                'multiplier': 100,
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'polygon_ticker': f"O:{underlying}{expiration_date.strftime('%y%m%d')}{option_right}{int(strike*1000):08d}"  # Mock Polygon ticker
            }
            
            contracts.append(contract)
        
        return contracts
    
    def generate_quotes(self, contracts: List[Dict[str, Any]], trade_date: date) -> List[Dict[str, Any]]:
        """
        Generate mock quotes for contracts
        
        Args:
            contracts: List of contracts
            trade_date: Trading date
        
        Returns:
            List of quote dictionaries
        """
        quotes = []
        
        for contract in contracts:
            # Parse contract details
            underlying = contract['underlying']
            expiration = datetime.strptime(contract['expiration'], '%Y-%m-%d').date()
            strike = contract['strike_cents'] / 100.0
            option_right = contract['option_right']
            option_id = contract['option_id']
            
            # Calculate time to expiry
            days_to_expiry = (expiration - trade_date).days
            if days_to_expiry <= 0:
                continue
            
            T = days_to_expiry / 365.0
            
            # Calculate theoretical price and Greeks
            greeks = self.calculate_black_scholes_greeks(
                S=self.base_price,
                K=strike,
                T=T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                option_type=option_right
            )
            
            # Add some realistic noise to prices
            price_noise = random.uniform(-0.05, 0.05)  # ±5% noise
            theoretical_price = greeks['price'] * (1 + price_noise)
            
            # Calculate bid/ask spread (typically 1-5% for liquid options)
            spread_pct = random.uniform(0.01, 0.05)
            mid_price = max(theoretical_price, 0.01)
            spread = mid_price * spread_pct
            
            bid = max(mid_price - spread/2, 0.01)
            ask = mid_price + spread/2
            
            # Generate realistic volume and open interest
            volume = random.randint(10, 1000)
            open_interest = random.randint(100, 10000)
            
            # Add some noise to Greeks
            delta_noise = random.uniform(-0.02, 0.02)
            iv_noise = random.uniform(-0.02, 0.02)
            
            quote = {
                'ts': trade_date.strftime('%Y-%m-%d 16:00:00'),
                'option_id': option_id,
                'bid': round(bid, 2),
                'ask': round(ask, 2),
                'last': round(mid_price, 2),
                'volume': volume,
                'open_interest': open_interest,
                'iv': max(0, greeks['iv'] + iv_noise),
                'delta': max(-1, min(1, greeks['delta'] + delta_noise)),
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'snapshot_type': 'eod'
            }
            
            quotes.append(quote)
        
        return quotes
    
    def generate_data_for_date_range(self, underlying: str, start_date: date, end_date: date, 
                                    sample_rate: int = 5) -> Dict[str, Any]:
        """
        Generate mock options data for a date range
        
        Args:
            underlying: Underlying symbol
            start_date: Start date
            end_date: End date
            sample_rate: Generate data every Nth day
        
        Returns:
            Summary of generated data
        """
        logger.info(f"Generating mock options data for {underlying} from {start_date} to {end_date}")
        
        # Generate expiration dates (monthly expirations)
        expiration_dates = []
        current_date = start_date
        while current_date <= end_date + timedelta(days=365):  # Look ahead 1 year
            # Third Friday of each month (simplified)
            expiration_date = current_date.replace(day=15)  # Approximate
            expiration_dates.append(expiration_date)
            current_date += timedelta(days=30)
        
        # Generate trading dates
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Sample trading dates
        sampled_dates = trading_dates[::sample_rate]
        
        total_contracts = 0
        total_quotes = 0
        
        with self.get_connection() as conn:
            for trade_date in sampled_dates:
                logger.info(f"Generating data for {trade_date}")
                
                # Generate contracts for each expiration date
                for expiration_date in expiration_dates:
                    if expiration_date > trade_date:
                        # Generate call contracts
                        call_contracts = self.generate_contracts(underlying, expiration_date, 'C')
                        
                        # Insert contracts
                        if call_contracts:
                            self.upsert_contracts(conn, call_contracts)
                            total_contracts += len(call_contracts)
                            
                            # Generate quotes for contracts
                            quotes = self.generate_quotes(call_contracts, trade_date)
                            if quotes:
                                self.upsert_quotes(conn, quotes)
                                total_quotes += len(quotes)
                        
                        # Generate put contracts (fewer for testing)
                        if random.random() < 0.3:  # 30% chance to include puts
                            put_contracts = self.generate_contracts(underlying, expiration_date, 'P')
                            if put_contracts:
                                self.upsert_contracts(conn, put_contracts)
                                total_contracts += len(put_contracts)
                                
                                put_quotes = self.generate_quotes(put_contracts, trade_date)
                                if put_quotes:
                                    self.upsert_quotes(conn, put_quotes)
                                    total_quotes += len(put_quotes)
        
        summary = {
            'total_contracts': total_contracts,
            'total_quotes': total_quotes,
            'trading_dates_processed': len(sampled_dates),
            'expiration_dates': len(expiration_dates),
            'underlying': underlying,
            'date_range': f"{start_date} to {end_date}"
        }
        
        logger.info(f"Mock data generation completed. Summary: {summary}")
        return summary
    
    def upsert_contracts(self, conn, contracts: List[Dict[str, Any]]) -> int:
        """Upsert contracts into database"""
        if not contracts:
            return 0
        
        contract_columns = ['option_id', 'underlying', 'expiration', 'strike_cents', 
                           'option_right', 'multiplier', 'first_seen', 'last_seen', 'polygon_ticker']
        
        # Convert datetime objects to strings for COPY
        for contract in contracts:
            contract['first_seen'] = contract['first_seen'].strftime('%Y-%m-%d %H:%M:%S')
            contract['last_seen'] = contract['last_seen'].strftime('%Y-%m-%d %H:%M:%S')
        
        return copy_rows_with_upsert(
            conn, 'option_contracts', contract_columns, iter(contracts),
            conflict_columns=['option_id'],
            update_columns=['last_seen', 'underlying', 'expiration', 'strike_cents', 
                           'option_right', 'multiplier', 'polygon_ticker']
        )
    
    def upsert_quotes(self, conn, quotes: List[Dict[str, Any]]) -> int:
        """Upsert quotes into database"""
        if not quotes:
            return 0
        
        quote_columns = ['ts', 'option_id', 'bid', 'ask', 'last', 'volume', 'open_interest',
                        'iv', 'delta', 'gamma', 'theta', 'vega', 'snapshot_type']
        
        return copy_rows_with_upsert(
            conn, 'option_quotes', quote_columns, iter(quotes),
            conflict_columns=['ts', 'option_id'],
            update_columns=['bid', 'ask', 'last', 'volume', 'open_interest',
                           'iv', 'delta', 'gamma', 'theta', 'vega', 'snapshot_type']
        )


def main():
    """Main function to generate mock options data"""
    try:
        logger.info("Starting mock options data generation")
        
        generator = MockOptionsDataGenerator()
        
        # Generate data for the last 6 months
        end_date = date.today()
        start_date = end_date - timedelta(days=180)
        
        summary = generator.generate_data_for_date_range(
            underlying='QQQ',
            start_date=start_date,
            end_date=end_date,
            sample_rate=5  # Every 5th trading day
        )
        
        logger.info(f"Mock data generation completed successfully")
        logger.info(f"Summary: {summary}")
        
        print("\n" + "="*60)
        print("MOCK OPTIONS DATA GENERATION SUMMARY")
        print("="*60)
        print(f"Underlying: {summary['underlying']}")
        print(f"Date Range: {summary['date_range']}")
        print(f"Trading Dates Processed: {summary['trading_dates_processed']}")
        print(f"Expiration Dates: {summary['expiration_dates']}")
        print(f"Total Contracts Generated: {summary['total_contracts']}")
        print(f"Total Quotes Generated: {summary['total_quotes']}")
        print("="*60)
        print("✅ Mock data generation completed successfully!")
        print("You can now test the LEAPS strategy with this data.")
        
    except Exception as e:
        logger.error(f"Mock data generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
