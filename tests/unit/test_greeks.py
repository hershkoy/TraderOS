"""
Tests for the Greeks utility functions.
"""

import unittest
import math
from utils.greeks import (
    black_scholes_call,
    black_scholes_put,
    calculate_greeks,
    implied_volatility,
    calculate_option_greeks_with_iv
)

class TestGreeks(unittest.TestCase):
    
    def test_black_scholes_call(self):
        """Test Black-Scholes call option pricing."""
        # Test case: S=100, K=100, T=1, r=0.05, sigma=0.2
        price = black_scholes_call(100, 100, 1, 0.05, 0.2)
        self.assertGreater(price, 0)
        self.assertLess(price, 20)  # Should be reasonable
        
        # Test at expiration (T=0)
        price = black_scholes_call(100, 90, 0, 0.05, 0.2)
        self.assertEqual(price, 10)  # Intrinsic value
        
    def test_black_scholes_put(self):
        """Test Black-Scholes put option pricing."""
        # Test case: S=100, K=100, T=1, r=0.05, sigma=0.2
        price = black_scholes_put(100, 100, 1, 0.05, 0.2)
        self.assertGreater(price, 0)
        self.assertLess(price, 20)  # Should be reasonable
        
        # Test at expiration (T=0)
        price = black_scholes_put(100, 110, 0, 0.05, 0.2)
        self.assertEqual(price, 10)  # Intrinsic value
        
    def test_calculate_greeks(self):
        """Test Greeks calculation."""
        # Test case: S=100, K=100, T=1, r=0.05, sigma=0.2, call
        greeks = calculate_greeks(100, 100, 1, 0.05, 0.2, 'C')
        
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('theta', greeks)
        self.assertIn('vega', greeks)
        
        # Delta should be between 0 and 1 for calls
        self.assertGreater(greeks['delta'], 0)
        self.assertLess(greeks['delta'], 1)
        
        # Gamma should be positive
        self.assertGreater(greeks['gamma'], 0)
        
        # Vega should be positive
        self.assertGreater(greeks['vega'], 0)
        
    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        # Test case: known price, solve for IV
        S, K, T, r = 100, 100, 1, 0.05
        sigma_true = 0.2
        price = black_scholes_call(S, K, T, r, sigma_true)
        
        # Calculate implied volatility
        iv = implied_volatility(price, S, K, T, r, 'C')
        
        self.assertIsNotNone(iv)
        self.assertAlmostEqual(iv, sigma_true, places=3)
        
    def test_calculate_option_greeks_with_iv(self):
        """Test complete Greeks calculation with IV."""
        # Test case
        result = calculate_option_greeks_with_iv(
            underlying_price=100,
            strike=100,
            days_to_exp=365,
            option_price=10.0,
            option_type='C',
            risk_free_rate=0.05,
            dividend_yield=0.0
        )
        
        self.assertIn('implied_volatility', result)
        self.assertIn('delta', result)
        self.assertIn('gamma', result)
        self.assertIn('theta', result)
        self.assertIn('vega', result)
        
        self.assertIsNotNone(result['implied_volatility'])
        self.assertIsNotNone(result['delta'])
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with zero time to expiration
        greeks = calculate_greeks(100, 90, 0, 0.05, 0.2, 'C')
        self.assertEqual(greeks['delta'], 1.0)  # ITM call at expiration
        self.assertEqual(greeks['gamma'], 0.0)
        
        # Test with very low volatility - this might not converge
        iv = implied_volatility(0.1, 100, 100, 1, 0.05, 'C')
        # For very low volatility, IV calculation might fail to converge
        # This is acceptable behavior
        if iv is not None:
            self.assertGreater(iv, 0)

if __name__ == '__main__':
    unittest.main()
