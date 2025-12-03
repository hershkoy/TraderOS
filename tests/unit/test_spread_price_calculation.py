"""
Test for spread price calculation function.

Tests the calculate_buy_limit_price_sequence function to ensure
it correctly calculates starting price and minimum price for BUY orders.
"""
import unittest
import sys
import os

# Add parent directory to path to import from scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.spreads_trader import calculate_buy_limit_price_sequence


class TestSpreadPriceCalculation(unittest.TestCase):
    """Test cases for buy limit price calculation."""
    
    def test_initial_estimate_24_cents(self):
        """Test with initial estimate of 24 cents - should start at 25 cents."""
        initial_estimate = 0.24
        min_price = 0.21
        
        starting_price, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=initial_estimate,
            min_price=min_price
        )
        
        # Starting price should be -0.25 (one cent more than 0.24)
        self.assertEqual(starting_price, -0.25)
        # Minimum price should be -0.21
        self.assertEqual(min_price_ib, -0.21)
    
    def test_initial_estimate_20_cents(self):
        """Test with initial estimate of 20 cents - should start at 21 cents (min)."""
        initial_estimate = 0.20
        min_price = 0.21
        
        starting_price, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=initial_estimate,
            min_price=min_price
        )
        
        # Starting price should be clamped to minimum (0.21)
        self.assertEqual(starting_price, -0.21)
        self.assertEqual(min_price_ib, -0.21)
    
    def test_initial_estimate_30_cents(self):
        """Test with initial estimate of 30 cents - should start at 31 cents."""
        initial_estimate = 0.30
        min_price = 0.21
        
        starting_price, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=initial_estimate,
            min_price=min_price
        )
        
        # Starting price should be -0.31 (one cent more than 0.30)
        self.assertEqual(starting_price, -0.31)
        # Minimum price should be -0.21
        self.assertEqual(min_price_ib, -0.21)
    
    def test_custom_minimum_price(self):
        """Test with custom minimum price."""
        initial_estimate = 0.24
        min_price = 0.22
        
        starting_price, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=initial_estimate,
            min_price=min_price
        )
        
        # Starting price should be -0.25
        self.assertEqual(starting_price, -0.25)
        # Minimum price should be -0.22
        self.assertEqual(min_price_ib, -0.22)
    
    def test_price_sequence_logic(self):
        """Test that the price sequence logic is correct."""
        initial_estimate = 0.24
        min_price = 0.21
        price_reduction = 0.01
        
        starting_price, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=initial_estimate,
            min_price=min_price,
            price_reduction_per_minute=price_reduction
        )
        
        # Verify starting price
        self.assertEqual(starting_price, -0.25)
        
        # Simulate the sequence: -0.25, -0.24, -0.23, -0.22, -0.21
        # For BUY orders, we make price less negative (closer to zero) by adding
        # So we go from -0.25 (more negative) to -0.21 (less negative)
        current = starting_price
        steps = []
        while current <= min_price_ib:  # Less than or equal (both negative, so <= means less negative)
            steps.append(current)
            if current >= min_price_ib:  # Stop if we've reached minimum
                break
            current = round(current + price_reduction, 2)
        
        expected_steps = [-0.25, -0.24, -0.23, -0.22, -0.21]
        self.assertEqual(steps, expected_steps)
    
    def test_estimate_below_minimum(self):
        """Test when estimate is below minimum - should clamp to minimum."""
        initial_estimate = 0.15
        min_price = 0.21
        
        starting_price, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=initial_estimate,
            min_price=min_price
        )
        
        # Should be clamped to minimum
        self.assertEqual(starting_price, -0.21)
        self.assertEqual(min_price_ib, -0.21)


if __name__ == '__main__':
    unittest.main()

