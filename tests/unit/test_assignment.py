"""
Tests for the Assignment Risk Helper module.
"""

import unittest
from datetime import date, timedelta
from utils.assignment import (
    AssignmentRiskHelper, 
    DividendInfo, 
    AssignmentRisk,
    should_flag_assignment,
    assess_assignment_risk
)

class TestAssignmentRiskHelper(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.helper = AssignmentRiskHelper()
        
        # Add some test dividends
        self.helper.add_dividend(
            symbol='QQQ',
            ex_date=date(2024, 2, 15),
            amount=0.50,
            payment_date=date(2024, 2, 28)
        )
        
        self.helper.add_dividend(
            symbol='QQQ',
            ex_date=date(2024, 5, 15),
            amount=0.50,
            payment_date=date(2024, 5, 28)
        )
    
    def test_add_dividend(self):
        """Test adding dividend information."""
        initial_count = len(self.helper.dividend_calendar)
        
        self.helper.add_dividend(
            symbol='SPY',
            ex_date=date(2024, 3, 15),
            amount=1.00,
            payment_date=date(2024, 3, 28)
        )
        
        self.assertEqual(len(self.helper.dividend_calendar), initial_count + 1)
        
        # Check the added dividend
        spy_dividend = next(
            (div for div in self.helper.dividend_calendar if div.symbol == 'SPY'),
            None
        )
        self.assertIsNotNone(spy_dividend)
        self.assertEqual(spy_dividend.amount, 1.00)
        self.assertEqual(spy_dividend.ex_date, date(2024, 3, 15))
    
    def test_get_next_dividend(self):
        """Test getting next dividend."""
        current_date = date(2024, 1, 15)
        next_dividend = self.helper.get_next_dividend('QQQ', current_date)
        
        self.assertIsNotNone(next_dividend)
        self.assertEqual(next_dividend.ex_date, date(2024, 2, 15))
        self.assertEqual(next_dividend.amount, 0.50)
        
        # Test with date after all dividends
        future_date = date(2024, 6, 1)
        next_dividend = self.helper.get_next_dividend('QQQ', future_date)
        self.assertIsNone(next_dividend)
        
        # Test with non-existent symbol
        next_dividend = self.helper.get_next_dividend('SPY', current_date)
        self.assertIsNone(next_dividend)
    
    def test_should_flag_assignment_immediate_risk(self):
        """Test assignment risk for immediate expiration."""
        # ITM and very close to expiration
        result = self.helper.should_flag_assignment(
            short_delta=-0.7,
            moneyness=1.02,
            days_to_exp=2
        )
        self.assertTrue(result)
        
        # OTM and close to expiration (should not flag)
        result = self.helper.should_flag_assignment(
            short_delta=-0.3,
            moneyness=0.98,
            days_to_exp=2
        )
        self.assertFalse(result)
    
    def test_should_flag_assignment_dividend_risk(self):
        """Test assignment risk related to dividends."""
        # Test with dividend approaching
        current_date = date(2024, 2, 10)  # 5 days before ex-dividend
        
        # ITM and dividend approaching
        result = self.helper.should_flag_assignment(
            short_delta=-0.6,
            moneyness=1.01,
            days_to_exp=30,
            symbol='QQQ',
            current_date=current_date
        )
        self.assertTrue(result)
        
        # OTM and dividend approaching (should not flag)
        result = self.helper.should_flag_assignment(
            short_delta=-0.3,
            moneyness=0.98,
            days_to_exp=30,
            symbol='QQQ',
            current_date=current_date
        )
        self.assertFalse(result)
    
    def test_should_flag_assignment_high_delta(self):
        """Test assignment risk for high delta positions."""
        # Deep ITM with high delta
        result = self.helper.should_flag_assignment(
            short_delta=-0.85,
            moneyness=1.06,
            days_to_exp=45
        )
        self.assertTrue(result)
        
        # Moderate delta (should not flag)
        result = self.helper.should_flag_assignment(
            short_delta=-0.6,
            moneyness=1.03,
            days_to_exp=45
        )
        self.assertFalse(result)
    
    def test_assess_assignment_risk(self):
        """Test comprehensive assignment risk assessment."""
        current_date = date(2024, 2, 10)  # 5 days before ex-dividend
        
        risk = self.helper.assess_assignment_risk(
            short_delta=-0.7,
            moneyness=1.02,
            days_to_exp=30,
            symbol='QQQ',
            current_date=current_date
        )
        
        self.assertIsInstance(risk, AssignmentRisk)
        self.assertTrue(risk.is_high_risk)
        self.assertTrue(risk.is_itm)
        self.assertEqual(risk.moneyness, 1.02)
        self.assertEqual(risk.days_to_expiration, 30)
        self.assertEqual(risk.days_to_ex_div, 5)
        self.assertIn("Ex-dividend in 5 days", risk.risk_factors)
        self.assertIn("ITM (moneyness: 1.020)", risk.risk_factors)
        self.assertIn("Moderate delta (-0.700)", risk.risk_factors)
    
    def test_get_roll_recommendations(self):
        """Test roll recommendations."""
        current_date = date(2024, 1, 15)
        current_strike = 350.0
        current_expiration = date(2024, 1, 20)  # 5 days to expiration
        underlying_price = 370.0  # Deep ITM (moneyness = 1.057)
        
        recommendations = self.helper.get_roll_recommendations(
            current_strike=current_strike,
            current_expiration=current_expiration,
            underlying_price=underlying_price,
            current_date=current_date
        )
        
        self.assertTrue(recommendations['should_roll'])
        self.assertEqual(recommendations['reason'], 'Close to expiration and ITM')
        # The suggested strike should be calculated as current_strike * 1.05 = 350 * 1.05 = 367.5
        self.assertIsNotNone(recommendations['suggested_strike'])
        self.assertAlmostEqual(recommendations['suggested_strike'], 367.5)
        self.assertIn('Higher strike reduces assignment probability', recommendations['risk_reduction'])
        
        # Test with OTM position
        underlying_price_otm = 340.0
        recommendations = self.helper.get_roll_recommendations(
            current_strike=current_strike,
            current_expiration=current_expiration,
            underlying_price=underlying_price_otm,
            current_date=current_date
        )
        
        self.assertFalse(recommendations['should_roll'])

class TestConvenienceFunctions(unittest.TestCase):
    
    def test_should_flag_assignment_function(self):
        """Test convenience function should_flag_assignment."""
        # Test high risk scenario
        result = should_flag_assignment(
            short_delta=-0.8,
            moneyness=1.05,
            days_to_exp=2
        )
        self.assertTrue(result)
        
        # Test low risk scenario
        result = should_flag_assignment(
            short_delta=-0.3,
            moneyness=0.95,
            days_to_exp=45
        )
        self.assertFalse(result)
    
    def test_assess_assignment_risk_function(self):
        """Test convenience function assess_assignment_risk."""
        risk = assess_assignment_risk(
            short_delta=-0.7,
            moneyness=1.02,
            days_to_exp=30,
            symbol='QQQ'
        )
        
        self.assertIsInstance(risk, AssignmentRisk)
        self.assertEqual(risk.moneyness, 1.02)
        self.assertEqual(risk.days_to_expiration, 30)
        self.assertTrue(risk.is_itm)

class TestDividendInfo(unittest.TestCase):
    
    def test_dividend_info_creation(self):
        """Test DividendInfo dataclass creation."""
        dividend = DividendInfo(
            ex_date=date(2024, 2, 15),
            amount=0.50,
            payment_date=date(2024, 2, 28),
            symbol='QQQ'
        )
        
        self.assertEqual(dividend.ex_date, date(2024, 2, 15))
        self.assertEqual(dividend.amount, 0.50)
        self.assertEqual(dividend.payment_date, date(2024, 2, 28))
        self.assertEqual(dividend.symbol, 'QQQ')

class TestAssignmentRisk(unittest.TestCase):
    
    def test_assignment_risk_creation(self):
        """Test AssignmentRisk dataclass creation."""
        risk = AssignmentRisk(
            is_high_risk=True,
            risk_factors=['ITM', 'Close to expiration'],
            days_to_expiration=5,
            is_itm=True,
            moneyness=1.02,
            days_to_ex_div=None,
            recommendation='Close position immediately'
        )
        
        self.assertTrue(risk.is_high_risk)
        self.assertTrue(risk.is_itm)
        self.assertEqual(risk.moneyness, 1.02)
        self.assertEqual(risk.days_to_expiration, 5)
        self.assertEqual(len(risk.risk_factors), 2)
        self.assertIn('ITM', risk.risk_factors)

if __name__ == '__main__':
    unittest.main()
