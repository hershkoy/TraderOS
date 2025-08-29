#!/usr/bin/env python3
"""
Unit tests for PMCC strategy provider.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta
import pandas as pd

from strategies.pmcc_provider import PMCCProvider


class TestPMCCProvider(unittest.TestCase):
    """Test PMCC strategy provider."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }
        self.provider = PMCCProvider(self.db_config, 'QQQ')
        self.test_date = datetime(2024, 1, 15)
        
    def test_initialization(self):
        """Test provider initialization."""
        self.assertEqual(self.provider.underlying, 'QQQ')
        self.assertEqual(self.provider.leaps_delta_band, (0.6, 0.85))
        self.assertEqual(self.provider.short_call_dte_band, (25, 45))
        self.assertEqual(self.provider.short_call_delta_band, (0.15, 0.35))
        self.assertEqual(self.provider.spread_haircut_pct, 0.5)
    
    @patch('strategies.pmcc_provider.OptionsRepository')
    def test_get_available_dates(self, mock_repo_class):
        """Test getting available dates."""
        mock_repo = Mock()
        mock_repo.get_available_dates.return_value = [
            date(2024, 1, 10),
            date(2024, 1, 11),
            date(2024, 1, 12)
        ]
        mock_repo_class.return_value = mock_repo
        
        self.provider.options_repo = mock_repo
        
        dates = self.provider.get_available_dates()
        
        self.assertEqual(len(dates), 3)
        mock_repo.get_available_dates.assert_called_once_with('QQQ')
    
    @patch('strategies.pmcc_provider.OptionsRepository')
    def test_get_pmcc_candidates_success(self, mock_repo_class):
        """Test successful PMCC candidate selection."""
        # Mock options repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
                # Mock LEAPS data
        leaps_data = pd.DataFrame({
            'option_id': ['QQQ_2025-06-20_035000C'],
            'expiration': [date(2025, 6, 20)],
            'strike_cents': [35000],
            'delta': [0.75],
            'bid': [15.50],
            'ask': [15.75],
            'volume': [100],
            'open_interest': [500],
            'ts': [self.test_date],
            'underlying_close': [350.00],
            'moneyness': [1.00]
        })

        # Mock short call data - use a much higher strike to ensure positive max profit
        short_call_data = pd.DataFrame({
            'option_id': ['QQQ_2024-02-16_038000C'],
            'expiration': [date(2024, 2, 16)],
            'strike_cents': [38000],  # Much higher strike for positive max profit
            'delta': [0.25],
            'bid': [2.50],
            'ask': [2.60],
            'volume': [200],
            'open_interest': [1000],
            'ts': [self.test_date],
            'underlying_close': [350.00],
            'moneyness': [0.92]
        })

        mock_repo.select_leaps.return_value = leaps_data
        mock_repo.select_short_calls.return_value = short_call_data

        self.provider.options_repo = mock_repo

        # Mock assignment helper
        mock_assignment = Mock()
        mock_assignment.assess_assignment_risk.return_value = ['Low risk']
        self.provider.assignment_helper = mock_assignment

        # Mock the selection methods to return properly formatted data
        def mock_select_leaps(leaps_df):
            if leaps_df.empty:
                return None
            leaps = leaps_df.iloc[0].to_dict()
            leaps['mid_price'] = (leaps['bid'] + leaps['ask']) / 2
            leaps['spread_pct'] = (leaps['ask'] - leaps['bid']) / leaps['mid_price'] * 100
            leaps['fill_price'] = leaps['mid_price'] * (1 + self.provider.spread_haircut_pct / 100)
            return leaps

        def mock_select_short_call(short_calls_df):
            if short_calls_df.empty:
                return None
            short_call = short_calls_df.iloc[0].to_dict()
            short_call['mid_price'] = (short_call['bid'] + short_call['ask']) / 2
            short_call['spread_pct'] = (short_call['ask'] - short_call['bid']) / short_call['mid_price'] * 100
            short_call['fill_price'] = short_call['mid_price'] * (1 - self.provider.spread_haircut_pct / 100)
            short_call['assignment_risk'] = ['Low risk']
            return short_call

        # Patch the selection methods
        with patch.object(self.provider, '_select_best_leaps', side_effect=mock_select_leaps):
            with patch.object(self.provider, '_select_best_short_call', side_effect=mock_select_short_call):
                result = self.provider.get_pmcc_candidates(self.test_date)
        
        self.assertEqual(result['date'], self.test_date)
        self.assertEqual(result['underlying'], 'QQQ')
        self.assertIsNotNone(result['leaps'])
        self.assertIsNotNone(result['short_call'])
        self.assertIsNotNone(result['strategy_metrics'])
        self.assertNotIn('error', result)
        
        # Check LEAPS data
        leaps = result['leaps']
        self.assertEqual(leaps['option_id'], 'QQQ_2025-06-20_035000C')
        self.assertEqual(leaps['delta'], 0.75)
        self.assertEqual(leaps['mid_price'], 15.625)
        self.assertAlmostEqual(leaps['spread_pct'], 1.6, places=1)
        
        # Check short call data
        short_call = result['short_call']
        self.assertEqual(short_call['option_id'], 'QQQ_2024-02-16_038000C')
        self.assertEqual(short_call['delta'], 0.25)
        self.assertEqual(short_call['mid_price'], 2.55)
        self.assertAlmostEqual(short_call['spread_pct'], 3.9, places=1)
        
        # Check strategy metrics
        metrics = result['strategy_metrics']
        self.assertGreater(metrics['net_debit'], 0)
        self.assertGreater(metrics['max_profit'], 0)
        self.assertGreater(metrics['breakeven'], 350)  # leaps strike
    
    @patch('strategies.pmcc_provider.OptionsRepository')
    def test_get_pmcc_candidates_no_leaps(self, mock_repo_class):
        """Test PMCC candidate selection when no LEAPS available."""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Empty LEAPS data
        mock_repo.select_leaps.return_value = pd.DataFrame()
        mock_repo.select_short_calls.return_value = pd.DataFrame()
        
        self.provider.options_repo = mock_repo
        
        result = self.provider.get_pmcc_candidates(self.test_date)
        
        self.assertIsNone(result['leaps'])
        self.assertIsNone(result['short_call'])
        self.assertEqual(result['strategy_metrics']['net_debit'], 0)
    
    @patch('strategies.pmcc_provider.OptionsRepository')
    def test_get_pmcc_candidates_exception(self, mock_repo_class):
        """Test PMCC candidate selection with exception."""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Raise exception
        mock_repo.select_leaps.side_effect = Exception("Database error")
        
        self.provider.options_repo = mock_repo
        
        result = self.provider.get_pmcc_candidates(self.test_date)
        
        self.assertIn('error', result)
        self.assertIsNone(result['leaps'])
        self.assertIsNone(result['short_call'])
    
    def test_select_best_leaps(self):
        """Test LEAPS candidate selection logic."""
        # Create test data with multiple candidates
        leaps_data = pd.DataFrame({
            'option_id': ['LEAPS1', 'LEAPS2', 'LEAPS3'],
            'delta': [0.80, 0.70, 0.75],  # 0.75 is optimal
            'volume': [100, 200, 150],
            'open_interest': [500, 1000, 800],
            'bid': [15.0, 14.0, 15.5],
            'ask': [15.5, 14.5, 15.75]
        })
        
        best_leaps = self.provider._select_best_leaps(leaps_data)
        
        # Should select LEAPS3 (delta closest to 0.75)
        self.assertEqual(best_leaps['option_id'], 'LEAPS3')
        self.assertEqual(best_leaps['delta'], 0.75)
        self.assertEqual(best_leaps['mid_price'], 15.625)
    
    def test_select_best_short_call(self):
        """Test short call candidate selection logic."""
        # Create test data with multiple candidates
        short_call_data = pd.DataFrame({
            'option_id': ['SC1', 'SC2', 'SC3'],
            'delta': [0.30, 0.20, 0.25],  # 0.25 is optimal
            'volume': [100, 200, 150],
            'open_interest': [500, 1000, 800],
            'bid': [2.0, 1.5, 2.5],
            'ask': [2.2, 1.7, 2.6],
            'expiration': [date(2024, 2, 16)] * 3,
            'ts': [self.test_date] * 3,
            'moneyness': [1.02, 0.98, 1.01]
        })
        
        # Mock assignment helper
        mock_assignment = Mock()
        mock_assignment.assess_assignment_risk.return_value = ['Low risk']
        self.provider.assignment_helper = mock_assignment
        
        best_short_call = self.provider._select_best_short_call(short_call_data)
        
        # Should select SC3 (delta closest to 0.25)
        self.assertEqual(best_short_call['option_id'], 'SC3')
        self.assertEqual(best_short_call['delta'], 0.25)
        self.assertEqual(best_short_call['mid_price'], 2.55)
        self.assertIn('assignment_risk', best_short_call)
    
    def test_calculate_strategy_metrics(self):
        """Test strategy metrics calculation."""
        leaps = {
            'fill_price': 15.625,
            'strike_cents': 35000,
            'delta': 0.75
        }
        
        short_call = {
            'fill_price': 2.55,
            'strike_cents': 36000
        }
        
        metrics = self.provider._calculate_strategy_metrics(leaps, short_call)
        
        # Calculate expected values
        leaps_cost = 15.625 * 100
        short_call_credit = 2.55 * 100
        net_debit = leaps_cost - short_call_credit
        leaps_strike = 350.00
        short_strike = 360.00
        max_profit = (short_strike - leaps_strike) * 100 - net_debit
        
        self.assertEqual(metrics['net_debit'], net_debit)
        self.assertEqual(metrics['max_profit'], max_profit)
        self.assertEqual(metrics['max_loss'], net_debit)
        self.assertEqual(metrics['leaps_strike'], leaps_strike)
        self.assertEqual(metrics['short_strike'], short_strike)
        self.assertAlmostEqual(metrics['probability_of_profit'], 0.25, places=2)
    
    def test_calculate_strategy_metrics_none_candidates(self):
        """Test strategy metrics with None candidates."""
        metrics = self.provider._calculate_strategy_metrics(None, None)
        
        self.assertEqual(metrics['net_debit'], 0)
        self.assertEqual(metrics['max_profit'], 0)
        self.assertEqual(metrics['max_loss'], 0)
        self.assertEqual(metrics['breakeven'], 0)
        self.assertEqual(metrics['probability_of_profit'], 0)
    
    @patch('strategies.pmcc_provider.OptionsRepository')
    def test_get_historical_pmcc_data(self, mock_repo_class):
        """Test getting historical PMCC data."""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock available dates
        available_dates = [
            date(2024, 1, 10),
            date(2024, 1, 11),
            date(2024, 1, 12),
            date(2024, 1, 15),
            date(2024, 1, 16)
        ]
        mock_repo.get_available_dates.return_value = available_dates
        
        # Mock PMCC candidates
        mock_repo.select_leaps.return_value = pd.DataFrame()
        mock_repo.select_short_calls.return_value = pd.DataFrame()
        
        self.provider.options_repo = mock_repo
        
        start_date = date(2024, 1, 11)
        end_date = date(2024, 1, 15)
        
        results = self.provider.get_historical_pmcc_data(start_date, end_date)
        
        # Should get data for 3 dates (11, 12, 15)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIn('date', result)
            self.assertEqual(result['underlying'], 'QQQ')
    
    def test_set_strategy_parameters(self):
        """Test updating strategy parameters."""
        # Test initial values
        self.assertEqual(self.provider.leaps_delta_band, (0.6, 0.85))
        self.assertEqual(self.provider.short_call_dte_band, (25, 45))
        self.assertEqual(self.provider.short_call_delta_band, (0.15, 0.35))
        self.assertEqual(self.provider.spread_haircut_pct, 0.5)
        
        # Update parameters
        self.provider.set_strategy_parameters(
            leaps_delta_band=(0.7, 0.9),
            short_call_dte_band=(30, 50),
            short_call_delta_band=(0.2, 0.4),
            spread_haircut_pct=1.0
        )
        
        # Check updated values
        self.assertEqual(self.provider.leaps_delta_band, (0.7, 0.9))
        self.assertEqual(self.provider.short_call_dte_band, (30, 50))
        self.assertEqual(self.provider.short_call_delta_band, (0.2, 0.4))
        self.assertEqual(self.provider.spread_haircut_pct, 1.0)


if __name__ == '__main__':
    unittest.main()
