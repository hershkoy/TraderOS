"""
Tests for the Options Repository module.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, date
from data.options_repo import OptionsRepository, get_chain_at, select_leaps, select_short_calls

class TestOptionsRepository(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo = OptionsRepository()
        self.test_date = datetime(2024, 1, 15)
        
    def test_get_chain_at(self):
        """Test getting option chain at specific timestamp."""
        # Mock the database connection and query result
        mock_df = pd.DataFrame({
            'ts': [self.test_date],
            'underlying': ['QQQ'],
            'expiration': [date(2025, 1, 17)],
            'strike_cents': [35000],
            'option_right': ['C'],
            'bid': [10.0],
            'ask': [10.5],
            'last': [10.25],
            'volume': [100],
            'open_interest': [1000],
            'iv': [0.25],
            'delta': [0.7],
            'gamma': [0.01],
            'theta': [-0.05],
            'vega': [0.1],
            'option_id': ['QQQ_2025-01-17_000350C'],
            'multiplier': [100],
            'strike_price': [350.0],
            'underlying_close': [360.0],
            'moneyness': [1.03],
            'days_to_expiration': [367]
        })
        
        with patch('data.options_repo.pd.read_sql_query', return_value=mock_df):
            result = self.repo.get_chain_at(self.test_date, 'QQQ')
            
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['underlying'], 'QQQ')
        self.assertEqual(result.iloc[0]['option_id'], 'QQQ_2025-01-17_000350C')
    
    def test_select_leaps(self):
        """Test selecting LEAPS candidates."""
        # Mock the database connection and query result
        mock_df = pd.DataFrame({
            'ts': [self.test_date],
            'underlying': ['QQQ'],
            'expiration': [date(2025, 1, 17)],
            'strike_cents': [35000],
            'option_right': ['C'],
            'bid': [10.0],
            'ask': [10.5],
            'last': [10.25],
            'volume': [100],
            'open_interest': [1000],
            'iv': [0.25],
            'delta': [0.7],
            'gamma': [0.01],
            'theta': [-0.05],
            'vega': [0.1],
            'option_id': ['QQQ_2025-01-17_000350C'],
            'multiplier': [100],
            'strike_price': [350.0],
            'underlying_close': [360.0],
            'moneyness': [1.03],
            'days_to_expiration': [367],
            'suitability_score': ['Optimal']
        })
        
        with patch('data.options_repo.pd.read_sql_query', return_value=mock_df):
            result = self.repo.select_leaps(self.test_date, 'QQQ')
            
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['delta'], 0.7)
        self.assertEqual(result.iloc[0]['days_to_expiration'], 367)
    
    def test_select_short_calls(self):
        """Test selecting short call candidates."""
        # Mock the database connection and query result
        mock_df = pd.DataFrame({
            'ts': [self.test_date],
            'underlying': ['QQQ'],
            'expiration': [date(2024, 2, 16)],
            'strike_cents': [36000],
            'option_right': ['C'],
            'bid': [5.0],
            'ask': [5.5],
            'last': [5.25],
            'volume': [50],
            'open_interest': [500],
            'iv': [0.3],
            'delta': [0.25],
            'gamma': [0.02],
            'theta': [-0.1],
            'vega': [0.05],
            'option_id': ['QQQ_2024-02-16_000360C'],
            'multiplier': [100],
            'strike_price': [360.0],
            'underlying_close': [360.0],
            'moneyness': [1.0],
            'days_to_expiration': [32],
            'suitability_score': ['Optimal']
        })
        
        with patch('data.options_repo.pd.read_sql_query', return_value=mock_df):
            result = self.repo.select_short_calls(self.test_date, 'QQQ')
            
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['delta'], 0.25)
        self.assertEqual(result.iloc[0]['days_to_expiration'], 32)
    
    def test_get_pmcc_candidates(self):
        """Test getting PMCC candidates."""
        # Mock both LEAPS and short calls
        leaps_df = pd.DataFrame({
            'option_id': ['QQQ_2025-01-17_000350C'],
            'delta': [0.7],
            'days_to_expiration': [367]
        })
        
        short_calls_df = pd.DataFrame({
            'option_id': ['QQQ_2024-02-16_000360C'],
            'delta': [0.25],
            'days_to_expiration': [32]
        })
        
        with patch.object(self.repo, 'select_leaps', return_value=leaps_df), \
             patch.object(self.repo, 'select_short_calls', return_value=short_calls_df):
            result = self.repo.get_pmcc_candidates(self.test_date, 'QQQ')
            
        self.assertIn('leaps', result)
        self.assertIn('short_calls', result)
        self.assertEqual(len(result['leaps']), 1)
        self.assertEqual(len(result['short_calls']), 1)
    
    def test_get_option_by_id(self):
        """Test getting option by ID."""
        mock_df = pd.DataFrame({
            'option_id': ['QQQ_2025-01-17_000350C'],
            'delta': [0.7],
            'strike_price': [350.0]
        })
        
        with patch('data.options_repo.pd.read_sql_query', return_value=mock_df):
            result = self.repo.get_option_by_id('QQQ_2025-01-17_000350C', self.test_date)
            
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result['option_id'], 'QQQ_2025-01-17_000350C')
        self.assertEqual(result['delta'], 0.7)
    
    def test_get_historical_quotes(self):
        """Test getting historical quotes."""
        mock_df = pd.DataFrame({
            'ts': [self.test_date, self.test_date],
            'bid': [10.0, 10.1],
            'ask': [10.5, 10.6],
            'last': [10.25, 10.35],
            'option_id': ['QQQ_2025-01-17_000350C', 'QQQ_2025-01-17_000350C']
        })
        
        with patch('data.options_repo.pd.read_sql_query', return_value=mock_df):
            result = self.repo.get_historical_quotes(
                'QQQ_2025-01-17_000350C', 
                date(2024, 1, 1), 
                date(2024, 1, 31)
            )
            
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['option_id'], 'QQQ_2025-01-17_000350C')
    
    def test_get_available_dates(self):
        """Test getting available dates."""
        mock_df = pd.DataFrame({
            'trade_date': [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]
        })
        
        with patch('data.options_repo.pd.read_sql_query', return_value=mock_df):
            result = self.repo.get_available_dates('QQQ')
            
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], date(2024, 1, 15))

class TestConvenienceFunctions(unittest.TestCase):
    
    def test_get_chain_at_function(self):
        """Test convenience function get_chain_at."""
        with patch('data.options_repo.OptionsRepository') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_chain_at.return_value = pd.DataFrame()
            
            result = get_chain_at(datetime(2024, 1, 15), 'QQQ')
            
            mock_repo.get_chain_at.assert_called_once_with(datetime(2024, 1, 15), 'QQQ')
            self.assertIsInstance(result, pd.DataFrame)
    
    def test_select_leaps_function(self):
        """Test convenience function select_leaps."""
        with patch('data.options_repo.OptionsRepository') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.select_leaps.return_value = pd.DataFrame()
            
            result = select_leaps(datetime(2024, 1, 15), 'QQQ')
            
            mock_repo.select_leaps.assert_called_once_with(datetime(2024, 1, 15), 'QQQ', (0.6, 0.85))
            self.assertIsInstance(result, pd.DataFrame)
    
    def test_select_short_calls_function(self):
        """Test convenience function select_short_calls."""
        with patch('data.options_repo.OptionsRepository') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.select_short_calls.return_value = pd.DataFrame()
            
            result = select_short_calls(datetime(2024, 1, 15), 'QQQ')
            
            mock_repo.select_short_calls.assert_called_once_with(
                datetime(2024, 1, 15), 'QQQ', (25, 45), (0.15, 0.35)
            )
            self.assertIsInstance(result, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
