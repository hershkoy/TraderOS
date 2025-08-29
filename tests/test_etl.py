"""
ETL Unit Tests for Options Data Pipeline

This module contains comprehensive unit tests for the ETL (Extract, Transform, Load)
functionality of the options data pipeline.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, date
import psycopg2
from psycopg2.extras import RealDictCursor

# Import the modules to test
from utils.option_utils import build_option_id
from utils.pg_copy import copy_rows
from utils.greeks import calculate_option_greeks_with_iv
from data.options_repo import OptionsRepository

class TestOptionUtils(unittest.TestCase):
    """Test option utility functions."""
    
    def test_build_option_id(self):
        """Test option ID building with various inputs."""
        # Test basic call option
        option_id = build_option_id('QQQ', date(2025, 6, 20), 350.0, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_035000C')
        
        # Test put option
        option_id = build_option_id('QQQ', date(2025, 6, 20), 350.0, 'P')
        self.assertEqual(option_id, 'QQQ_2025-06-20_035000P')
        
        # Test different underlying
        option_id = build_option_id('SPY', date(2025, 6, 20), 350.0, 'C')
        self.assertEqual(option_id, 'SPY_2025-06-20_035000C')
        
        # Test decimal strike
        option_id = build_option_id('QQQ', date(2025, 6, 20), 350.50, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_035050C')
        
        # Test very high strike
        option_id = build_option_id('QQQ', date(2025, 6, 20), 1000.0, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_100000C')
    
    def test_build_option_id_edge_cases(self):
        """Test option ID building with edge cases."""
        # Test zero strike (should not happen in practice)
        option_id = build_option_id('QQQ', date(2025, 6, 20), 0.0, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_000000C')
        
        # Test very small strike
        option_id = build_option_id('QQQ', date(2025, 6, 20), 0.01, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_000001C')
        
        # Test rounding behavior
        option_id = build_option_id('QQQ', date(2025, 6, 20), 350.49, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_035049C')
        
        option_id = build_option_id('QQQ', date(2025, 6, 20), 350.51, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_035051C')

class TestPgCopy(unittest.TestCase):
    """Test PostgreSQL COPY functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = Mock()
        self.mock_cursor = Mock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        
        # Make the cursor support context manager protocol
        self.mock_cursor.__enter__ = Mock(return_value=self.mock_cursor)
        self.mock_cursor.__exit__ = Mock(return_value=None)
        
    def test_copy_rows_basic(self):
        """Test basic COPY functionality."""
        # Mock data
        table = 'option_contracts'
        columns = ['option_id', 'underlying', 'expiration', 'strike_cents', 'option_right']
        rows = [
            ('QQQ_2025-06-20_000350C', 'QQQ', '2025-06-20', 35000, 'C'),
            ('QQQ_2025-06-20_000360C', 'QQQ', '2025-06-20', 36000, 'C'),
        ]
        
        # Mock the COPY command
        self.mock_cursor.copy_expert.return_value = None
        
        # Test the function
        result = copy_rows(self.mock_conn, table, columns, rows)
        
        # Verify the COPY command was called
        self.mock_cursor.copy_expert.assert_called_once()
        copy_command = self.mock_cursor.copy_expert.call_args[0][0]
        self.assertIn('COPY option_contracts', copy_command)
        self.assertIn('option_id,underlying,expiration,strike_cents,option_right', copy_command)
        
        # Verify the data was written
        self.mock_cursor.write.assert_called()
    
    def test_copy_rows_empty_data(self):
        """Test COPY with empty data."""
        table = 'option_contracts'
        columns = ['option_id', 'underlying']
        rows = []
        
        result = copy_rows(self.mock_conn, table, columns, rows)
        
        # Should not call copy_expert for empty data
        self.mock_cursor.copy_expert.assert_not_called()
    
    def test_copy_rows_error_handling(self):
        """Test COPY error handling."""
        table = 'option_contracts'
        columns = ['option_id', 'underlying']
        rows = [('QQQ_2025-06-20_000350C', 'QQQ')]
        
        # Mock an error
        self.mock_cursor.copy_expert.side_effect = psycopg2.Error("COPY failed")
        
        # Test that the error is raised
        with self.assertRaises(psycopg2.Error):
            copy_rows(self.mock_conn, table, columns, rows)

class TestGreeksCalculation(unittest.TestCase):
    """Test Greeks calculation functionality."""
    
    def test_calculate_option_greeks_with_iv_basic(self):
        """Test basic Greeks calculation."""
        result = calculate_option_greeks_with_iv(
            underlying_price=100.0,
            strike=100.0,
            days_to_exp=365,
            option_price=10.0,
            option_type='C',
            risk_free_rate=0.05,
            dividend_yield=0.0
        )
        
        # Check that all expected fields are present
        expected_fields = ['implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        for field in expected_fields:
            self.assertIn(field, result)
            self.assertIsNotNone(result[field])
        
        # Check reasonable ranges
        self.assertGreater(result['implied_volatility'], 0)
        self.assertLess(result['implied_volatility'], 1)
        self.assertGreater(result['delta'], 0)
        self.assertLess(result['delta'], 1)
        self.assertGreater(result['gamma'], 0)
        self.assertLess(result['theta'], 0)  # Theta should be negative for long options
        self.assertGreater(result['vega'], 0)
    
    def test_calculate_option_greeks_with_iv_put(self):
        """Test Greeks calculation for put options."""
        result = calculate_option_greeks_with_iv(
            underlying_price=100.0,
            strike=100.0,
            days_to_exp=365,
            option_price=10.0,
            option_type='P',
            risk_free_rate=0.05,
            dividend_yield=0.0
        )
        
        # Check that all expected fields are present
        expected_fields = ['implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        for field in expected_fields:
            self.assertIn(field, result)
            self.assertIsNotNone(result[field])
        
        # Check reasonable ranges for put
        self.assertGreater(result['implied_volatility'], 0)
        self.assertLess(result['implied_volatility'], 1)
        self.assertLess(result['delta'], 0)  # Put delta should be negative
        self.assertGreater(result['gamma'], 0)
        self.assertLess(result['theta'], 0)  # Theta should be negative
        self.assertGreater(result['vega'], 0)
    
    def test_calculate_option_greeks_with_iv_edge_cases(self):
        """Test Greeks calculation with edge cases."""
        # Test with very short time to expiration
        result = calculate_option_greeks_with_iv(
            underlying_price=100.0,
            strike=90.0,  # ITM call
            days_to_exp=1,
            option_price=10.0,
            option_type='C',
            risk_free_rate=0.05,
            dividend_yield=0.0
        )
        
        # For very short time to expiration, IV calculation might fail
        # but the function should still return a result structure
        self.assertIn('delta', result)
        self.assertIn('implied_volatility', result)
        # IV might be None for edge cases, which is acceptable
        
        # Test with zero time to expiration
        result = calculate_option_greeks_with_iv(
            underlying_price=100.0,
            strike=90.0,
            days_to_exp=0,
            option_price=10.0,
            option_type='C',
            risk_free_rate=0.05,
            dividend_yield=0.0
        )
        
        # Should handle zero time gracefully
        self.assertIsNotNone(result['delta'])

class TestOptionsRepository(unittest.TestCase):
    """Test options repository functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo = OptionsRepository()
        self.test_date = datetime(2024, 1, 15)
    
    @patch('data.options_repo.pd.read_sql_query')
    def test_get_chain_at(self, mock_read_sql):
        """Test getting option chain at specific timestamp."""
        # Mock the database response
        mock_df = pd.DataFrame({
            'option_id': ['QQQ_2025-06-20_000350C'],
            'underlying': ['QQQ'],
            'delta': [0.7]
        })
        mock_read_sql.return_value = mock_df
        
        result = self.repo.get_chain_at(self.test_date, 'QQQ')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['option_id'], 'QQQ_2025-06-20_000350C')
        
        # Verify the query was called
        mock_read_sql.assert_called_once()
    
    @patch('data.options_repo.pd.read_sql_query')
    def test_select_leaps(self, mock_read_sql):
        """Test selecting LEAPS candidates."""
        # Mock the database response
        mock_df = pd.DataFrame({
            'option_id': ['QQQ_2025-06-20_000350C'],
            'delta': [0.7],
            'days_to_expiration': [367],
            'suitability_score': ['Optimal']
        })
        mock_read_sql.return_value = mock_df
        
        result = self.repo.select_leaps(self.test_date, 'QQQ')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['delta'], 0.7)
        self.assertEqual(result.iloc[0]['days_to_expiration'], 367)
        
        # Verify the query was called
        mock_read_sql.assert_called_once()
    
    @patch('data.options_repo.pd.read_sql_query')
    def test_select_short_calls(self, mock_read_sql):
        """Test selecting short call candidates."""
        # Mock the database response
        mock_df = pd.DataFrame({
            'option_id': ['QQQ_2024-02-16_000360C'],
            'delta': [0.25],
            'days_to_expiration': [32],
            'suitability_score': ['Optimal']
        })
        mock_read_sql.return_value = mock_df
        
        result = self.repo.select_short_calls(self.test_date, 'QQQ')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['delta'], 0.25)
        self.assertEqual(result.iloc[0]['days_to_expiration'], 32)
        
        # Verify the query was called
        mock_read_sql.assert_called_once()
    
    @patch('data.options_repo.pd.read_sql_query')
    def test_get_option_by_id(self, mock_read_sql):
        """Test getting option by ID."""
        # Mock the database response
        mock_df = pd.DataFrame({
            'option_id': ['QQQ_2025-06-20_000350C'],
            'delta': [0.7],
            'strike_price': [350.0]
        })
        mock_read_sql.return_value = mock_df
        
        result = self.repo.get_option_by_id('QQQ_2025-06-20_000350C', self.test_date)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result['option_id'], 'QQQ_2025-06-20_000350C')
        self.assertEqual(result['delta'], 0.7)
        
        # Verify the query was called
        mock_read_sql.assert_called_once()
    
    @patch('data.options_repo.pd.read_sql_query')
    def test_get_historical_quotes(self, mock_read_sql):
        """Test getting historical quotes."""
        # Mock the database response
        mock_df = pd.DataFrame({
            'ts': [self.test_date, self.test_date],
            'bid': [10.0, 10.1],
            'ask': [10.5, 10.6],
            'option_id': ['QQQ_2025-06-20_000350C', 'QQQ_2025-06-20_000350C']
        })
        mock_read_sql.return_value = mock_df
        
        result = self.repo.get_historical_quotes(
            'QQQ_2025-06-20_000350C',
            date(2024, 1, 1),
            date(2024, 1, 31)
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['option_id'], 'QQQ_2025-06-20_000350C')
        
        # Verify the query was called
        mock_read_sql.assert_called_once()
    
    @patch('data.options_repo.pd.read_sql_query')
    def test_get_available_dates(self, mock_read_sql):
        """Test getting available dates."""
        # Mock the database response
        mock_df = pd.DataFrame({
            'trade_date': [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]
        })
        mock_read_sql.return_value = mock_df
        
        result = self.repo.get_available_dates('QQQ')
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], date(2024, 1, 15))
        
        # Verify the query was called
        mock_read_sql.assert_called_once()

class TestETLIntegration(unittest.TestCase):
    """Test ETL integration scenarios."""
    
    def test_end_to_end_option_processing(self):
        """Test end-to-end option processing workflow."""
        # This test simulates the complete ETL workflow
        # 1. Build option ID
        option_id = build_option_id('QQQ', date(2025, 6, 20), 350.0, 'C')
        self.assertEqual(option_id, 'QQQ_2025-06-20_035000C')
        
        # 2. Calculate Greeks
        greeks = calculate_option_greeks_with_iv(
            underlying_price=360.0,
            strike=350.0,
            days_to_exp=367,
            option_price=15.0,
            option_type='C',
            risk_free_rate=0.05,
            dividend_yield=0.0
        )
        
        # Verify Greeks are reasonable
        self.assertIsNotNone(greeks['implied_volatility'])
        self.assertIsNotNone(greeks['delta'])
        self.assertGreater(greeks['delta'], 0.5)  # Should be ITM
        
        # 3. Test repository operations (with mocked database)
        with patch('data.options_repo.pd.read_sql_query') as mock_read_sql:
            mock_df = pd.DataFrame({
                'option_id': [option_id],
                'delta': [greeks['delta']],
                'underlying': ['QQQ']
            })
            mock_read_sql.return_value = mock_df
            
            repo = OptionsRepository()
            result = repo.get_chain_at(datetime(2024, 1, 15), 'QQQ')
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]['option_id'], option_id)

if __name__ == '__main__':
    unittest.main()
