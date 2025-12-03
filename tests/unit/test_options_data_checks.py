#!/usr/bin/env python3
"""
Unit tests for options data integrity checks.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

from scripts.options_data_checks import OptionsDataChecker


class TestOptionsDataChecker(unittest.TestCase):
    """Test options data integrity checker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }
        self.checker = OptionsDataChecker(self.db_config)
        self.test_date = date(2024, 1, 15)
    
    def _setup_mock_connection(self, mock_connect, mock_cursor):
        """Helper method to set up mock connection and cursor."""
        mock_conn = Mock()
        # Set up the connection as a context manager
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        
        # Set up the cursor as a context manager
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        return mock_conn
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_contract_count_growth_normal(self, mock_connect):
        """Test contract count growth check with normal growth."""
        # Mock database response
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {'date': date(2024, 1, 14), 'contract_count': 1000},
            {'date': date(2024, 1, 15), 'contract_count': 1050}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_contract_count_growth(self.test_date)
        
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['today_count'], 1050)
        self.assertEqual(result['yesterday_count'], 1000)
        self.assertEqual(result['growth_pct'], 5.0)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_contract_count_growth_warning(self, mock_connect):
        """Test contract count growth check with warning-level growth."""
        # Mock database response - 30% growth (warning)
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {'date': date(2024, 1, 14), 'contract_count': 1000},
            {'date': date(2024, 1, 15), 'contract_count': 1300}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_contract_count_growth(self.test_date)
        
        self.assertEqual(result['status'], 'WARNING')
        self.assertEqual(result['growth_pct'], 30.0)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_contract_count_growth_fail(self, mock_connect):
        """Test contract count growth check with fail-level growth."""
        # Mock database response - 60% growth (fail)
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {'date': date(2024, 1, 14), 'contract_count': 1000},
            {'date': date(2024, 1, 15), 'contract_count': 1600}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_contract_count_growth(self.test_date)
        
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['growth_pct'], 60.0)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_monotonic_last_seen_pass(self, mock_connect):
        """Test monotonic last_seen check with no issues."""
        # Mock database responses
        mock_cursor = Mock()
        mock_cursor.fetchall.side_effect = [
            [{'option_id': 'QQQ_2025-06-20_035000C', 'underlying': 'QQQ', 'expiration': date(2025, 6, 20)}],
            [{'first_seen': datetime(2024, 1, 1), 'last_seen': datetime(2024, 1, 15)}]
        ]
        mock_cursor.fetchone.return_value = {'first_seen': datetime(2024, 1, 1), 'last_seen': datetime(2024, 1, 15)}
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_monotonic_last_seen(sample_size=1)
        
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['issues']), 0)
        self.assertEqual(result['checked_count'], 1)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_monotonic_last_seen_fail(self, mock_connect):
        """Test monotonic last_seen check with issues."""
        # Mock database responses - last_seen < first_seen
        mock_cursor = Mock()
        mock_cursor.fetchall.side_effect = [
            [{'option_id': 'QQQ_2025-06-20_035000C', 'underlying': 'QQQ', 'expiration': date(2025, 6, 20)}],
            [{'first_seen': datetime(2024, 1, 15), 'last_seen': datetime(2024, 1, 1)}]
        ]
        mock_cursor.fetchone.return_value = {'first_seen': datetime(2024, 1, 15), 'last_seen': datetime(2024, 1, 1)}
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_monotonic_last_seen(sample_size=1)
        
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(len(result['issues']), 1)
        self.assertEqual(result['checked_count'], 1)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_quotes_contract_consistency_pass(self, mock_connect):
        """Test quotes consistency check with good consistency."""
        # Mock database responses
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            {'active_contracts': 1000},
            {'quoted_contracts': 950}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_quotes_contract_consistency(self.test_date)
        
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['consistency_pct'], 95.0)
        self.assertEqual(result['active_contracts'], 1000)
        self.assertEqual(result['quoted_contracts'], 950)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_quotes_contract_consistency_warning(self, mock_connect):
        """Test quotes consistency check with warning-level consistency."""
        # Mock database responses - 85% consistency (warning)
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            {'active_contracts': 1000},
            {'quoted_contracts': 850}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_quotes_contract_consistency(self.test_date)
        
        self.assertEqual(result['status'], 'WARNING')
        self.assertEqual(result['consistency_pct'], 85.0)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_quotes_contract_consistency_fail(self, mock_connect):
        """Test quotes consistency check with fail-level consistency."""
        # Mock database responses - 40% consistency (fail)
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            {'active_contracts': 1000},
            {'quoted_contracts': 400}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_quotes_contract_consistency(self.test_date)
        
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['consistency_pct'], 40.0)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_data_quality_pass(self, mock_connect):
        """Test data quality check with no issues."""
        # Mock database responses - no issues
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            {'null_expiration_count': 0},
            {'invalid_spread_count': 0},
            {'null_underlying_count': 0}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_data_quality(self.test_date)
        
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['issues']), 0)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_data_quality_warning(self, mock_connect):
        """Test data quality check with warning-level issues."""
        # Mock database responses - some issues
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            {'null_expiration_count': 5},
            {'invalid_spread_count': 0},
            {'null_underlying_count': 0}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_data_quality(self.test_date)
        
        self.assertEqual(result['status'], 'WARNING')
        self.assertEqual(len(result['issues']), 1)
        self.assertIn('5 contracts with NULL expiration', result['issues'])
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_check_data_quality_fail(self, mock_connect):
        """Test data quality check with fail-level issues."""
        # Mock database responses - many issues
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            {'null_expiration_count': 50},
            {'invalid_spread_count': 100},
            {'null_underlying_count': 25}
        ]
        
        mock_conn = self._setup_mock_connection(mock_connect, mock_cursor)
        
        result = self.checker.check_data_quality(self.test_date)
        
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(len(result['issues']), 3)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_run_all_checks_pass(self, mock_connect):
        """Test running all checks with all passing."""
        # Mock all checks to pass
        with patch.object(self.checker, 'check_contract_count_growth') as mock_growth, \
             patch.object(self.checker, 'check_monotonic_last_seen') as mock_monotonic, \
             patch.object(self.checker, 'check_quotes_contract_consistency') as mock_consistency, \
             patch.object(self.checker, 'check_data_quality') as mock_quality:
            
            mock_growth.return_value = {'status': 'PASS', 'message': 'Growth OK'}
            mock_monotonic.return_value = {'status': 'PASS', 'message': 'Monotonic OK'}
            mock_consistency.return_value = {'status': 'PASS', 'message': 'Consistency OK'}
            mock_quality.return_value = {'status': 'PASS', 'message': 'Quality OK'}
            
            result = self.checker.run_all_checks(self.test_date)
            
            self.assertEqual(result['overall_status'], 'PASS')
            self.assertEqual(len(result['checks']), 4)
            self.assertEqual(result['check_date'], self.test_date)
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_run_all_checks_fail(self, mock_connect):
        """Test running all checks with one failing."""
        # Mock one check to fail
        with patch.object(self.checker, 'check_contract_count_growth') as mock_growth, \
             patch.object(self.checker, 'check_monotonic_last_seen') as mock_monotonic, \
             patch.object(self.checker, 'check_quotes_contract_consistency') as mock_consistency, \
             patch.object(self.checker, 'check_data_quality') as mock_quality:
            
            mock_growth.return_value = {'status': 'PASS', 'message': 'Growth OK'}
            mock_monotonic.return_value = {'status': 'FAIL', 'message': 'Monotonic FAIL'}
            mock_consistency.return_value = {'status': 'PASS', 'message': 'Consistency OK'}
            mock_quality.return_value = {'status': 'PASS', 'message': 'Quality OK'}
            
            result = self.checker.run_all_checks(self.test_date)
            
            self.assertEqual(result['overall_status'], 'FAIL')
    
    @patch('scripts.options_data_checks.psycopg2.connect')
    def test_run_all_checks_warning(self, mock_connect):
        """Test running all checks with one warning."""
        # Mock one check to warn
        with patch.object(self.checker, 'check_contract_count_growth') as mock_growth, \
             patch.object(self.checker, 'check_monotonic_last_seen') as mock_monotonic, \
             patch.object(self.checker, 'check_quotes_contract_consistency') as mock_consistency, \
             patch.object(self.checker, 'check_data_quality') as mock_quality:
            
            mock_growth.return_value = {'status': 'PASS', 'message': 'Growth OK'}
            mock_monotonic.return_value = {'status': 'WARNING', 'message': 'Monotonic WARNING'}
            mock_consistency.return_value = {'status': 'PASS', 'message': 'Consistency OK'}
            mock_quality.return_value = {'status': 'PASS', 'message': 'Quality OK'}
            
            result = self.checker.run_all_checks(self.test_date)
            
            self.assertEqual(result['overall_status'], 'WARNING')


if __name__ == '__main__':
    unittest.main()
