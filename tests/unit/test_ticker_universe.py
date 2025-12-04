"""
Tests for the Ticker Universe Management System
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data.ticker_universe import TickerUniverseManager


class TestTickerUniverseManager(unittest.TestCase):
    """Test cases for TickerUniverseManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the database client
        self.mock_db_client = Mock()
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        
        # Set up the mock chain
        self.mock_db_client.connection = self.mock_connection
        self.mock_connection.cursor.return_value = self.mock_cursor
        
        # Create manager with mocked client
        self.manager = TickerUniverseManager(db_client=self.mock_db_client)
    
    def test_ensure_tables_exist(self):
        """Test table creation"""
        # Mock successful table creation
        self.mock_cursor.execute.return_value = None
        self.mock_connection.commit.return_value = None
        
        # This should not raise an exception
        self.manager._ensure_tables_exist()
        
        # Verify that CREATE TABLE statements were executed
        self.mock_cursor.execute.assert_called()
        self.mock_connection.commit.assert_called()
    
    def test_is_cache_valid_valid_cache(self):
        """Test cache validity check with valid cache"""
        # Mock valid cache data
        mock_row = (datetime.now(), 24)
        self.mock_cursor.fetchone.return_value = mock_row
        
        result = self.manager._is_cache_valid('sp500')
        self.assertTrue(result)
    
    def test_is_cache_valid_expired_cache(self):
        """Test cache validity check with expired cache"""
        # Mock expired cache data (25 hours ago)
        expired_time = datetime.now() - timedelta(hours=25)
        mock_row = (expired_time, 24)
        self.mock_cursor.fetchone.return_value = mock_row
        
        result = self.manager._is_cache_valid('sp500')
        self.assertFalse(result)
    
    def test_is_cache_valid_no_cache(self):
        """Test cache validity check with no cache"""
        # Mock no cache data
        self.mock_cursor.fetchone.return_value = None
        
        result = self.manager._is_cache_valid('sp500')
        self.assertFalse(result)
    
    @patch('pandas.read_html')
    def test_fetch_sp500_tickers_success(self, mock_read_html):
        """Test successful S&P 500 ticker fetching"""
        # Mock pandas read_html response
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {'Symbol': 'AAPL', 'Security': 'Apple Inc.', 'GICS Sector': 'Technology'}),
            (1, {'Symbol': 'MSFT', 'Security': 'Microsoft Corporation', 'GICS Sector': 'Technology'}),
        ]
        mock_read_html.return_value = [mock_df]
        
        result = self.manager._fetch_sp500_tickers()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        self.assertEqual(result[0]['company_name'], 'Apple Inc.')
        self.assertEqual(result[0]['sector'], 'Technology')
    
    @patch('pandas.read_html')
    def test_fetch_sp500_tickers_failure(self, mock_read_html):
        """Test S&P 500 ticker fetching failure"""
        # Mock pandas read_html to raise exception
        mock_read_html.side_effect = Exception("Network error")
        
        result = self.manager._fetch_sp500_tickers()
        
        self.assertEqual(result, [])
    
    @patch('pandas.read_html')
    def test_fetch_nasdaq100_tickers_success(self, mock_read_html):
        """Test successful NASDAQ-100 ticker fetching"""
        # Mock pandas read_html response
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {'Ticker': 'AAPL', 'Company': 'Apple Inc.'}),
            (1, {'Ticker': 'GOOGL', 'Company': 'Alphabet Inc.'}),
        ]
        mock_read_html.return_value = [Mock(), Mock(), Mock(), Mock(), mock_df]
        
        result = self.manager._fetch_nasdaq100_tickers()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        self.assertEqual(result[0]['company_name'], 'Apple Inc.')
        self.assertEqual(result[0]['sector'], 'Technology')
    
    def test_cache_tickers(self):
        """Test ticker caching functionality"""
        # Mock successful caching
        self.mock_cursor.execute.return_value = None
        self.mock_connection.commit.return_value = None
        
        test_tickers = [
            {'symbol': 'AAPL', 'company_name': 'Apple Inc.', 'sector': 'Technology'},
            {'symbol': 'MSFT', 'company_name': 'Microsoft Corporation', 'sector': 'Technology'}
        ]
        
        # This should not raise an exception
        self.manager._cache_tickers('sp500', test_tickers)
        
        # Verify that caching operations were executed
        self.mock_cursor.execute.assert_called()
        self.mock_connection.commit.assert_called()
    
    def test_get_cached_tickers(self):
        """Test getting cached tickers"""
        # Mock cached ticker data
        mock_rows = [('AAPL',), ('MSFT',)]
        self.mock_cursor.fetchall.return_value = mock_rows
        
        result = self.manager._get_cached_tickers('sp500')
        
        self.assertEqual(result, ['AAPL', 'MSFT'])
    
    def test_get_cached_tickers_empty(self):
        """Test getting cached tickers when none exist"""
        # Mock empty result
        self.mock_cursor.fetchall.return_value = []
        
        result = self.manager._get_cached_tickers('sp500')
        
        self.assertIsNone(result)
    
    @patch.object(TickerUniverseManager, '_is_cache_valid')
    @patch.object(TickerUniverseManager, '_get_cached_tickers')
    @patch.object(TickerUniverseManager, '_fetch_sp500_tickers')
    @patch.object(TickerUniverseManager, '_cache_tickers')
    def test_get_sp500_tickers_use_cache(self, mock_cache, mock_fetch, mock_get_cached, mock_is_valid):
        """Test getting S&P 500 tickers using cache"""
        # Mock valid cache
        mock_is_valid.return_value = True
        mock_get_cached.return_value = ['AAPL', 'MSFT']
        
        result = self.manager.get_sp500_tickers()
        
        self.assertEqual(result, ['AAPL', 'MSFT'])
        mock_fetch.assert_not_called()  # Should not fetch from Wikipedia
        mock_cache.assert_not_called()  # Should not cache
    
    @patch.object(TickerUniverseManager, '_is_cache_valid')
    @patch.object(TickerUniverseManager, '_get_cached_tickers')
    @patch.object(TickerUniverseManager, '_fetch_sp500_tickers')
    @patch.object(TickerUniverseManager, '_cache_tickers')
    def test_get_sp500_tickers_fetch_new(self, mock_cache, mock_fetch, mock_get_cached, mock_is_valid):
        """Test getting S&P 500 tickers with fresh fetch"""
        # Mock invalid cache
        mock_is_valid.return_value = False
        mock_get_cached.return_value = None
        
        # Mock successful fetch
        mock_fetch.return_value = [
            {'symbol': 'AAPL', 'company_name': 'Apple Inc.', 'sector': 'Technology'}
        ]
        
        result = self.manager.get_sp500_tickers()
        
        self.assertEqual(result, ['AAPL'])
        mock_fetch.assert_called_once()  # Should fetch from Wikipedia
        mock_cache.assert_called_once()  # Should cache the result
    
    def test_get_combined_universe(self):
        """Test getting combined universe"""
        # Mock the individual ticker methods
        self.manager.get_sp500_tickers = Mock(return_value=['AAPL', 'MSFT'])
        self.manager.get_nasdaq100_tickers = Mock(return_value=['GOOGL', 'AAPL'])  # AAPL appears in both
        
        result = self.manager.get_combined_universe()
        
        # Should deduplicate and sort
        self.assertEqual(result, ['AAPL', 'GOOGL', 'MSFT'])
    
    def test_get_ticker_info(self):
        """Test getting ticker information"""
        # Mock ticker info data
        mock_row = ('sp500', 'AAPL', 'Apple Inc.', 'Technology', datetime.now(), True)
        self.mock_cursor.fetchone.return_value = mock_row
        
        result = self.manager.get_ticker_info('AAPL')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['company_name'], 'Apple Inc.')
        self.assertEqual(result['sector'], 'Technology')
    
    def test_get_universe_stats(self):
        """Test getting universe statistics"""
        # Mock statistics data
        mock_index_counts = [('sp500', 500), ('nasdaq100', 100)]
        mock_total_unique = 550
        mock_cache_status = [
            ('sp500', datetime.now(), 500),
            ('nasdaq100', datetime.now(), 100)
        ]
        
        self.mock_cursor.fetchall.side_effect = [mock_index_counts, [(mock_total_unique,)], mock_cache_status]
        
        result = self.manager.get_universe_stats()
        
        self.assertEqual(result['total_unique_symbols'], 550)
        self.assertEqual(result['index_counts']['sp500'], 500)
        self.assertEqual(result['index_counts']['nasdaq100'], 100)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions"""
    
    @patch('utils.ticker_universe.TickerUniverseManager')
    def test_convenience_functions(self, mock_manager_class):
        """Test convenience functions create manager and call methods"""
        from utils.data.ticker_universe import get_sp500_tickers, get_nasdaq100_tickers, get_combined_universe
        
        # Mock manager instance
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Test convenience functions
        get_sp500_tickers()
        mock_manager.get_sp500_tickers.assert_called_once()
        
        get_nasdaq100_tickers()
        mock_manager.get_nasdaq100_tickers.assert_called_once()
        
        get_combined_universe()
        mock_manager.get_combined_universe.assert_called_once()


if __name__ == '__main__':
    unittest.main()
