"""
Tests for the Universe Data Updater
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.update_universe_data import UniverseDataUpdater


class TestUniverseDataUpdater(unittest.TestCase):
    """Test cases for UniverseDataUpdater"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the ticker manager
        self.mock_ticker_manager = Mock()
        self.mock_ticker_manager.get_combined_universe.return_value = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.mock_ticker_manager.get_cached_combined_universe.return_value = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create updater with mocked dependencies
        with patch('utils.update_universe_data.TickerUniverseManager') as mock_manager_class:
            mock_manager_class.return_value = self.mock_ticker_manager
            self.updater = UniverseDataUpdater(provider="alpaca", timeframe="1d")
    
    def test_initialization(self):
        """Test updater initialization"""
        self.assertEqual(self.updater.provider, "alpaca")
        self.assertEqual(self.updater.timeframe, "1d")
        self.assertIsNotNone(self.updater.ticker_manager)
    
    def test_initialization_invalid_provider(self):
        """Test initialization with invalid provider"""
        with self.assertRaises(ValueError):
            UniverseDataUpdater(provider="invalid", timeframe="1d")
    
    def test_initialization_invalid_timeframe(self):
        """Test initialization with invalid timeframe"""
        with self.assertRaises(ValueError):
            UniverseDataUpdater(provider="alpaca", timeframe="invalid")
    
    def test_get_universe_tickers_success(self):
        """Test getting universe tickers successfully"""
        tickers = self.updater.get_universe_tickers()
        self.assertEqual(tickers, ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        self.mock_ticker_manager.get_combined_universe.assert_called_once()
    
    def test_get_universe_tickers_with_force_refresh(self):
        """Test getting universe tickers with force refresh"""
        tickers = self.updater.get_universe_tickers(force_refresh=True)
        self.assertEqual(tickers, ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        self.mock_ticker_manager.refresh_all_indices.assert_called_once()
    
    def test_get_universe_tickers_fallback(self):
        """Test getting universe tickers with fallback to cache"""
        # Mock failure in get_combined_universe
        self.mock_ticker_manager.get_combined_universe.side_effect = Exception("Database error")
        
        tickers = self.updater.get_universe_tickers()
        self.assertEqual(tickers, ['AAPL', 'MSFT', 'GOOGL'])
        self.mock_ticker_manager.get_cached_combined_universe.assert_called_once()
    
    def test_get_universe_tickers_no_cache(self):
        """Test getting universe tickers when no cache available"""
        # Mock failure in both methods
        self.mock_ticker_manager.get_combined_universe.side_effect = Exception("Database error")
        self.mock_ticker_manager.get_cached_combined_universe.return_value = []
        
        tickers = self.updater.get_universe_tickers()
        self.assertEqual(tickers, [])
    
    @patch('utils.update_universe_data.fetch_from_alpaca')
    def test_fetch_ticker_data_alpaca_success(self, mock_fetch):
        """Test successful ticker data fetching from Alpaca"""
        # Mock successful fetch
        mock_df = Mock()
        mock_df.empty = False
        mock_fetch.return_value = mock_df
        
        result = self.updater.fetch_ticker_data("AAPL")
        self.assertTrue(result)
        mock_fetch.assert_called_once_with("AAPL", "max", "1d")
    
    @patch('utils.update_universe_data.fetch_from_alpaca')
    def test_fetch_ticker_data_alpaca_failure(self, mock_fetch):
        """Test failed ticker data fetching from Alpaca"""
        # Mock failed fetch
        mock_fetch.return_value = None
        
        result = self.updater.fetch_ticker_data("AAPL")
        self.assertFalse(result)
    
    @patch('utils.update_universe_data.fetch_from_alpaca')
    def test_fetch_ticker_data_alpaca_exception(self, mock_fetch):
        """Test ticker data fetching with exception"""
        # Mock exception
        mock_fetch.side_effect = Exception("Network error")
        
        result = self.updater.fetch_ticker_data("AAPL")
        self.assertFalse(result)
    
    @patch('utils.update_universe_data.fetch_from_ib')
    def test_fetch_ticker_data_ib_success(self, mock_fetch):
        """Test successful ticker data fetching from IBKR"""
        # Create IB updater
        with patch('utils.update_universe_data.TickerUniverseManager') as mock_manager_class:
            mock_manager_class.return_value = self.mock_ticker_manager
            ib_updater = UniverseDataUpdater(provider="ib", timeframe="1h")
        
        # Mock successful fetch
        mock_df = Mock()
        mock_df.empty = False
        mock_fetch.return_value = mock_df
        
        result = ib_updater.fetch_ticker_data("AAPL")
        self.assertTrue(result)
        mock_fetch.assert_called_once_with("AAPL", "max", "1h")
    
    def test_update_universe_data_basic(self):
        """Test basic universe data update"""
        # Mock successful fetch for all tickers
        with patch.object(self.updater, 'fetch_ticker_data') as mock_fetch:
            mock_fetch.return_value = True
            
            results = self.updater.update_universe_data(
                batch_size=2,
                delay_between_batches=0.1,
                delay_between_tickers=0.1,
                max_tickers=4
            )
        
        # Verify results
        self.assertEqual(results['total_tickers'], 4)
        self.assertEqual(results['successful'], 4)
        self.assertEqual(results['failed'], 0)
        self.assertEqual(results['success_rate'], 100.0)
        self.assertEqual(results['provider'], 'alpaca')
        self.assertEqual(results['timeframe'], '1d')
    
    def test_update_universe_data_with_failures(self):
        """Test universe data update with some failures"""
        # Mock mixed success/failure
        with patch.object(self.updater, 'fetch_ticker_data') as mock_fetch:
            mock_fetch.side_effect = [True, False, True, False, True]
            
            results = self.updater.update_universe_data(
                batch_size=2,
                delay_between_batches=0.1,
                delay_between_tickers=0.1,
                max_tickers=5
            )
        
        # Verify results
        self.assertEqual(results['total_tickers'], 5)
        self.assertEqual(results['successful'], 3)
        self.assertEqual(results['failed'], 2)
        self.assertEqual(results['success_rate'], 60.0)
        self.assertEqual(len(results['failed_symbols']), 2)
    
    def test_update_universe_data_no_tickers(self):
        """Test universe data update with no tickers"""
        # Mock empty ticker list
        self.mock_ticker_manager.get_combined_universe.return_value = []
        
        results = self.updater.update_universe_data()
        self.assertIn('error', results)
        self.assertEqual(results['error'], 'No tickers available')
    
    def test_update_universe_data_with_start_index(self):
        """Test universe data update with start index"""
        with patch.object(self.updater, 'fetch_ticker_data') as mock_fetch:
            mock_fetch.return_value = True
            
            results = self.updater.update_universe_data(
                start_from_index=2,
                max_tickers=2,
                batch_size=1,
                delay_between_batches=0.1,
                delay_between_tickers=0.1
            )
        
        # Should process tickers starting from index 2
        self.assertEqual(results['total_tickers'], 2)
        self.assertEqual(results['last_processed_index'], 3)  # 2 + 2 - 1
    
    def test_resume_update(self):
        """Test resume update functionality"""
        with patch.object(self.updater, 'update_universe_data') as mock_update:
            mock_update.return_value = {'success': True}
            
            results = self.updater.resume_update(
                last_processed_index=150,
                batch_size=20
            )
            
            # Should call update_universe_data with start_from_index=151
            mock_update.assert_called_once_with(start_from_index=151, batch_size=20)
    
    def test_update_universe_data_keyboard_interrupt(self):
        """Test handling of keyboard interrupt during update"""
        # Mock fetch to raise KeyboardInterrupt after first ticker
        with patch.object(self.updater, 'fetch_ticker_data') as mock_fetch:
            mock_fetch.side_effect = [True, KeyboardInterrupt()]
            
            results = self.updater.update_universe_data(
                batch_size=2,
                delay_between_batches=0.1,
                delay_between_tickers=0.1,
                max_tickers=5
            )
        
        # Should handle interruption gracefully
        self.assertEqual(results['total_tickers'], 5)
        self.assertEqual(results['successful'], 1)
        self.assertEqual(results['failed'], 0)


class TestUniverseDataUpdaterIntegration(unittest.TestCase):
    """Integration tests for UniverseDataUpdater"""
    
    @patch('utils.update_universe_data.TickerUniverseManager')
    @patch('utils.update_universe_data.fetch_from_alpaca')
    def test_full_integration_flow(self, mock_fetch, mock_manager_class):
        """Test full integration flow"""
        # Mock ticker manager
        mock_manager = Mock()
        mock_manager.get_combined_universe.return_value = ['AAPL', 'MSFT', 'GOOGL']
        mock_manager_class.return_value = mock_manager
        
        # Mock successful data fetch
        mock_df = Mock()
        mock_df.empty = False
        mock_fetch.return_value = mock_df
        
        # Create updater and run update
        updater = UniverseDataUpdater(provider="alpaca", timeframe="1d")
        results = updater.update_universe_data(
            batch_size=2,
            delay_between_batches=0.1,
            delay_between_tickers=0.1
        )
        
        # Verify integration
        self.assertEqual(results['total_tickers'], 3)
        self.assertEqual(results['successful'], 3)
        self.assertEqual(results['failed'], 0)
        self.assertEqual(mock_fetch.call_count, 3)


if __name__ == '__main__':
    unittest.main()
