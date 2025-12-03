"""
Unit tests for StrategyDetector class.

Tests strategy detection logic from raw IB API execution data.
"""

import unittest
from datetime import datetime
import pandas as pd
from utils.strategy_detector import StrategyDetector


class TestStrategyDetector(unittest.TestCase):
    """Test cases for StrategyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = StrategyDetector()
    
    def test_generate_strategy_id_bull_put_spread(self):
        """Test strategy ID generation for bull put spread."""
        legs = [
            {'Symbol': 'QQQ', 'Expiry': 20241205, 'Strike': 591, 'Put/Call': 'P'},
            {'Symbol': 'QQQ', 'Expiry': 20241205, 'Strike': 595, 'Put/Call': 'P'}
        ]
        strategy_id = self.detector.generate_strategy_id(legs, 'QQQ')
        self.assertEqual(strategy_id, "QQQ Dec05 595/591 Bull Put Spread")
    
    def test_generate_strategy_id_bull_call_spread(self):
        """Test strategy ID generation for bull call spread."""
        legs = [
            {'Symbol': 'SPY', 'Expiry': '2024-12-20', 'Strike': 450, 'Put/Call': 'C'},
            {'Symbol': 'SPY', 'Expiry': '2024-12-20', 'Strike': 455, 'Put/Call': 'C'}
        ]
        strategy_id = self.detector.generate_strategy_id(legs, 'SPY')
        self.assertEqual(strategy_id, "SPY Dec20 450/455 Bull Call Spread")
    
    def test_generate_strategy_id_insufficient_legs(self):
        """Test strategy ID generation with insufficient legs."""
        legs = [
            {'Symbol': 'QQQ', 'Expiry': 20241205, 'Strike': 591, 'Put/Call': 'P'}
        ]
        strategy_id = self.detector.generate_strategy_id(legs)
        self.assertEqual(strategy_id, "N/A")
    
    def test_generate_strategy_id_empty_legs(self):
        """Test strategy ID generation with empty legs."""
        strategy_id = self.detector.generate_strategy_id([])
        self.assertEqual(strategy_id, "N/A")
    
    def test_infer_leg_direction_bull_put_credit_spread(self):
        """Test leg direction inference for bull put credit spread."""
        execution_data = [
            {
                'ParentID': '725',
                'TradeID': '0000fb0a.6929afbd.02.01',
                'Date/Time': '2024-12-03 10:00:00',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 591,
                'Put/Call': 'P',
                'Price': 0.10
            },
            {
                'ParentID': '725',
                'TradeID': '0000fb0a.6929afbd.03.01',
                'Date/Time': '2024-12-03 10:00:00',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 595,
                'Put/Call': 'P',
                'Price': 0.35
            }
        ]
        
        directions = self.detector.infer_leg_direction_for_combo(execution_data, '725')
        
        # High strike (595) should be SELL, low strike (591) should be BUY for credit spread
        self.assertEqual(directions['0000fb0a.6929afbd.03.01'], 'SELL')
        self.assertEqual(directions['0000fb0a.6929afbd.02.01'], 'BUY')
    
    def test_infer_leg_direction_bull_call_debit_spread(self):
        """Test leg direction inference for bull call debit spread."""
        execution_data = [
            {
                'ParentID': '726',
                'TradeID': '0000fb0b.6929afbe.02.01',
                'Date/Time': '2024-12-03 11:00:00',
                'Symbol': 'SPY',
                'Expiry': 20241220,
                'Strike': 450,
                'Put/Call': 'C',
                'Price': 2.50
            },
            {
                'ParentID': '726',
                'TradeID': '0000fb0b.6929afbe.03.01',
                'Date/Time': '2024-12-03 11:00:00',
                'Symbol': 'SPY',
                'Expiry': 20241220,
                'Strike': 455,
                'Put/Call': 'C',
                'Price': 1.00
            }
        ]
        
        directions = self.detector.infer_leg_direction_for_combo(execution_data, '726')
        
        # Low strike (450) should be BUY, high strike (455) should be SELL for debit spread
        self.assertEqual(directions['0000fb0b.6929afbe.02.01'], 'BUY')
        self.assertEqual(directions['0000fb0b.6929afbe.03.01'], 'SELL')
    
    def test_infer_leg_direction_single_leg(self):
        """Test leg direction inference with single leg (should return empty)."""
        execution_data = [
            {
                'ParentID': '727',
                'TradeID': '0000fb0c.6929afbf.02.01',
                'Date/Time': '2024-12-03 12:00:00',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 591,
                'Put/Call': 'P',
                'Price': 0.10
            }
        ]
        
        directions = self.detector.infer_leg_direction_for_combo(execution_data, '727')
        self.assertEqual(directions, {})
    
    def test_summarize_vertical_put_spreads(self):
        """Test vertical put spread summarization."""
        df = pd.DataFrame([
            {
                'ParentID': '725',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 591,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.10,
                'Date/Time': '2024-12-03 10:00:00',
                'Buy/Sell': 'BUY'
            },
            {
                'ParentID': '725',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 595,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.35,
                'Date/Time': '2024-12-03 10:00:00',
                'Buy/Sell': 'SELL'
            }
        ])
        
        summary = self.detector.summarize_vertical_put_spreads(df)
        
        self.assertFalse(summary.empty)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]['ParentID'], '725')
        self.assertEqual(summary.iloc[0]['LowStrike'], 591)
        self.assertEqual(summary.iloc[0]['HighStrike'], 595)
        self.assertEqual(summary.iloc[0]['Quantity'], 2)
        # Spread price should be negative for credit spread: -(0.35 - 0.10) = -0.25
        self.assertAlmostEqual(summary.iloc[0]['PricePerSpread'], -0.25, places=2)
    
    def test_summarize_vertical_put_spreads_no_puts(self):
        """Test summarization with no put spreads."""
        df = pd.DataFrame([
            {
                'ParentID': '728',
                'Symbol': 'SPY',
                'Expiry': 20241220,
                'Strike': 450,
                'Put/Call': 'C',
                'Quantity': 2,
                'Price': 2.50,
                'Date/Time': '2024-12-03 11:00:00',
                'Buy/Sell': 'BUY'
            }
        ])
        
        summary = self.detector.summarize_vertical_put_spreads(df)
        self.assertTrue(summary.empty)
    
    def test_summarize_vertical_put_spreads_not_two_strikes(self):
        """Test summarization with spread that doesn't have exactly 2 strikes."""
        df = pd.DataFrame([
            {
                'ParentID': '729',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 591,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.10,
                'Date/Time': '2024-12-03 10:00:00',
                'Buy/Sell': 'BUY'
            }
        ])
        
        summary = self.detector.summarize_vertical_put_spreads(df)
        self.assertTrue(summary.empty)
    
    def test_group_executions_by_combo(self):
        """Test grouping executions by combo order."""
        df = pd.DataFrame([
            {
                'ParentID': '725',
                'OrderID': 725,
                'TradeID': '0000fb0a.6929afbd.02.01',
                'BAG_ID': '0000fb0a',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 591,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.10,
                'Date/Time': '2024-12-03 10:00:00',
                'DateTime': pd.to_datetime('2024-12-03 10:00:00'),
                'Buy/Sell': 'BUY',
                'NetCash': -20.0,
                'Commission': 1.0
            },
            {
                'ParentID': '725',
                'OrderID': 725,
                'TradeID': '0000fb0a.6929afbd.03.01',
                'BAG_ID': '0000fb0a',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 595,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.35,
                'Date/Time': '2024-12-03 10:00:00',
                'DateTime': pd.to_datetime('2024-12-03 10:00:00'),
                'Buy/Sell': 'SELL',
                'NetCash': 70.0,
                'Commission': 1.0
            }
        ])
        
        strategies = self.detector.group_executions_by_combo(df)
        
        self.assertEqual(len(strategies), 1)
        strategy = strategies[0]
        self.assertEqual(strategy['OrderID'], '725')
        self.assertEqual(strategy['BAG_ID'], '0000fb0a')
        self.assertIsNotNone(strategy['StrategyID'])
        self.assertIn('Bull Put Spread', strategy['StrategyID'])
        self.assertEqual(strategy['NumLegs'], 2)
        self.assertEqual(strategy['Underlying'], 'QQQ')
        self.assertEqual(strategy['Quantity'], 2)
    
    def test_group_executions_by_combo_empty_dataframe(self):
        """Test grouping with empty DataFrame."""
        df = pd.DataFrame()
        strategies = self.detector.group_executions_by_combo(df)
        self.assertEqual(strategies, [])
    
    def test_group_executions_by_combo_with_opening_and_closing(self):
        """Test grouping with opening and closing transactions."""
        df = pd.DataFrame([
            # Opening transaction
            {
                'ParentID': '725',
                'OrderID': 725,
                'TradeID': '0000fb0a.6929afbd.02.01',
                'BAG_ID': '0000fb0a',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 591,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.10,
                'Date/Time': '2024-12-03 10:00:00',
                'DateTime': pd.to_datetime('2024-12-03 10:00:00'),
                'Buy/Sell': 'BUY',
                'NetCash': -20.0,
                'Commission': 1.0
            },
            {
                'ParentID': '725',
                'OrderID': 725,
                'TradeID': '0000fb0a.6929afbd.03.01',
                'BAG_ID': '0000fb0a',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 595,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.35,
                'Date/Time': '2024-12-03 10:00:00',
                'DateTime': pd.to_datetime('2024-12-03 10:00:00'),
                'Buy/Sell': 'SELL',
                'NetCash': 70.0,
                'Commission': 1.0
            },
            # Closing transaction
            {
                'ParentID': '725',
                'OrderID': 725,
                'TradeID': '0000fb0a.692d915d.02.01',
                'BAG_ID': '0000fb0a',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 591,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.15,
                'Date/Time': '2024-12-04 14:00:00',
                'DateTime': pd.to_datetime('2024-12-04 14:00:00'),
                'Buy/Sell': 'SELL',
                'NetCash': -30.0,
                'Commission': 1.0
            },
            {
                'ParentID': '725',
                'OrderID': 725,
                'TradeID': '0000fb0a.692d915d.03.01',
                'BAG_ID': '0000fb0a',
                'Symbol': 'QQQ',
                'Expiry': 20241205,
                'Strike': 595,
                'Put/Call': 'P',
                'Quantity': 2,
                'Price': 0.40,
                'Date/Time': '2024-12-04 14:00:00',
                'DateTime': pd.to_datetime('2024-12-04 14:00:00'),
                'Buy/Sell': 'BUY',
                'NetCash': -80.0,
                'Commission': 1.0
            }
        ])
        
        strategies = self.detector.group_executions_by_combo(df)
        
        self.assertEqual(len(strategies), 1)
        strategy = strategies[0]
        self.assertIsNotNone(strategy['OpenPrice'])
        self.assertIsNotNone(strategy['ClosePrice'])
        # P&L should be calculated: (close_price - open_price) * quantity * 100
        # open_price = -0.25, close_price = -0.25 (same spread price)
        # P&L = (-0.25 - (-0.25)) * 2 * 100 = 0
        self.assertAlmostEqual(strategy['PnL'], 0, places=2)


if __name__ == '__main__':
    unittest.main()

