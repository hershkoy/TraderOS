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
    
    def test_generate_short_strategy_string_bull_put_spread(self):
        """Test short strategy string generation for bull put spread."""
        legs = [
            {'Symbol': 'QQQ', 'Expiry': 20241205, 'Strike': 591, 'Put/Call': 'P', 'Buy/Sell': 'BUY'},
            {'Symbol': 'QQQ', 'Expiry': 20241205, 'Strike': 595, 'Put/Call': 'P', 'Buy/Sell': 'SELL'}
        ]
        short_str = self.detector.generate_short_strategy_string(legs, 'QQQ')
        self.assertEqual(short_str, "QQQ + Dec05 591P - Dec05 595P")
    
    def test_generate_short_strategy_string_bull_call_spread(self):
        """Test short strategy string generation for bull call spread."""
        legs = [
            {'Symbol': 'SPY', 'Expiry': '2024-12-20', 'Strike': 450, 'Put/Call': 'C', 'Buy/Sell': 'BUY'},
            {'Symbol': 'SPY', 'Expiry': '2024-12-20', 'Strike': 455, 'Put/Call': 'C', 'Buy/Sell': 'SELL'}
        ]
        short_str = self.detector.generate_short_strategy_string(legs, 'SPY')
        self.assertEqual(short_str, "SPY + Dec20 450C - Dec20 455C")
    
    def test_generate_short_strategy_string_complex_multi_leg(self):
        """Test short strategy string generation for complex multi-leg strategy."""
        legs = [
            {'Symbol': 'RUT', 'Expiry': 20251218, 'Strike': 2545.0, 'Put/Call': 'C', 'Buy/Sell': 'SELL'},
            {'Symbol': 'RUT', 'Expiry': 20251212, 'Strike': 2605.0, 'Put/Call': 'C', 'Buy/Sell': 'SELL'},
            {'Symbol': 'RUT', 'Expiry': 20251212, 'Strike': 2585.0, 'Put/Call': 'C', 'Buy/Sell': 'SELL'},
            {'Symbol': 'RUT', 'Expiry': 20251212, 'Strike': 2545.0, 'Put/Call': 'C', 'Buy/Sell': 'BUY'}
        ]
        short_str = self.detector.generate_short_strategy_string(legs, 'RUT')
        # Should be sorted by expiry and strike
        self.assertIn('RUT', short_str)
        self.assertIn('Dec12', short_str)
        self.assertIn('Dec18', short_str)
        self.assertIn('2545C', short_str)
        self.assertIn('2585C', short_str)
        self.assertIn('2605C', short_str)
    
    def test_generate_short_strategy_string_insufficient_legs(self):
        """Test short strategy string generation with insufficient legs."""
        legs = [
            {'Symbol': 'QQQ', 'Expiry': 20241205, 'Strike': 591, 'Put/Call': 'P', 'Buy/Sell': 'BUY'}
        ]
        short_str = self.detector.generate_short_strategy_string(legs)
        self.assertEqual(short_str, "N/A")
    
    def test_generate_short_strategy_string_empty_legs(self):
        """Test short strategy string generation with empty legs."""
        short_str = self.detector.generate_short_strategy_string([])
        self.assertEqual(short_str, "N/A")
    
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
        # Should have short strategy string
        self.assertIsNotNone(strategy.get('ShortStrategyString'))
        self.assertIn('QQQ', strategy['ShortStrategyString'])
        self.assertEqual(strategy['NumLegs'], 2)
        self.assertEqual(strategy['Underlying'], 'QQQ')
        self.assertEqual(strategy['Quantity'], 2)
        # Price should be the total price of the entire combo (opening)
        # For credit spread: open_price = -0.25, quantity = 2
        # Price = -0.25 * 2 * 100 = -50.0 (credit received)
        self.assertAlmostEqual(strategy['Price'], -50.0, places=2)
        self.assertEqual(strategy['Price'], strategy['BuyPrice'])
    
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
        # Price should be the total price of the entire combo (opening)
        # open_price = -0.25, quantity = 2
        # Price = -0.25 * 2 * 100 = -50.0 (credit received)
        self.assertAlmostEqual(strategy['Price'], -50.0, places=2)
        self.assertEqual(strategy['Price'], strategy['BuyPrice'])
    
    def test_summarize_vertical_call_spreads(self):
        """Test vertical call spread summarization."""
        df = pd.DataFrame([
            {
                'ParentID': '730',
                'Symbol': 'SPY',
                'Expiry': 20241220,
                'Strike': 450,
                'Put/Call': 'C',
                'Quantity': 2,
                'Price': 2.50,
                'Date/Time': '2024-12-03 11:00:00',
                'Buy/Sell': 'BUY'
            },
            {
                'ParentID': '730',
                'Symbol': 'SPY',
                'Expiry': 20241220,
                'Strike': 455,
                'Put/Call': 'C',
                'Quantity': 2,
                'Price': 1.00,
                'Date/Time': '2024-12-03 11:00:00',
                'Buy/Sell': 'SELL'
            }
        ])
        
        summary = self.detector.summarize_vertical_call_spreads(df)
        
        self.assertFalse(summary.empty)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]['ParentID'], '730')
        self.assertEqual(summary.iloc[0]['LowStrike'], 450)
        self.assertEqual(summary.iloc[0]['HighStrike'], 455)
        self.assertEqual(summary.iloc[0]['Quantity'], 2)
        # Spread price should be positive for debit spread: 2.50 - 1.00 = 1.50
        self.assertAlmostEqual(summary.iloc[0]['PricePerSpread'], 1.50, places=2)
    
    def test_summarize_vertical_call_spreads_no_calls(self):
        """Test summarization with no call spreads."""
        df = pd.DataFrame([
            {
                'ParentID': '731',
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
        
        summary = self.detector.summarize_vertical_call_spreads(df)
        self.assertTrue(summary.empty)
    
    def test_group_executions_by_combo_call_spread(self):
        """Test grouping executions by combo order for call spread."""
        df = pd.DataFrame([
            {
                'ParentID': '730',
                'OrderID': 730,
                'TradeID': '0000fb0d.6929afbf.02.01',
                'BAG_ID': '0000fb0d',
                'Symbol': 'SPY',
                'Expiry': 20241220,
                'Strike': 450,
                'Put/Call': 'C',
                'Quantity': 2,
                'Price': 2.50,
                'Date/Time': '2024-12-03 11:00:00',
                'DateTime': pd.to_datetime('2024-12-03 11:00:00'),
                'Buy/Sell': 'BUY',
                'NetCash': -500.0,
                'Commission': 1.0
            },
            {
                'ParentID': '730',
                'OrderID': 730,
                'TradeID': '0000fb0d.6929afbf.03.01',
                'BAG_ID': '0000fb0d',
                'Symbol': 'SPY',
                'Expiry': 20241220,
                'Strike': 455,
                'Put/Call': 'C',
                'Quantity': 2,
                'Price': 1.00,
                'Date/Time': '2024-12-03 11:00:00',
                'DateTime': pd.to_datetime('2024-12-03 11:00:00'),
                'Buy/Sell': 'SELL',
                'NetCash': 200.0,
                'Commission': 1.0
            }
        ])
        
        strategies = self.detector.group_executions_by_combo(df)
        
        self.assertEqual(len(strategies), 1)
        strategy = strategies[0]
        self.assertEqual(strategy['OrderID'], '730')
        self.assertEqual(strategy['BAG_ID'], '0000fb0d')
        self.assertIsNotNone(strategy['StrategyID'])
        self.assertIn('Bull Call Spread', strategy['StrategyID'])
        # Should have short strategy string
        self.assertIsNotNone(strategy.get('ShortStrategyString'))
        self.assertIn('SPY', strategy['ShortStrategyString'])
        self.assertEqual(strategy['NumLegs'], 2)
        self.assertEqual(strategy['Underlying'], 'SPY')
        self.assertEqual(strategy['Quantity'], 2)
        # Price should be the total price of the entire combo (opening)
        # For debit spread: open_price = 1.50, quantity = 2
        # Price = 1.50 * 2 * 100 = 300.0 (debit paid)
        self.assertAlmostEqual(strategy['Price'], 300.0, places=2)
        self.assertEqual(strategy['Price'], strategy['BuyPrice'])
    
    def test_group_executions_by_combo_complex_multi_leg(self):
        """Test grouping complex multi-leg strategy (like RUT order with 4 legs)."""
        df = pd.DataFrame([
            {
                'ParentID': '000243ef',
                'OrderID': '000243ef',
                'TradeID': '000243ef.6931997d.02.01',
                'BAG_ID': '000243ef',
                'Symbol': 'RUT',
                'Expiry': 20251218,
                'Strike': 2545.0,
                'Put/Call': 'C',
                'Quantity': -1.0,
                'Price': 41.5,
                'Date/Time': '2025-12-04 17:58:51',
                'DateTime': pd.to_datetime('2025-12-04 17:58:51'),
                'Buy/Sell': 'SELL',
                'NetCash': 4150.0,
                'Commission': 0.0
            },
            {
                'ParentID': '000243ef',
                'OrderID': '000243ef',
                'TradeID': '000243ef.6931997d.03.01',
                'BAG_ID': '000243ef',
                'Symbol': 'RUT',
                'Expiry': 20251212,
                'Strike': 2605.0,
                'Put/Call': 'C',
                'Quantity': 1.0,
                'Price': 8.47,
                'Date/Time': '2025-12-04 17:58:51',
                'DateTime': pd.to_datetime('2025-12-04 17:58:51'),
                'Buy/Sell': 'BUY',  # Should be BUY, not SELL
                'NetCash': -847.0,  # Negative NetCash for BUY
                'Commission': 0.0
            },
            {
                'ParentID': '000243ef',
                'OrderID': '000243ef',
                'TradeID': '000243ef.6931997d.04.01',
                'BAG_ID': '000243ef',
                'Symbol': 'RUT',
                'Expiry': 20251212,
                'Strike': 2585.0,
                'Put/Call': 'C',
                'Quantity': -1.0,
                'Price': 13.28,
                'Date/Time': '2025-12-04 17:58:51',
                'DateTime': pd.to_datetime('2025-12-04 17:58:51'),
                'Buy/Sell': 'SELL',
                'NetCash': 1328.0,
                'Commission': 0.0
            },
            {
                'ParentID': '000243ef',
                'OrderID': '000243ef',
                'TradeID': '000243ef.6931997d.05.01',
                'BAG_ID': '000243ef',
                'Symbol': 'RUT',
                'Expiry': 20251212,
                'Strike': 2545.0,
                'Put/Call': 'C',
                'Quantity': 1.0,
                'Price': 28.94,
                'Date/Time': '2025-12-04 17:58:51',
                'DateTime': pd.to_datetime('2025-12-04 17:58:51'),
                'Buy/Sell': 'BUY',
                'NetCash': -2894.0,
                'Commission': 0.0
            }
        ])
        
        strategies = self.detector.group_executions_by_combo(df)
        
        # Should detect this as a complex multi-leg strategy (not a simple 2-leg spread)
        # and use the fallback method
        self.assertEqual(len(strategies), 1)
        strategy = strategies[0]
        self.assertEqual(strategy['OrderID'], '000243ef')
        self.assertEqual(strategy['BAG_ID'], '000243ef')
        self.assertEqual(strategy['Underlying'], 'RUT')
        # Should have 4 legs
        self.assertEqual(strategy['NumLegs'], 4)
        # Should have a strategy ID generated
        self.assertIsNotNone(strategy['StrategyID'])
        # Should have short strategy string
        self.assertIsNotNone(strategy.get('ShortStrategyString'))
        short_str = strategy.get('ShortStrategyString', '')
        self.assertIn('RUT', short_str)
        # Verify the correct strategy string format: RUT + Dec12 2545C - Dec12 2585C + Dec12 2605C - Dec18 2545C
        self.assertIn('+ Dec12 2545C', short_str)  # BUY Dec12 2545C
        self.assertIn('- Dec12 2585C', short_str)  # SELL Dec12 2585C
        self.assertIn('+ Dec12 2605C', short_str)  # BUY Dec12 2605C
        self.assertIn('- Dec18 2545C', short_str)  # SELL Dec18 2545C
        # Should have leg descriptions
        self.assertEqual(len(strategy['Legs']), 4)
        # Verify leg directions are correct
        legs_str = ' '.join(strategy['Legs'])
        self.assertIn('BUY', legs_str)
        self.assertIn('SELL', legs_str)
        # Price should be the total price of the entire combo (opening)
        # Total buy price = sum of all BUY NetCash = -2894.0 + (-847.0) = -3741.0
        self.assertAlmostEqual(strategy['Price'], -3741.0, places=2)
        self.assertEqual(strategy['Price'], strategy['BuyPrice'])
    
    def test_group_executions_by_combo_with_bag_price(self):
        """Test grouping executions with BAG price from metadata."""
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
                'Quantity': 1,
                'Price': 0.10,
                'Date/Time': '2024-12-03 10:00:00',
                'DateTime': pd.to_datetime('2024-12-03 10:00:00'),
                'Buy/Sell': 'BUY',
                'NetCash': -10.0,
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
                'Quantity': 1,
                'Price': 0.35,
                'Date/Time': '2024-12-03 10:00:00',
                'DateTime': pd.to_datetime('2024-12-03 10:00:00'),
                'Buy/Sell': 'SELL',
                'NetCash': 35.0,
                'Commission': 1.0
            }
        ])
        
        # Add BAG price to DataFrame metadata
        # BAG price = 7.75 * 100 = 775 (for 1 contract)
        if hasattr(df, 'attrs'):
            df.attrs['bag_prices'] = {'725': 775.0}
        
        strategies = self.detector.group_executions_by_combo(df)
        
        self.assertEqual(len(strategies), 1)
        strategy = strategies[0]
        # Price should use BAG price if available
        self.assertAlmostEqual(strategy['Price'], 775.0, places=2)
        # BuyPrice should still be calculated from legs
        self.assertAlmostEqual(strategy['BuyPrice'], -25.0, places=2)  # -0.25 * 1 * 100
        # Price and BuyPrice should be different when BAG price is used
        self.assertNotEqual(strategy['Price'], strategy['BuyPrice'])


if __name__ == '__main__':
    unittest.main()

