"""
Unit tests for Flex Strategy Reconstructor

Tests conversion from Flex Query CSV to multi-leg strategy events.
"""

import unittest
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.api.ib.flex_strategy_reconstructor import (
    normalize_flex_columns,
    classify_strategy_structure,
    flex_csv_to_strategies,
    convert_flex_strategies_to_tws_format
)


class TestFlexStrategyReconstructor(unittest.TestCase):
    """Test cases for Flex Strategy Reconstructor."""
    
    def test_normalize_flex_columns(self):
        """Test column name normalization."""
        df = pd.DataFrame({
            'Order ID': [1, 2],
            'Put/Call': ['P', 'C'],
            'Buy/Sell': ['BUY', 'SELL'],
            'Net Cash': [100, -100]
        })
        
        normalized = normalize_flex_columns(df)
        
        self.assertIn('OrderID', normalized.columns)
        self.assertIn('Put/Call', normalized.columns)
        self.assertIn('Buy/Sell', normalized.columns)
        self.assertIn('NetCash', normalized.columns)
    
    def test_classify_vertical_put_spread(self):
        """Test classification of vertical put spread."""
        df = pd.DataFrame({
            'Symbol': ['IWM', 'IWM'],
            'Expiry': [20251205, 20251205],
            'Strike': [237.0, 241.0],
            'Put/Call': ['P', 'P'],
            'Buy/Sell': ['Buy', 'SELL'],
            'Quantity': [1.0, -1.0],
            'Price': [0.21, 0.46],
            'NetCash': [-21.0, 46.0]
        })
        
        result = classify_strategy_structure(df)
        
        self.assertEqual(result['structure'], 'vertical')
        self.assertTrue(result['valid'])
        self.assertEqual(result['right'], 'P')
        self.assertEqual(result['strikes'], [237.0, 241.0])
        self.assertEqual(result['quantity'], 1)
        # Bull put: sell high (0.46), buy low (0.21) = 0.46 - 0.21 = 0.25 credit
        self.assertAlmostEqual(result['price'], 0.25, places=2)
    
    def test_classify_vertical_call_spread(self):
        """Test classification of vertical call spread."""
        df = pd.DataFrame({
            'Symbol': ['SPY', 'SPY'],
            'Expiry': [20251220, 20251220],
            'Strike': [450.0, 455.0],
            'Put/Call': ['C', 'C'],
            'Buy/Sell': ['Buy', 'SELL'],
            'Quantity': [1.0, -1.0],
            'Price': [5.0, 2.0],
            'NetCash': [-500.0, 200.0]
        })
        
        result = classify_strategy_structure(df)
        
        self.assertEqual(result['structure'], 'vertical')
        self.assertTrue(result['valid'])
        self.assertEqual(result['right'], 'C')
        # Bull call: buy low (5.0), sell high (2.0) = 5.0 - 2.0 = 3.0 debit
        self.assertAlmostEqual(result['price'], 3.0, places=2)
    
    def test_classify_iron_condor(self):
        """Test classification of iron condor."""
        df = pd.DataFrame({
            'Symbol': ['SPY', 'SPY', 'SPY', 'SPY'],
            'Expiry': [20251220, 20251220, 20251220, 20251220],
            'Strike': [440.0, 445.0, 450.0, 455.0],
            'Put/Call': ['P', 'P', 'C', 'C'],
            'Buy/Sell': ['Buy', 'SELL', 'SELL', 'Buy'],
            'Quantity': [1.0, -1.0, -1.0, 1.0],
            'Price': [1.0, 2.0, 2.0, 1.0]
        })
        
        result = classify_strategy_structure(df)
        
        self.assertEqual(result['structure'], 'iron_condor')
        self.assertTrue(result['valid'])
        self.assertEqual(result['put_strikes'], [440.0, 445.0])
        self.assertEqual(result['call_strikes'], [450.0, 455.0])
    
    def test_flex_csv_to_strategies_vertical_spread(self):
        """Test conversion of Flex CSV to strategy events."""
        df = pd.DataFrame({
            'OrderID': ['123', '123'],
            'DateTime': [pd.Timestamp('2025-12-03 20:46:37'), pd.Timestamp('2025-12-03 20:46:37')],
            'Symbol': ['IWM', 'IWM'],
            'Expiry': [20251210, 20251210],
            'Strike': [237.0, 241.0],
            'Put/Call': ['P', 'P'],
            'Buy/Sell': ['Buy', 'SELL'],
            'Quantity': [1.0, -1.0],
            'Price': [0.21, 0.46],
            'NetCash': [-21.0, 46.0],
            'Commission': [0.0, 0.0]
        })
        
        strategies = flex_csv_to_strategies(df)
        
        self.assertEqual(len(strategies), 1)
        strat = strategies[0]
        self.assertEqual(strat['OrderID'], '123')
        self.assertEqual(strat['Structure'], 'vertical')
        self.assertEqual(strat['Put/Call'], 'P')
        self.assertEqual(strat['Quantity'], 1)
        self.assertAlmostEqual(strat['Price'], 0.25, places=2)
    
    def test_classify_single_leg_filtered_out(self):
        """Test that multiple fills of the same option are filtered out (not a strategy)."""
        df = pd.DataFrame({
            'Symbol': ['QQQ', 'QQQ', 'QQQ'],
            'Expiry': [20251201, 20251201, 20251201],
            'Strike': [577.0, 577.0, 577.0],
            'Put/Call': ['P', 'P', 'P'],
            'Buy/Sell': ['Buy', 'Buy', 'Buy'],
            'Quantity': [2.0, 1.0, 1.0],
            'Price': [0.25, 0.25, 0.25],
            'NetCash': [-50.0, -25.0, -25.0]
        })
        
        result = classify_strategy_structure(df)
        
        self.assertEqual(result['structure'], 'single_leg')
        self.assertFalse(result['valid'])
    
    def test_convert_flex_strategies_to_tws_format(self):
        """Test conversion to TWS format."""
        flex_strategies = [
            {
                'OrderID': '123',
                'DateTime': pd.Timestamp('2025-12-03 20:46:37'),
                'Symbol': 'IWM',
                'Structure': 'vertical',
                'NumLegs': 2,
                'Expiry': 20251210,
                'Strikes': [237.0, 241.0],
                'Put/Call': 'P',
                'Quantity': 1,
                'Price': -0.25,
                'LowStrike': 237.0,
                'HighStrike': 241.0,
                'LowPrice': 0.21,
                'HighPrice': 0.46,
                'NetCash': 25.0,
                'Commission': 0.0,
                'Legs': [
                    {
                        'Symbol': 'IWM',
                        'Expiry': 20251210,
                        'Strike': 237.0,
                        'Put/Call': 'P',
                        'Buy/Sell': 'Buy',
                        'Quantity': 1.0,
                        'Price': 0.21
                    },
                    {
                        'Symbol': 'IWM',
                        'Expiry': 20251210,
                        'Strike': 241.0,
                        'Put/Call': 'P',
                        'Buy/Sell': 'SELL',
                        'Quantity': -1.0,
                        'Price': 0.46
                    }
                ]
            }
        ]
        
        tws_strategies = convert_flex_strategies_to_tws_format(flex_strategies)
        
        self.assertEqual(len(tws_strategies), 1)
        tws_strat = tws_strategies[0]
        self.assertEqual(tws_strat['OrderID'], '123')
        self.assertEqual(tws_strat['NumLegs'], 2)
        self.assertEqual(tws_strat['Underlying'], 'IWM')
        self.assertEqual(len(tws_strat['Legs']), 2)
        self.assertIn('SpreadType', tws_strat)
        self.assertEqual(tws_strat['Quantity'], 1)
        self.assertAlmostEqual(tws_strat['OpenPrice'], -0.25, places=2)


if __name__ == '__main__':
    unittest.main()

