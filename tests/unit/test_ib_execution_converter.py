"""
Unit tests for IB Execution Converter

Tests conversion from IB API Fill objects to DataFrame format.
"""

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.api.ib.ib_execution_converter import ExecutionConverter


class TestExecutionConverter(unittest.TestCase):
    """Test cases for ExecutionConverter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = ExecutionConverter()
    
    def _create_mock_fill(
        self,
        sec_type: str,
        symbol: str,
        exec_id: str,
        order_id: int,
        side: str,  # 'B' or 'S'
        shares: float,
        price: float,
        expiry: str = None,
        strike: float = None,
        right: str = None,
        exchange: str = 'CBOE2',
        commission: float = 0.0,
        exec_time: datetime = None
    ) -> Mock:
        """
        Create a mock Fill object for testing.
        
        Args:
            sec_type: Security type ('BAG', 'OPT', etc.)
            symbol: Underlying symbol
            exec_id: Execution ID
            order_id: Order ID
            side: Execution side ('B' for Buy/BOT, 'S' for Sell/SLD)
            shares: Number of shares/contracts
            price: Execution price
            expiry: Expiry date (YYYYMMDD format) for options
            strike: Strike price for options
            right: 'P' or 'C' for options
            exchange: Exchange name
            commission: Commission amount
            exec_time: Execution time (datetime)
            
        Returns:
            Mock Fill object
        """
        if exec_time is None:
            exec_time = datetime(2025, 12, 4, 17, 58, 51, tzinfo=timezone.utc)
        
        # Create mock contract
        contract = Mock()
        contract.secType = sec_type
        contract.symbol = symbol
        contract.lastTradeDateOrContractMonth = expiry if expiry else ''
        contract.strike = strike if strike else 0.0
        contract.right = right if right else ''
        
        # Create mock execution
        execution = Mock()
        execution.execId = exec_id
        execution.orderId = order_id
        execution.orderRef = ''
        execution.side = side
        execution.shares = shares
        execution.price = price
        execution.exchange = exchange
        execution.acctNumber = 'U907499'
        
        # Create mock commission report
        commission_report = Mock()
        commission_report.commission = commission
        
        # Create mock fill
        fill = Mock()
        fill.contract = contract
        fill.execution = execution
        fill.commissionReport = commission_report
        fill.time = exec_time
        
        return fill
    
    def test_convert_bag_execution(self):
        """Test that BAG executions are captured but not included in DataFrame."""
        # Create BAG execution
        bag_fill = self._create_mock_fill(
            sec_type='BAG',
            symbol='RUT',
            exec_id='000243ef.6931997d.01.01',
            order_id=0,
            side='B',  # BOT
            shares=1.0,
            price=7.75
        )
        
        # Create option execution
        opt_fill = self._create_mock_fill(
            sec_type='OPT',
            symbol='RUT',
            exec_id='000243ef.6931997d.02.01',
            order_id=0,
            side='B',  # BOT (Buy)
            shares=1.0,
            price=41.5,
            expiry='20251218',
            strike=2545.0,
            right='C'
        )
        
        fills = [bag_fill, opt_fill]
        df = self.converter.convert_fills_to_dataframe(fills)
        
        # Should only have 1 row (option execution, BAG is skipped)
        self.assertEqual(len(df), 1)
        
        # BAG price should be captured
        self.assertIn('000243ef', self.converter.bag_prices)
        self.assertEqual(self.converter.bag_prices['000243ef'], 7.75 * 1.0 * 100)
    
    def test_convert_buy_execution(self):
        """Test conversion of Buy (BOT) execution."""
        fill = self._create_mock_fill(
            sec_type='OPT',
            symbol='RUT',
            exec_id='000243ef.6931997d.02.01',
            order_id=0,
            side='B',  # BOT (Buy)
            shares=1.0,
            price=41.5,
            expiry='20251218',
            strike=2545.0,
            right='C'
        )
        
        df = self.converter.convert_fills_to_dataframe([fill])
        
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        
        # Check Buy/Sell mapping
        self.assertEqual(row['Buy/Sell'], 'Buy')
        self.assertEqual(row['Quantity'], 1.0)  # Positive for Buy
        self.assertEqual(row['NetCash'], -4150.0)  # Negative for Buy (money out)
        self.assertEqual(row['Price'], 41.5)
        self.assertEqual(row['Symbol'], 'RUT')
        self.assertEqual(row['Expiry'], 20251218)
        self.assertEqual(row['Strike'], 2545.0)
        self.assertEqual(row['Put/Call'], 'C')
        self.assertEqual(row['TradeID'], '000243ef.6931997d.02.01')
        self.assertEqual(row['BAG_ID'], '000243ef')
        self.assertEqual(row['ParentID'], '000243ef')
        self.assertEqual(row['OrderID'], '000243ef')
    
    def test_convert_sell_execution(self):
        """Test conversion of Sell (SLD) execution."""
        fill = self._create_mock_fill(
            sec_type='OPT',
            symbol='RUT',
            exec_id='000243ef.6931997d.04.01',
            order_id=0,
            side='S',  # SLD (Sell)
            shares=1.0,
            price=13.28,
            expiry='20251212',
            strike=2585.0,
            right='C'
        )
        
        df = self.converter.convert_fills_to_dataframe([fill])
        
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        
        # Check Sell mapping
        self.assertEqual(row['Buy/Sell'], 'SELL')
        self.assertEqual(row['Quantity'], -1.0)  # Negative for Sell
        self.assertEqual(row['NetCash'], 1328.0)  # Positive for Sell (money in)
        self.assertEqual(row['Price'], 13.28)
        self.assertEqual(row['Symbol'], 'RUT')
        self.assertEqual(row['Expiry'], 20251212)
        self.assertEqual(row['Strike'], 2585.0)
        self.assertEqual(row['Put/Call'], 'C')
        self.assertEqual(row['TradeID'], '000243ef.6931997d.04.01')
        self.assertEqual(row['BAG_ID'], '000243ef')
        self.assertEqual(row['ParentID'], '000243ef')
        self.assertEqual(row['OrderID'], '000243ef')
    
    def test_convert_combo_order(self):
        """Test conversion of a complete combo order with multiple legs."""
        # Create fills matching the user's example
        fills = [
            # BAG execution (should be captured but not included)
            self._create_mock_fill(
                sec_type='BAG',
                symbol='RUT',
                exec_id='000243ef.6931997d.01.01',
                order_id=0,
                side='B',
                shares=1.0,
                price=7.75
            ),
            # Leg 1: Buy call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.02.01',
                order_id=0,
                side='B',  # BOT (Buy)
                shares=1.0,
                price=41.5,
                expiry='20251218',
                strike=2545.0,
                right='C'
            ),
            # Leg 2: Buy call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.03.01',
                order_id=0,
                side='B',  # BOT (Buy)
                shares=1.0,
                price=8.47,
                expiry='20251212',
                strike=2605.0,
                right='C'
            ),
            # Leg 3: Sell call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.04.01',
                order_id=0,
                side='S',  # SLD (Sell)
                shares=1.0,
                price=13.28,
                expiry='20251212',
                strike=2585.0,
                right='C'
            ),
            # Leg 4: Sell call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.05.01',
                order_id=0,
                side='S',  # SLD (Sell)
                shares=1.0,
                price=28.94,
                expiry='20251212',
                strike=2545.0,
                right='C'
            ),
        ]
        
        df = self.converter.convert_fills_to_dataframe(fills)
        
        # Should have 4 rows (BAG is excluded)
        self.assertEqual(len(df), 4)
        
        # Check BAG price was captured
        self.assertIn('000243ef', self.converter.bag_prices)
        self.assertEqual(self.converter.bag_prices['000243ef'], 7.75 * 1.0 * 100)
        
        # Check first leg (Buy)
        row1 = df[df['TradeID'] == '000243ef.6931997d.02.01'].iloc[0]
        self.assertEqual(row1['Buy/Sell'], 'Buy')
        self.assertEqual(row1['Quantity'], 1.0)
        self.assertEqual(row1['NetCash'], -4150.0)
        self.assertEqual(row1['Price'], 41.5)
        self.assertEqual(row1['Expiry'], 20251218)
        self.assertEqual(row1['Strike'], 2545.0)
        
        # Check second leg (Buy)
        row2 = df[df['TradeID'] == '000243ef.6931997d.03.01'].iloc[0]
        self.assertEqual(row2['Buy/Sell'], 'Buy')
        self.assertEqual(row2['Quantity'], 1.0)
        self.assertAlmostEqual(row2['NetCash'], -847.0, places=1)
        self.assertEqual(row2['Price'], 8.47)
        self.assertEqual(row2['Expiry'], 20251212)
        self.assertEqual(row2['Strike'], 2605.0)
        
        # Check third leg (Sell)
        row3 = df[df['TradeID'] == '000243ef.6931997d.04.01'].iloc[0]
        self.assertEqual(row3['Buy/Sell'], 'SELL')
        self.assertEqual(row3['Quantity'], -1.0)
        self.assertEqual(row3['NetCash'], 1328.0)
        self.assertEqual(row3['Price'], 13.28)
        self.assertEqual(row3['Expiry'], 20251212)
        self.assertEqual(row3['Strike'], 2585.0)
        
        # Check fourth leg (Sell)
        row4 = df[df['TradeID'] == '000243ef.6931997d.05.01'].iloc[0]
        self.assertEqual(row4['Buy/Sell'], 'SELL')
        self.assertEqual(row4['Quantity'], -1.0)
        self.assertEqual(row4['NetCash'], 2894.0)
        self.assertEqual(row4['Price'], 28.94)
        self.assertEqual(row4['Expiry'], 20251212)
        self.assertEqual(row4['Strike'], 2545.0)
        
        # All should have same ParentID and BAG_ID
        for _, row in df.iterrows():
            self.assertEqual(row['ParentID'], '000243ef')
            self.assertEqual(row['BAG_ID'], '000243ef')
            self.assertEqual(row['OrderID'], '000243ef')
    
    def test_convert_to_csv_format(self):
        """Test that DataFrame matches expected CSV format."""
        fills = [
            # Leg 1: Buy call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.02.01',
                order_id=0,
                side='B',
                shares=1.0,
                price=41.5,
                expiry='20251218',
                strike=2545.0,
                right='C'
            ),
            # Leg 2: Buy call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.03.01',
                order_id=0,
                side='B',
                shares=1.0,
                price=8.47,
                expiry='20251212',
                strike=2605.0,
                right='C'
            ),
            # Leg 3: Sell call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.04.01',
                order_id=0,
                side='S',
                shares=1.0,
                price=13.28,
                expiry='20251212',
                strike=2585.0,
                right='C'
            ),
            # Leg 4: Sell call
            self._create_mock_fill(
                sec_type='OPT',
                symbol='RUT',
                exec_id='000243ef.6931997d.05.01',
                order_id=0,
                side='S',
                shares=1.0,
                price=28.94,
                expiry='20251212',
                strike=2545.0,
                right='C'
            ),
        ]
        
        df = self.converter.convert_fills_to_dataframe(fills)
        
        # Remove Account column for comparison
        df_compare = df.drop(columns=['Account'], errors='ignore')
        
        # Check columns match expected CSV format
        expected_columns = [
            'OrderID', 'TradeID', 'BAG_ID', 'ParentID', 'Symbol', 'Expiry',
            'Strike', 'Put/Call', 'Buy/Sell', 'Quantity', 'Price', 'NetCash',
            'Commission', 'Date/Time', 'DateTime', 'Exchange'
        ]
        self.assertEqual(list(df_compare.columns), expected_columns)
        
        # Check first row matches expected values
        row1 = df_compare[df_compare['TradeID'] == '000243ef.6931997d.02.01'].iloc[0]
        self.assertEqual(row1['OrderID'], '000243ef')
        self.assertEqual(row1['TradeID'], '000243ef.6931997d.02.01')
        self.assertEqual(row1['BAG_ID'], '000243ef')
        self.assertEqual(row1['ParentID'], '000243ef')
        self.assertEqual(row1['Symbol'], 'RUT')
        self.assertEqual(row1['Expiry'], 20251218)
        self.assertEqual(row1['Strike'], 2545.0)
        self.assertEqual(row1['Put/Call'], 'C')
        self.assertEqual(row1['Buy/Sell'], 'Buy')
        self.assertEqual(row1['Quantity'], 1.0)
        self.assertEqual(row1['Price'], 41.5)
        self.assertAlmostEqual(row1['NetCash'], -4150.0, places=1)
        self.assertEqual(row1['Commission'], 0.0)
        self.assertEqual(row1['Exchange'], 'CBOE2')
    
    def test_filter_by_date(self):
        """Test date filtering."""
        fill1 = self._create_mock_fill(
            sec_type='OPT',
            symbol='RUT',
            exec_id='exec1',
            order_id=0,
            side='B',
            shares=1.0,
            price=10.0,
            expiry='20251201',
            strike=100.0,
            right='C',
            exec_time=datetime(2025, 12, 1, 10, 0, 0, tzinfo=timezone.utc)
        )
        
        fill2 = self._create_mock_fill(
            sec_type='OPT',
            symbol='RUT',
            exec_id='exec2',
            order_id=0,
            side='B',
            shares=1.0,
            price=10.0,
            expiry='20251205',
            strike=100.0,
            right='C',
            exec_time=datetime(2025, 12, 5, 10, 0, 0, tzinfo=timezone.utc)
        )
        
        # Filter to only include executions on or after 2025-12-03
        # Test with timezone-naive since_date (should be converted to UTC)
        since_date = datetime(2025, 12, 3)  # timezone-naive
        df = self.converter.convert_fills_to_dataframe([fill1, fill2], since_date=since_date)
        
        # Should only have 1 row (fill2)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['TradeID'], 'exec2')
        
        # Test with timezone-aware since_date
        since_date_aware = datetime(2025, 12, 3, tzinfo=timezone.utc)
        df2 = self.converter.convert_fills_to_dataframe([fill1, fill2], since_date=since_date_aware)
        
        # Should only have 1 row (fill2)
        self.assertEqual(len(df2), 1)
        self.assertEqual(df2.iloc[0]['TradeID'], 'exec2')
    
    def test_skip_non_option_executions(self):
        """Test that non-option executions are skipped."""
        fill1 = self._create_mock_fill(
            sec_type='STK',  # Stock
            symbol='AAPL',
            exec_id='exec1',
            order_id=0,
            side='B',
            shares=100.0,
            price=150.0
        )
        
        fill2 = self._create_mock_fill(
            sec_type='OPT',
            symbol='AAPL',
            exec_id='exec2',
            order_id=0,
            side='B',
            shares=1.0,
            price=5.0,
            expiry='20251220',
            strike=150.0,
            right='C'
        )
        
        df = self.converter.convert_fills_to_dataframe([fill1, fill2])
        
        # Should only have 1 row (option execution)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['TradeID'], 'exec2')


if __name__ == '__main__':
    unittest.main()

