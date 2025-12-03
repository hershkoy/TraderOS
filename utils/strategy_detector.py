"""
Strategy Detection from IB API Execution Data

This module provides a class to detect and group multi-leg option strategies
from raw IB API execution data.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyDetector:
    """
    Detects and groups multi-leg option strategies from IB API execution data.
    
    Handles:
    - Leg direction inference (BUY/SELL) for combo orders
    - Vertical spread detection and summarization
    - Strategy grouping by combo order
    - Strategy ID generation
    """
    
    def __init__(self):
        """Initialize the strategy detector."""
        pass
    
    def infer_leg_direction_for_combo(self, execution_data: List[Dict[str, Any]], parent_id: str) -> Dict[str, str]:
        """
        Infer the actual leg direction (BUY/SELL) for combo order legs based on strike prices and spread structure.
        
        For credit spreads (bull put):
        - Higher strike put = SELL (receive premium)
        - Lower strike put = BUY (pay premium)
        
        For debit spreads (bull call):
        - Lower strike call = BUY (pay premium)
        - Higher strike call = SELL (receive premium)
        
        Args:
            execution_data: List of execution records (partial, being built)
            parent_id: Parent ID of the combo order
            
        Returns:
            Dictionary mapping TradeID to inferred Buy/Sell direction
        """
        # Get all executions for this combo
        combo_executions = [e for e in execution_data if e.get('ParentID') == parent_id]
        
        if len(combo_executions) < 2:
            # Not a multi-leg combo, return empty dict (use original side)
            return {}
        
        # Group by fill ID extracted from TradeID to ensure we only compare legs from the same fill
        # TradeID format: ParentID.FillID.LegID.ExecutionID (e.g., 0000fb0a.6929afbd.02.01)
        # Multiple fills can have the same ParentID and timestamp, so we need to group by FillID
        def extract_fill_id(trade_id: str) -> str:
            """Extract fill ID from TradeID format: ParentID.FillID.LegID.ExecutionID"""
            if not trade_id:
                return ''
            parts = trade_id.split('.')
            if len(parts) >= 2:
                return parts[1]  # FillID is the second part
            return trade_id  # Fallback to full TradeID if format is unexpected
        
        by_fill_id = {}
        for exec_record in combo_executions:
            trade_id = exec_record.get('TradeID', '')
            fill_id = extract_fill_id(trade_id)
            # Use combination of fill_id and timestamp as key (in case fill_id extraction fails)
            timestamp = exec_record.get('Date/Time', '')
            fill_key = (fill_id, timestamp) if fill_id else timestamp
            if fill_key not in by_fill_id:
                by_fill_id[fill_key] = []
            by_fill_id[fill_key].append(exec_record)
        
        leg_directions = {}
        
        # Process each fill group separately
        for fill_key, group_executions in by_fill_id.items():
            if len(group_executions) < 2:
                continue
            
            # Further group by expiry and symbol to ensure we're comparing the same spread
            by_expiry_symbol = {}
            for exec_record in group_executions:
                key = (exec_record.get('Expiry'), exec_record.get('Symbol'))
                if key not in by_expiry_symbol:
                    by_expiry_symbol[key] = []
                by_expiry_symbol[key].append(exec_record)
            
            # Process each expiry/symbol group within this timestamp
            for (expiry, symbol), spread_executions in by_expiry_symbol.items():
                if len(spread_executions) < 2:
                    continue
                
                # Group by Put/Call type
                puts = [e for e in spread_executions if e.get('Put/Call') == 'P']
                calls = [e for e in spread_executions if e.get('Put/Call') == 'C']
                
                # Process PUT spreads (typically credit spreads - bull put)
                if len(puts) >= 2:
                    # Sort by strike
                    puts_sorted = sorted(puts, key=lambda x: x.get('Strike', 0))
                    low_strike_put = puts_sorted[0]
                    high_strike_put = puts_sorted[-1]
                    
                    low_strike = low_strike_put.get('Strike', 0)
                    high_strike = high_strike_put.get('Strike', 0)
                    
                    if low_strike >= high_strike:
                        # Invalid spread, skip
                        continue
                    
                    # Get prices (absolute values)
                    high_strike_price = abs(high_strike_put.get('Price', 0))
                    low_strike_price = abs(low_strike_put.get('Price', 0))
                    
                    # For credit spreads (most common for puts):
                    # - High strike put premium > low strike put premium (you receive more for selling high strike)
                    # Pattern: SELL high strike (receive premium), BUY low strike (pay premium)
                    # This is the typical bull put credit spread
                    
                    # For debit spreads (less common for puts):
                    # - Low strike put premium > high strike put premium
                    # Pattern: BUY high strike, SELL low strike
                    
                    # Determine spread type based on price comparison
                    # Credit spreads are more common for puts, so default to that if prices are equal
                    if high_strike_price >= low_strike_price:
                        # Credit spread: SELL high, BUY low
                        leg_directions[high_strike_put.get('TradeID')] = 'SELL'
                        leg_directions[low_strike_put.get('TradeID')] = 'BUY'
                    else:
                        # Debit spread: BUY high, SELL low (less common)
                        leg_directions[high_strike_put.get('TradeID')] = 'BUY'
                        leg_directions[low_strike_put.get('TradeID')] = 'SELL'
                
                # Process CALL spreads (typically debit spreads - bull call)
                if len(calls) >= 2:
                    # Sort by strike
                    calls_sorted = sorted(calls, key=lambda x: x.get('Strike', 0))
                    low_strike_call = calls_sorted[0]
                    high_strike_call = calls_sorted[-1]
                    
                    low_strike = low_strike_call.get('Strike', 0)
                    high_strike = high_strike_call.get('Strike', 0)
                    
                    if low_strike >= high_strike:
                        # Invalid spread, skip
                        continue
                    
                    # Get prices (absolute values)
                    low_strike_price = abs(low_strike_call.get('Price', 0))
                    high_strike_price = abs(high_strike_call.get('Price', 0))
                    
                    # For debit spreads (most common for calls):
                    # - Low strike call premium > high strike call premium (you pay more for buying low strike)
                    # Pattern: BUY low strike (pay premium), SELL high strike (receive premium)
                    # This is the typical bull call debit spread
                    
                    # For credit spreads (less common for calls):
                    # - High strike call premium > low strike call premium
                    # Pattern: SELL low strike, BUY high strike
                    
                    # Determine spread type based on price comparison
                    # Debit spreads are more common for calls, so default to that if prices are equal
                    if low_strike_price >= high_strike_price:
                        # Debit spread: BUY low, SELL high
                        leg_directions[low_strike_call.get('TradeID')] = 'BUY'
                        leg_directions[high_strike_call.get('TradeID')] = 'SELL'
                    else:
                        # Credit spread: SELL low, BUY high (less common)
                        leg_directions[low_strike_call.get('TradeID')] = 'SELL'
                        leg_directions[high_strike_call.get('TradeID')] = 'BUY'
        
        return leg_directions
    
    def summarize_vertical_put_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        From a TWS executions DataFrame, detect vertical put credit spreads per combo (ParentID) 
        and timestamp, and return IB-style summaries: quantity, price per spread, etc.
        
        Args:
            df: DataFrame with execution data from fetch_tws_executions()
            
        Returns:
            DataFrame with spread-level summaries
        """
        results = []
        
        for parent_id, combo in df.groupby("ParentID"):
            # Only option combos with puts
            if combo["Put/Call"].nunique() != 1 or combo["Put/Call"].iloc[0] != "P":
                continue
            
            symbol = combo["Symbol"].iloc[0]
            expiry = combo["Expiry"].iloc[0]
            
            # Group by timestamp (open / close events)
            for dt, group in combo.groupby("Date/Time"):
                strikes = sorted(group["Strike"].unique())
                if len(strikes) != 2:
                    # Not a simple 2-leg vertical spread
                    continue
                
                low_strike, high_strike = strikes
                
                # Average price per strike at this timestamp
                low_group = group.loc[group["Strike"] == low_strike]
                high_group = group.loc[group["Strike"] == high_strike]
                
                avg_low = low_group["Price"].mean()
                avg_high = high_group["Price"].mean()
                
                # Total contracts per strike (use absolute value since Quantity can be negative)
                contracts_low = low_group["Quantity"].abs().sum()
                contracts_high = high_group["Quantity"].abs().sum()
                
                qty_spreads = int(min(contracts_low, contracts_high))
                
                # Bull put credit: net price is high - low, shown as negative for credit
                # For credit spreads: we receive premium for high strike, pay premium for low strike
                # Spread price = -(high_strike_price - low_strike_price) = low_strike_price - high_strike_price
                spread_price = -(avg_high - avg_low)
                
                # Determine if this is opening or closing based on Buy/Sell
                # For credit spreads, opening is typically SELL (we receive credit)
                # But we'll use time ordering instead - earliest = open, later = close
                buy_sell = group["Buy/Sell"].iloc[0]  # All should be the same in a combo
                
                results.append({
                    "ParentID": parent_id,
                    "Symbol": symbol,
                    "Expiry": expiry,
                    "LowStrike": low_strike,
                    "HighStrike": high_strike,
                    "DateTime": pd.to_datetime(dt) if isinstance(dt, str) else dt,
                    "Date/Time": dt,
                    "Quantity": qty_spreads,
                    "PricePerSpread": spread_price,
                    "Buy/Sell": buy_sell,
                    "LowStrikePrice": avg_low,
                    "HighStrikePrice": avg_high,
                    "ContractsLow": contracts_low,
                    "ContractsHigh": contracts_high,
                })
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        logger.debug(f"Summarized {len(result_df)} spread events from {df['ParentID'].nunique()} combos")
        return result_df
    
    def generate_strategy_id(self, legs: List[Dict[str, Any]], underlying: str = None) -> str:
        """
        Generate a human-readable strategy ID from legs.
        Format: "SYMBOL Expiry Strike1/Strike2 SpreadType"
        Example: "QQQ Dec05 595/591 Bull Put Spread"
        
        Args:
            legs: List of leg dictionaries with Symbol, Expiry, Strike, Put/Call, Buy/Sell
            underlying: Optional underlying symbol (if not provided, extracted from legs)
            
        Returns:
            Strategy ID string
        """
        if not legs or len(legs) < 2:
            return "N/A"
        
        # Get unique values from legs
        symbols = set(leg.get('Symbol', '') for leg in legs if leg.get('Symbol'))
        expiries = set(leg.get('Expiry', '') for leg in legs if leg.get('Expiry'))
        strikes = sorted(set(leg.get('Strike', 0) for leg in legs if leg.get('Strike')), reverse=True)
        put_calls = set(leg.get('Put/Call', '') for leg in legs if leg.get('Put/Call'))
        
        if not symbols or not expiries or len(strikes) < 2:
            return "N/A"
        
        # Get symbol
        symbol = underlying or list(symbols)[0]
        
        # Format expiry (convert YYYYMMDD to "Dec05" format)
        expiry = list(expiries)[0]
        expiry_str = ""
        if isinstance(expiry, (int, float)) and expiry > 0:
            expiry_num = str(int(expiry))
            if len(expiry_num) == 8:
                try:
                    expiry_date = datetime.strptime(expiry_num, '%Y%m%d')
                    expiry_str = expiry_date.strftime('%b%d')  # Dec05 format
                except:
                    expiry_str = expiry_num
            else:
                expiry_str = str(expiry)
        elif isinstance(expiry, str):
            # Try to parse if it's in YYYY-MM-DD or YYYYMMDD format
            try:
                # Try YYYY-MM-DD format first
                if '-' in expiry:
                    expiry_date = pd.to_datetime(expiry)
                # Try YYYYMMDD format
                elif len(expiry) == 8 and expiry.isdigit():
                    expiry_date = datetime.strptime(expiry, '%Y%m%d')
                else:
                    expiry_date = pd.to_datetime(expiry)
                expiry_str = expiry_date.strftime('%b%d')  # Dec05 format
            except:
                expiry_str = expiry
        else:
            expiry_str = str(expiry)
        
        # Get strikes (high/low)
        high_strike = strikes[0]
        low_strike = strikes[-1]
        
        # Determine option type (Put or Call)
        option_type = list(put_calls)[0] if put_calls else "P"
        
        # Determine spread type
        # For puts: High/Low = Bull Put Spread, Low/High = Bear Put Spread
        # For calls: Low/High = Bull Call Spread, High/Low = Bear Call Spread
        if option_type == 'P':
            # Put spreads
            if high_strike > low_strike:
                spread_type = "Bull Put Spread"
                strike_str = f"{int(high_strike)}/{int(low_strike)}"
            else:
                spread_type = "Bear Put Spread"
                strike_str = f"{int(low_strike)}/{int(high_strike)}"
        else:
            # Call spreads
            if low_strike < high_strike:
                spread_type = "Bull Call Spread"
                strike_str = f"{int(low_strike)}/{int(high_strike)}"
            else:
                spread_type = "Bear Call Spread"
                strike_str = f"{int(high_strike)}/{int(low_strike)}"
        
        return f"{symbol} {expiry_str} {strike_str} {spread_type}"
    
    def group_executions_by_combo(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Group TWS API executions by parentId (combo order ID) to match IB's Trade History Summary.
        Detects vertical put credit spreads and groups them correctly with quantity and price.
        
        Args:
            df: DataFrame with execution data from fetch_tws_executions()
            
        Returns:
            List of strategy dictionaries (same format as group_multi_leg_strategies)
        """
        if df.empty:
            return []
        
        logger.info("Grouping TWS executions by combo order (parentId)...")
        
        # First, summarize vertical put spreads
        spread_summary = self.summarize_vertical_put_spreads(df)
        
        if spread_summary.empty:
            logger.warning("No vertical put spreads detected, falling back to raw leg grouping")
            return self._group_executions_raw(df)
        
        # Group spreads by ParentID to identify opening and closing
        strategies = []
        
        for parent_id, combo_spreads in spread_summary.groupby('ParentID'):
            # Sort by DateTime to identify opening (earliest) and closing (latest)
            combo_spreads = combo_spreads.sort_values('DateTime')
            
            # Get spread details
            symbol = combo_spreads['Symbol'].iloc[0]
            expiry = combo_spreads['Expiry'].iloc[0]
            low_strike = combo_spreads['LowStrike'].iloc[0]
            high_strike = combo_spreads['HighStrike'].iloc[0]
            quantity = combo_spreads['Quantity'].iloc[0]  # Should be same for all events
            
            # Format expiry
            if isinstance(expiry, (int, float)) and expiry > 0:
                expiry_str = str(int(expiry))
                if len(expiry_str) == 8:
                    expiry_str = f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"
            else:
                expiry_str = str(expiry)
            
            # Build leg description (IB-style: BUY low strike, SELL high strike for bull put)
            legs_desc = [
                f"BUY {quantity} x {symbol} {expiry_str} {low_strike}P",
                f"SELL {quantity} x {symbol} {expiry_str} {high_strike}P"
            ]
            
            # Identify opening and closing
            opening = combo_spreads.iloc[0]
            closing = combo_spreads.iloc[-1] if len(combo_spreads) > 1 else None
            
            open_price = opening['PricePerSpread']
            close_price = closing['PricePerSpread'] if closing is not None else None
            
            # For credit spreads:
            # Opening: BOT (buy the spread, receive credit) - price is negative (e.g., -0.25)
            # Closing: SLD (sell the spread, pay to close) - price is negative (e.g., -0.53 for loss, -0.02 for profit)
            # The spread action is determined by time: earliest = opening (BOT), later = closing (SLD)
            
            # Calculate P&L correctly for credit spreads
            # P&L = (close_price - open_price) * quantity * 100
            # For loss: open_price = -0.25, close_price = -0.53
            # P&L = (-0.53 - (-0.25)) * 2 * 100 = (-0.28) * 2 * 100 = -56 (loss of $56)
            # For profit: open_price = -0.25, close_price = -0.02
            # P&L = (-0.02 - (-0.25)) * 2 * 100 = (0.23) * 2 * 100 = 46 (profit of $46)
            if close_price is not None:
                strategy_pnl = (close_price - open_price) * quantity * 100  # *100 for contract multiplier
            else:
                strategy_pnl = 0  # Still open
            
            # Calculate NetCash from prices
            # Opening: receive credit (negative price * quantity * 100)
            total_buy_price = open_price * quantity * 100  # Opening credit received (negative)
            # Closing: pay to close (negative price * quantity * 100)
            total_sell_price = close_price * quantity * 100 if close_price is not None else 0  # Closing credit paid (negative)
            
            # Get dates
            purchase_datetime = opening['DateTime']
            purchase_datetime_str = purchase_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(purchase_datetime, datetime) else str(purchase_datetime)
            when = opening['Date/Time']
            
            # Get commission from original executions
            parent_executions = df[df['ParentID'] == parent_id]
            commission = parent_executions['Commission'].sum()
            net_cash = parent_executions['NetCash'].sum()
            
            # Extract BAG_ID from any execution (all legs share the same BAG_ID)
            bag_id = None
            if 'BAG_ID' in parent_executions.columns and not parent_executions.empty:
                bag_id = parent_executions['BAG_ID'].iloc[0] if parent_executions['BAG_ID'].notna().any() else None
            
            # Generate Strategy ID from legs (only for BAG orders)
            strategy_id = None
            if bag_id:
                legs_for_id = [
                    {'Symbol': symbol, 'Expiry': expiry, 'Strike': low_strike, 'Put/Call': 'P'},
                    {'Symbol': symbol, 'Expiry': expiry, 'Strike': high_strike, 'Put/Call': 'P'}
                ]
                strategy_id = self.generate_strategy_id(legs_for_id, symbol)
            
            strategy = {
                'OrderID': str(parent_id),
                'BAG_ID': bag_id,  # BAG ID from TradeID
                'StrategyID': strategy_id,  # Human-readable strategy identifier
                'NumLegs': 2,  # Vertical spread has 2 legs
                'Underlying': symbol,
                'When': when,
                'PurchaseDateTime': purchase_datetime_str,
                'Legs': legs_desc,
                'BuyPrice': total_buy_price,  # Opening credit (negative, e.g., -$50 for 2 spreads @ -0.25)
                'SellPrice': total_sell_price,  # Closing credit (negative, e.g., -$106 for 2 spreads @ -0.53)
                'PnL': strategy_pnl,
                'NetCash': net_cash,
                'Commission': commission,
                'TotalCost': net_cash + commission,
                'Quantity': quantity,  # Number of spreads
                'SpreadType': f'{symbol} {expiry_str} {high_strike}/{low_strike} Bull Put',
                'OpenPrice': open_price,  # Price per spread at opening
                'ClosePrice': close_price if close_price is not None else None  # Price per spread at closing
            }
            
            strategies.append(strategy)
        
        logger.info(f"Found {len(strategies)} multi-leg strategies from TWS API executions")
        return strategies
    
    def _group_executions_raw(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Fallback grouping method for non-standard combos (not vertical put spreads).
        Groups by ParentID and shows individual legs.
        """
        strategies = []
        
        if 'ParentID' in df.columns and df['ParentID'].notna().any():
            combo_groups = df.groupby('ParentID')
            logger.info(f"Found {len(combo_groups)} combo orders (by ParentID)")
        else:
            logger.warning("ParentID not available, falling back to OrderID + date/time grouping")
            df['GroupKey'] = df['OrderID'].astype(str) + '_' + df['Date/Time'].astype(str)
            combo_groups = df.groupby('GroupKey')
            logger.info(f"Found {len(combo_groups)} groups (by OrderID + date/time)")
        
        for combo_id, group in combo_groups:
            if len(group) < 2:
                continue
            
            # Build leg descriptions (same as before)
            legs_desc = []
            leg_list = []
            order_ids = []
            purchase_times = []
            strategy_buys = []
            strategy_sells = []
            
            for _, row in group.iterrows():
                symbol = row.get('Symbol', 'N/A')
                expiry = row.get('Expiry', 'N/A')
                strike = row.get('Strike', 'N/A')
                put_call = row.get('Put/Call', 'N/A')
                buy_sell = row.get('Buy/Sell', 'N/A')
                quantity = row.get('Quantity', 0)
                price = row.get('Price', 0)
                orderid = row.get('OrderID', '')
                netcash = row.get('NetCash', 0)
                
                # Format expiry
                if isinstance(expiry, (int, float)) and expiry > 0:
                    expiry_str = str(int(expiry))
                    if len(expiry_str) == 8:
                        expiry_str = f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"
                else:
                    expiry_str = str(expiry)
                
                leg_list.append({
                    'symbol': symbol,
                    'strike': strike,
                    'expiry': expiry_str,
                    'put_call': put_call,
                    'buy_sell': buy_sell,
                    'quantity': quantity,
                    'price': price,
                    'orderid': orderid
                })
                
                if buy_sell == 'BUY':
                    exec_time = row.get('DateTime')
                    if pd.notna(exec_time):
                        purchase_times.append(exec_time)
                
                if buy_sell == 'BUY':
                    strategy_buys.append(netcash)
                else:
                    strategy_sells.append(netcash)
            
            leg_list_sorted = sorted(leg_list, key=lambda x: (x['symbol'], x['strike'], x['put_call']))
            
            for leg in leg_list_sorted:
                leg_desc = f"{leg['buy_sell']} {abs(int(leg['quantity']))} x {leg['symbol']} {leg['expiry']} {leg['strike']}{leg['put_call']}"
                legs_desc.append(leg_desc)
                if leg['orderid'] and pd.notna(leg['orderid']):
                    order_ids.append(str(int(leg['orderid'])) if isinstance(leg['orderid'], (int, float)) else str(leg['orderid']))
            
            total_buy_price = sum(strategy_buys) if strategy_buys else 0
            total_sell_price = sum(strategy_sells) if strategy_sells else 0
            strategy_pnl = total_sell_price - total_buy_price
            
            purchase_datetime_str = None
            if purchase_times:
                purchase_datetimes = [pt for pt in purchase_times if pd.notna(pt)]
                if purchase_datetimes:
                    purchase_datetime = min(purchase_datetimes)
                    if isinstance(purchase_datetime, datetime):
                        purchase_datetime_str = purchase_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        purchase_datetime_str = str(purchase_datetime)
            
            first_row = group.iloc[0]
            underlying = first_row.get('Symbol', 'N/A')
            when = first_row.get('Date/Time', 'N/A')
            net_cash = group['NetCash'].sum()
            commission = group['Commission'].sum()
            
            if 'ParentID' in df.columns and pd.notna(combo_id):
                strategy_id = str(combo_id)
            else:
                strategy_id = ', '.join(order_ids) if order_ids else 'N/A'
            
            # Extract BAG_ID from any execution in the group (all legs share the same BAG_ID)
            bag_id = None
            if 'BAG_ID' in group.columns and not group.empty:
                bag_id = group['BAG_ID'].iloc[0] if group['BAG_ID'].notna().any() else None
            
            # Generate Strategy ID from legs (only for BAG orders)
            strategy_id_str = None
            if bag_id:
                legs_for_id = []
                for leg in leg_list_sorted:
                    legs_for_id.append({
                        'Symbol': leg['symbol'],
                        'Expiry': leg['expiry'],
                        'Strike': leg['strike'],
                        'Put/Call': leg['put_call']
                    })
                strategy_id_str = self.generate_strategy_id(legs_for_id, underlying)
            
            strategy = {
                'OrderID': strategy_id,
                'BAG_ID': bag_id,  # BAG ID from TradeID
                'StrategyID': strategy_id_str,  # Human-readable strategy identifier
                'NumLegs': len(group),
                'Underlying': underlying,
                'When': when,
                'PurchaseDateTime': purchase_datetime_str if purchase_datetime_str else when,
                'Legs': legs_desc,
                'BuyPrice': total_buy_price,
                'SellPrice': total_sell_price,
                'PnL': strategy_pnl,
                'NetCash': net_cash,
                'Commission': commission,
                'TotalCost': net_cash + commission
            }
            
            strategies.append(strategy)
        
        return strategies

