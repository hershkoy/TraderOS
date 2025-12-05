"""
Flex Query CSV Strategy Reconstructor

Converts Flex Query CSV data into multi-leg strategy events by reconstructing
combo orders from individual leg executions. Flex CSV does not contain BAG rows
or combo order metadata, so we must group and classify strategies based on
execution patterns.

Supports:
- Vertical put spreads (bull/bear)
- Vertical call spreads
- Iron condors
- Calendar spreads
- Diagonal spreads
- Partial fills
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def normalize_flex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Flex CSV column names to standard format.
    
    Args:
        df: DataFrame with potentially inconsistent column names
        
    Returns:
        DataFrame with normalized column names
    """
    # Create a mapping of common variations
    column_map = {}
    for col in df.columns:
        col_normalized = col.strip().replace(" ", "").replace("/", "").replace("_", "")
        col_lower = col_normalized.lower()
        
        # Map to standard names
        if col_lower in ['orderid', 'order_id']:
            column_map[col] = 'OrderID'
        elif col_lower in ['tradeid', 'trade_id']:
            column_map[col] = 'TradeID'
        elif col_lower in ['symbol', 'underlyingsymbol', 'underlying_symbol']:
            column_map[col] = 'Symbol'
        elif col_lower in ['expiry', 'expirationdate', 'expiration_date', 'lasttradedateorcontractmonth']:
            column_map[col] = 'Expiry'
        elif col_lower in ['strike', 'strikeprice', 'strike_price']:
            column_map[col] = 'Strike'
        elif col_lower in ['putcall', 'put/call', 'right', 'optiontype']:
            column_map[col] = 'Put/Call'
        elif col_lower in ['buysell', 'buy/sell', 'side']:
            column_map[col] = 'Buy/Sell'
        elif col_lower in ['quantity', 'qty', 'shares']:
            column_map[col] = 'Quantity'
        elif col_lower in ['price', 'executionprice', 'execution_price']:
            column_map[col] = 'Price'
        elif col_lower in ['netcash', 'net_cash', 'proceeds']:
            column_map[col] = 'NetCash'
        elif col_lower in ['commission', 'comm']:
            column_map[col] = 'Commission'
        elif col_lower in ['datetime', 'date/time', 'date_time', 'executiontime']:
            column_map[col] = 'DateTime'
        elif col_lower in ['exchange', 'exch']:
            column_map[col] = 'Exchange'
    
    if column_map:
        df = df.rename(columns=column_map)
    
    return df


def classify_strategy_structure(group: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify strategy type (vertical, condor, calendar, diagonal) and compute quantity + price.
    
    Args:
        group: DataFrame with legs of a potential strategy (same OrderID, DateTime, Symbol)
        
    Returns:
        Dictionary with strategy classification and computed values
    """
    if len(group) < 2:
        return {'structure': 'single_leg', 'valid': False}
    
    symbol = group['Symbol'].iloc[0]
    expiries = sorted(group['Expiry'].unique())
    rights = set(group['Put/Call'].unique())
    strikes = sorted(group['Strike'].unique())
    n_legs = len(group)
    
    # ---------- Vertical spreads (2 legs, both P or both C, same expiry) ----------
    if len(strikes) == 2 and len(rights) == 1 and len(expiries) == 1:
        low, high = strikes
        right = list(rights)[0]
        expiry = expiries[0]
        
        g_low = group[group['Strike'] == low]
        g_high = group[group['Strike'] == high]
        
        # Quantity is minimum absolute quantity across legs
        qty_low = abs(g_low['Quantity'].sum())
        qty_high = abs(g_high['Quantity'].sum())
        qty = min(qty_low, qty_high)
        
        if qty == 0:
            return {'structure': 'vertical', 'valid': False}
        
        # Compute weighted average price for each leg
        total_low = (g_low['Price'] * abs(g_low['Quantity'])).sum()
        total_high = (g_high['Price'] * abs(g_high['Quantity'])).sum()
        price_low = total_low / qty_low if qty_low > 0 else 0
        price_high = total_high / qty_high if qty_high > 0 else 0
        
        # For puts: Bull put = sell high, buy low
        # Spread Price = SellLeg - BuyLeg (net credit is negative in IB style)
        # For calls: Bull call = buy low, sell high
        # Spread Price = BuyLeg - SellLeg (net debit is positive in IB style)
        if right == 'P':
            # Bull put: sell high strike, buy low strike
            # Net credit = high_price - low_price (negative in IB)
            spread_price = price_high - price_low
        else:  # calls
            # Bull call: buy low strike, sell high strike
            # Net debit = low_price - high_price (positive in IB)
            spread_price = price_low - price_high
        
        return {
            'structure': 'vertical',
            'valid': True,
            'symbol': symbol,
            'expiry': expiry,
            'right': right,
            'strikes': strikes,
            'quantity': qty,
            'price': spread_price,
            'low_strike': low,
            'high_strike': high,
            'low_price': price_low,
            'high_price': price_high
        }
    
    # ---------- Iron condor (4 legs: 2 puts + 2 calls, same expiry) ----------
    if len(strikes) == 4 and len(rights) == 2 and len(expiries) == 1:
        expiry = expiries[0]
        puts = group[group['Put/Call'] == 'P']
        calls = group[group['Put/Call'] == 'C']
        
        put_strikes = sorted(puts['Strike'].unique())
        call_strikes = sorted(calls['Strike'].unique())
        
        # Quantity is minimum across all legs
        qty = min(abs(group['Quantity'].sum()) for _, leg_group in group.groupby(['Strike', 'Put/Call']))
        
        return {
            'structure': 'iron_condor',
            'valid': True,
            'symbol': symbol,
            'expiry': expiry,
            'put_strikes': put_strikes,
            'call_strikes': call_strikes,
            'strikes': strikes,
            'quantity': qty
        }
    
    # ---------- Calendar spread (same strike, same right, different expiry) ----------
    if len(strikes) == 1 and len(rights) == 1 and len(expiries) == 2:
        strike = strikes[0]
        right = list(rights)[0]
        
        # Quantity is minimum across expiries
        qty = min(abs(group[group['Expiry'] == exp].groupby('Expiry')['Quantity'].sum().abs().iloc[0]) 
                 for exp in expiries)
        
        return {
            'structure': 'calendar',
            'valid': True,
            'symbol': symbol,
            'strike': strike,
            'right': right,
            'expiries': expiries,
            'quantity': qty
        }
    
    # ---------- Diagonal spread (different expiry + different strike, same right) ----------
    if len(strikes) == 2 and len(rights) == 1 and len(expiries) == 2:
        right = list(rights)[0]
        
        return {
            'structure': 'diagonal',
            'valid': True,
            'symbol': symbol,
            'right': right,
            'strikes': strikes,
            'expiries': expiries
        }
    
    # ---------- Unknown/complex structure ----------
    return {
        'structure': 'unknown',
        'valid': True,  # Still valid, just not classified
        'symbol': symbol,
        'n_legs': n_legs,
        'strikes': strikes,
        'expiries': expiries,
        'rights': list(rights)
    }


def flex_csv_to_strategies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert Flex CSV rows into strategy events grouped like IB TWS.
    
    Groups by OrderID and DateTime, then classifies each group as a strategy type.
    
    Args:
        df: DataFrame with Flex CSV data (normalized columns)
        
    Returns:
        List of strategy dictionaries
    """
    # Normalize columns
    df = normalize_flex_columns(df)
    
    # Filter to only option trades
    if 'Put/Call' in df.columns:
        df = df[df['Put/Call'].notna()].copy()
    
    if df.empty:
        logger.warning("No option trades found in Flex CSV")
        return []
    
    # Ensure DateTime is datetime type
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    elif 'Date/Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    
    strategies = []
    
    # Group by OrderID (Flex groups multi-leg orders by OrderID)
    for order_id, order_group in df.groupby('OrderID'):
        if pd.isna(order_id):
            continue
        
        # Multiple trade events can share same order ID but different timestamps
        # Group by DateTime to handle partial fills or multiple combo orders
        for dt, dt_group in order_group.groupby('DateTime'):
            if pd.isna(dt):
                continue
            
            # Classify this group as a strategy
            strat_info = classify_strategy_structure(dt_group)
            
            if not strat_info.get('valid', False):
                continue
            
            # Build strategy event
            event = {
                'OrderID': str(order_id),
                'DateTime': dt,
                'Symbol': strat_info.get('symbol', ''),
                'Structure': strat_info['structure'],
                'NumLegs': len(dt_group)
            }
            
            # Add structure-specific fields
            if strat_info['structure'] == 'vertical':
                event.update({
                    'Expiry': strat_info['expiry'],
                    'Strikes': strat_info['strikes'],
                    'Put/Call': strat_info['right'],
                    'Quantity': strat_info['quantity'],
                    'Price': strat_info['price'],
                    'LowStrike': strat_info['low_strike'],
                    'HighStrike': strat_info['high_strike'],
                    'LowPrice': strat_info['low_price'],
                    'HighPrice': strat_info['high_price']
                })
            elif strat_info['structure'] == 'iron_condor':
                event.update({
                    'Expiry': strat_info['expiry'],
                    'PutStrikes': strat_info['put_strikes'],
                    'CallStrikes': strat_info['call_strikes'],
                    'Quantity': strat_info['quantity']
                })
            elif strat_info['structure'] == 'calendar':
                event.update({
                    'Strike': strat_info['strike'],
                    'Put/Call': strat_info['right'],
                    'Expiries': strat_info['expiries'],
                    'Quantity': strat_info['quantity']
                })
            elif strat_info['structure'] == 'diagonal':
                event.update({
                    'Put/Call': strat_info['right'],
                    'Strikes': strat_info['strikes'],
                    'Expiries': strat_info['expiries']
                })
            
            # Add execution details from original group
            event['NetCash'] = dt_group['NetCash'].sum() if 'NetCash' in dt_group.columns else 0
            event['Commission'] = dt_group['Commission'].sum() if 'Commission' in dt_group.columns else 0
            event['Legs'] = dt_group.to_dict('records')
            
            strategies.append(event)
    
    logger.info(f"Reconstructed {len(strategies)} strategy events from Flex CSV")
    return strategies


def convert_flex_strategies_to_tws_format(flex_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Flex-reconstructed strategies to TWS-style format for compatibility
    with existing report generation code.
    
    Args:
        flex_strategies: List of strategy dictionaries from flex_csv_to_strategies()
        
    Returns:
        List of strategy dictionaries in TWS format
    """
    tws_strategies = []
    
    for flex_strat in flex_strategies:
        tws_strat = {
            'OrderID': flex_strat.get('OrderID', 'N/A'),
            'BAG_ID': None,  # Flex doesn't have BAG_ID
            'StrategyID': None,  # Will be generated by strategy_detector if needed
            'NumLegs': flex_strat.get('NumLegs', 0),
            'Underlying': flex_strat.get('Symbol', 'N/A'),
            'When': flex_strat.get('DateTime', ''),
            'PurchaseDateTime': flex_strat.get('DateTime', ''),
            'NetCash': flex_strat.get('NetCash', 0),
            'Commission': flex_strat.get('Commission', 0),
            'TotalCost': flex_strat.get('NetCash', 0) + flex_strat.get('Commission', 0)
        }
        
        # Build leg descriptions from legs
        legs_desc = []
        for leg in flex_strat.get('Legs', []):
            buy_sell = leg.get('Buy/Sell', 'N/A')
            qty = abs(leg.get('Quantity', 0))
            symbol = leg.get('Symbol', 'N/A')
            expiry = leg.get('Expiry', 'N/A')
            strike = leg.get('Strike', 'N/A')
            put_call = leg.get('Put/Call', 'N/A')
            
            # Format expiry
            if isinstance(expiry, (int, float)) and expiry > 0:
                expiry_str = str(int(expiry))
                if len(expiry_str) == 8:
                    expiry_str = f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"
            else:
                expiry_str = str(expiry)
            
            leg_desc = f"{buy_sell} {qty} x {symbol} {expiry_str} {strike}{put_call}"
            legs_desc.append(leg_desc)
        
        tws_strat['Legs'] = legs_desc
        
        # Add structure-specific fields
        structure = flex_strat.get('Structure', 'unknown')
        if structure == 'vertical':
            tws_strat.update({
                'SpreadType': f"{flex_strat['Symbol']} {flex_strat['Expiry']} {flex_strat['HighStrike']}/{flex_strat['LowStrike']} Bull {'Put' if flex_strat['Put/Call'] == 'P' else 'Call'}",
                'Quantity': flex_strat.get('Quantity', 0),
                'OpenPrice': flex_strat.get('Price', 0),
                'ClosePrice': None,
                'BuyPrice': flex_strat.get('Price', 0) * flex_strat.get('Quantity', 0) * 100,
                'SellPrice': 0,
                'PnL': 0,
                'Price': flex_strat.get('Price', 0) * flex_strat.get('Quantity', 0) * 100
            })
        else:
            # For non-vertical spreads, use NetCash as price
            tws_strat.update({
                'BuyPrice': flex_strat.get('NetCash', 0),
                'SellPrice': 0,
                'PnL': 0,
                'Price': flex_strat.get('NetCash', 0)
            })
        
        tws_strategies.append(tws_strat)
    
    return tws_strategies

