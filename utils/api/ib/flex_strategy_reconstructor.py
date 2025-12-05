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
    # Track which standard names we already have to avoid duplicates
    existing_standard_names = set(df.columns)
    column_map = {}
    
    for col in df.columns:
        col_normalized = col.strip().replace(" ", "").replace("/", "").replace("_", "")
        col_lower = col_normalized.lower()
        
        # Map to standard names, but skip if standard name already exists
        if col_lower in ['orderid', 'order_id']:
            if 'OrderID' not in existing_standard_names:
                column_map[col] = 'OrderID'
        elif col_lower in ['tradeid', 'trade_id']:
            if 'TradeID' not in existing_standard_names:
                column_map[col] = 'TradeID'
        elif col_lower in ['underlyingsymbol', 'underlying_symbol']:
            # Prefer 'Symbol' if it exists, otherwise map UnderlyingSymbol to Symbol
            if 'Symbol' not in existing_standard_names:
                column_map[col] = 'Symbol'
        elif col_lower == 'symbol':
            # Keep as-is, already standard
            pass
        elif col_lower in ['expiry', 'expirationdate', 'expiration_date', 'lasttradedateorcontractmonth']:
            if 'Expiry' not in existing_standard_names:
                column_map[col] = 'Expiry'
        elif col_lower in ['strike', 'strikeprice', 'strike_price']:
            if 'Strike' not in existing_standard_names:
                column_map[col] = 'Strike'
        elif col_lower in ['putcall', 'put/call', 'right', 'optiontype']:
            if 'Put/Call' not in existing_standard_names:
                column_map[col] = 'Put/Call'
        elif col_lower in ['buysell', 'buy/sell', 'side']:
            if 'Buy/Sell' not in existing_standard_names:
                column_map[col] = 'Buy/Sell'
        elif col_lower in ['quantity', 'qty', 'shares']:
            if 'Quantity' not in existing_standard_names:
                column_map[col] = 'Quantity'
        elif col_lower in ['price', 'executionprice', 'execution_price']:
            if 'Price' not in existing_standard_names:
                column_map[col] = 'Price'
        elif col_lower in ['netcash', 'net_cash']:
            if 'NetCash' not in existing_standard_names:
                column_map[col] = 'NetCash'
        elif col_lower == 'proceeds':
            # Proceeds might map to NetCash, but only if NetCash doesn't exist
            if 'NetCash' not in existing_standard_names:
                column_map[col] = 'NetCash'
        elif col_lower in ['commission', 'comm']:
            if 'Commission' not in existing_standard_names:
                column_map[col] = 'Commission'
        elif col_lower in ['date/time', 'date_time', 'executiontime']:
            if 'DateTime' not in existing_standard_names:
                column_map[col] = 'DateTime'
        elif col_lower == 'datetime':
            # Keep as-is if already standard
            pass
        elif col_lower in ['exchange', 'exch']:
            if 'Exchange' not in existing_standard_names:
                column_map[col] = 'Exchange'
    
    if column_map:
        df = df.rename(columns=column_map)
        # Drop duplicate columns if any were created
        df = df.loc[:, ~df.columns.duplicated()]
    
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
    
    # Extract values and convert to Python native types
    # Use underlying symbol if available, otherwise extract from Symbol
    if 'UnderlyingSymbol' in group.columns:
        symbol = str(group['UnderlyingSymbol'].iloc[0]).strip()
    elif 'GroupingSymbol' in group.columns:
        symbol = str(group['GroupingSymbol'].iloc[0]).strip()
    else:
        symbol_val = str(group['Symbol'].iloc[0]) if 'Symbol' in group.columns else ''
        # Extract underlying from option symbol (e.g., "IWM   251201P00228000" -> "IWM")
        symbol = symbol_val.split()[0].strip() if symbol_val else ''
    
    expiries = [int(e) if isinstance(e, (int, float)) else e for e in sorted(group['Expiry'].unique())]
    rights = set(str(r) for r in group['Put/Call'].unique())
    strikes = [float(s) if isinstance(s, (int, float)) else s for s in sorted(group['Strike'].unique())]
    n_legs = len(group)
    
    # Filter out single-leg "strategies" - if all legs have same strike, expiry, and put/call,
    # these are just multiple fills of the same option contract, not a multi-leg strategy
    if len(strikes) == 1 and len(rights) == 1 and len(expiries) == 1:
        logger.debug(f"Filtered out single-leg strategy: {symbol} {expiries[0]} {strikes[0]}{list(rights)[0]} (all legs identical)")
        return {'structure': 'single_leg', 'valid': False}
    
    # ---------- Vertical spreads (2 legs, both P or both C, same expiry) ----------
    if len(strikes) == 2 and len(rights) == 1 and len(expiries) == 1:
        low, high = strikes
        right = list(rights)[0]
        expiry = expiries[0]
        
        g_low = group[group['Strike'] == low]
        g_high = group[group['Strike'] == high]
        
        # Quantity is minimum absolute quantity across legs
        qty_low = float(abs(g_low['Quantity'].sum()))
        qty_high = float(abs(g_high['Quantity'].sum()))
        qty = float(min(qty_low, qty_high))
        
        if qty == 0:
            return {'structure': 'vertical', 'valid': False}
        
        # Compute weighted average price for each leg
        total_low = float((g_low['Price'] * abs(g_low['Quantity'])).sum())
        total_high = float((g_high['Price'] * abs(g_high['Quantity'])).sum())
        price_low = float(total_low / qty_low if qty_low > 0 else 0)
        price_high = float(total_high / qty_high if qty_high > 0 else 0)
        
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
            'symbol': str(symbol),
            'expiry': int(expiry) if isinstance(expiry, (int, float)) else expiry,
            'right': str(right),
            'strikes': [float(s) for s in strikes],
            'quantity': qty,
            'price': float(spread_price),
            'low_strike': float(low),
            'high_strike': float(high),
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
    
    # Ensure Symbol column exists (use UnderlyingSymbol if Symbol doesn't exist)
    # For grouping, we need the underlying symbol, not the full option symbol
    if 'UnderlyingSymbol' in df.columns:
        df['UnderlyingSymbol'] = df['UnderlyingSymbol'].astype(str).str.strip()
        # Use UnderlyingSymbol for grouping (it's the actual stock symbol)
        df['GroupingSymbol'] = df['UnderlyingSymbol']
    elif 'Symbol' in df.columns:
        # Extract underlying from option symbol if needed
        df['GroupingSymbol'] = df['Symbol'].astype(str).str.split().str[0].str.strip()
    else:
        logger.warning("No Symbol or UnderlyingSymbol column found in Flex CSV")
        return []
    
    # Keep Symbol column for leg details (might be full option symbol)
    if 'Symbol' not in df.columns and 'UnderlyingSymbol' in df.columns:
        df['Symbol'] = df['UnderlyingSymbol']
    
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
    processed_groups = set()  # Track processed (Symbol, DateTime) groups to avoid duplicates
    
    # First, try grouping by OrderID (Flex may group multi-leg orders by OrderID)
    for order_id, order_group in df.groupby('OrderID'):
        if pd.isna(order_id):
            continue
        
        # Multiple trade events can share same order ID but different timestamps
        # Group by DateTime to handle partial fills or multiple combo orders
        for dt, dt_group in order_group.groupby('DateTime'):
            if pd.isna(dt):
                continue
            
            # Skip if already processed (will be handled by Symbol+DateTime grouping)
            # Use GroupingSymbol (underlying symbol) for consistent grouping
            symbol = str(dt_group['GroupingSymbol'].iloc[0]) if 'GroupingSymbol' in dt_group.columns else (
                str(dt_group['UnderlyingSymbol'].iloc[0]) if 'UnderlyingSymbol' in dt_group.columns else (
                    str(dt_group['Symbol'].iloc[0]) if 'Symbol' in dt_group.columns else ''
                )
            )
            # Use rounded datetime for consistent grouping
            dt_rounded = dt.floor('s') if hasattr(dt, 'floor') else dt
            group_key = (symbol, dt_rounded)
            if group_key in processed_groups:
                continue
            
            # Classify this group as a strategy
            strat_info = classify_strategy_structure(dt_group)
            
            if not strat_info.get('valid', False):
                continue
            
            # Only mark as processed if valid
            processed_groups.add(group_key)
            
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
            
            # Add execution details from original group (ensure Python native types)
            event['NetCash'] = float(dt_group['NetCash'].sum()) if 'NetCash' in dt_group.columns else 0.0
            event['Commission'] = float(dt_group['Commission'].sum()) if 'Commission' in dt_group.columns else 0.0
            
            # Convert to dict, handling duplicate column names
            # Select only the columns we need to avoid duplicate column issues
            leg_columns = ['Symbol', 'UnderlyingSymbol', 'Expiry', 'Strike', 'Put/Call', 'Buy/Sell', 'Quantity', 'Price', 'NetCash', 'Commission']
            available_columns = [col for col in leg_columns if col in dt_group.columns]
            event['Legs'] = dt_group[available_columns].to_dict('records')
            
            strategies.append(event)
    
    # Also try grouping by Symbol + DateTime (within small time window) to catch
    # multi-leg strategies where legs might have different OrderIDs
    # Round DateTime to nearest second to group trades within same second
    df['DateTimeRounded'] = df['DateTime'].dt.floor('s')
    
    # Group by underlying symbol (GroupingSymbol) and DateTime
    for (symbol, dt_rounded), symbol_group in df.groupby(['GroupingSymbol', 'DateTimeRounded']):
        if pd.isna(symbol) or pd.isna(dt_rounded):
            continue
        
        # Skip if already processed
        group_key = (str(symbol), dt_rounded)
        if group_key in processed_groups:
            continue
        
        # Only process if this group has multiple unique strikes/expiries/rights
        # (indicating a potential multi-leg strategy)
        unique_strikes = symbol_group['Strike'].nunique() if 'Strike' in symbol_group.columns else 0
        unique_expiries = symbol_group['Expiry'].nunique() if 'Expiry' in symbol_group.columns else 0
        unique_rights = symbol_group['Put/Call'].nunique() if 'Put/Call' in symbol_group.columns else 0
        
        # Skip if only one unique strike, expiry, and right (single-leg)
        # This check is redundant since classify_strategy_structure will filter it, but it's faster to skip early
        if unique_strikes <= 1 and unique_expiries <= 1 and unique_rights <= 1:
            logger.debug(f"Skipping single-leg group: {symbol} at {dt_rounded} (unique strikes={unique_strikes}, expiries={unique_expiries}, rights={unique_rights})")
            continue
        
        # Classify this group as a strategy
        strat_info = classify_strategy_structure(symbol_group)
        
        if not strat_info.get('valid', False):
            continue
        
        processed_groups.add(group_key)
        
        # Build strategy event
        event = {
            'OrderID': str(symbol_group['OrderID'].iloc[0]) if 'OrderID' in symbol_group.columns else 'N/A',
            'DateTime': dt_rounded,
            'Symbol': strat_info.get('symbol', ''),
            'Structure': strat_info['structure'],
            'NumLegs': len(symbol_group)
        }
        
        # Add structure-specific fields (same as above)
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
        event['NetCash'] = float(symbol_group['NetCash'].sum()) if 'NetCash' in symbol_group.columns else 0.0
        event['Commission'] = float(symbol_group['Commission'].sum()) if 'Commission' in symbol_group.columns else 0.0
        
        # Convert to dict, handling duplicate column names
        leg_columns = ['Symbol', 'UnderlyingSymbol', 'Expiry', 'Strike', 'Put/Call', 'Buy/Sell', 'Quantity', 'Price', 'NetCash', 'Commission']
        available_columns = [col for col in leg_columns if col in symbol_group.columns]
        event['Legs'] = symbol_group[available_columns].to_dict('records')
        
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
    # Try to import strategy detector for generating strategy IDs
    try:
        from utils.backtesting.strategy_detector import StrategyDetector
        detector = StrategyDetector()
        use_detector = True
    except ImportError:
        logger.warning("StrategyDetector not available, strategy IDs will not be generated")
        detector = None
        use_detector = False
    
    tws_strategies = []
    
    for flex_strat in flex_strategies:
        # Ensure all numeric values are Python native types
        num_legs = int(flex_strat.get('NumLegs', 0))
        net_cash = float(flex_strat.get('NetCash', 0))
        commission = float(flex_strat.get('Commission', 0))
        
        # Format DateTime properly
        dt = flex_strat.get('DateTime', '')
        if isinstance(dt, pd.Timestamp):
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(dt, datetime):
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            dt_str = str(dt) if dt else ''
        
        # Build leg descriptions and prepare legs for strategy ID generation
        legs_desc = []
        legs_for_id = []
        
        # Get underlying symbol - prefer UnderlyingSymbol from legs, fallback to Symbol
        underlying = None
        for leg in flex_strat.get('Legs', []):
            # Check for UnderlyingSymbol in leg data (from Flex CSV)
            if 'UnderlyingSymbol' in leg:
                underlying = str(leg['UnderlyingSymbol']).strip()
                break
        
        # If not found, try to extract from Symbol field or use first leg's symbol
        if not underlying:
            first_leg = flex_strat.get('Legs', [{}])[0] if flex_strat.get('Legs') else {}
            symbol_val = first_leg.get('Symbol', flex_strat.get('Symbol', 'N/A'))
            # Symbol might be full option symbol like "IWM   251201P00228000", extract underlying
            if isinstance(symbol_val, str):
                # Extract underlying from option symbol (first part before spaces/numbers)
                parts = symbol_val.split()
                if parts:
                    underlying = parts[0].strip()
                else:
                    underlying = symbol_val.strip()
            else:
                underlying = str(symbol_val)
        
        underlying = underlying or 'N/A'
        
        for leg in flex_strat.get('Legs', []):
            buy_sell = leg.get('Buy/Sell', 'N/A')
            qty = abs(leg.get('Quantity', 0))
            # Use underlying symbol, not the full option symbol from Symbol field
            symbol = underlying
            expiry = leg.get('Expiry', 'N/A')
            strike = leg.get('Strike', 'N/A')
            put_call = leg.get('Put/Call', 'N/A')
            
            # Format expiry for leg description
            if isinstance(expiry, (int, float)) and expiry > 0:
                expiry_str = str(int(expiry))
                if len(expiry_str) == 8:
                    expiry_str = f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"
            else:
                expiry_str = str(expiry)
            
            leg_desc = f"{buy_sell} {qty} x {symbol} {expiry_str} {strike}{put_call}"
            legs_desc.append(leg_desc)
            
            # Prepare leg for strategy ID generation (use Quantity for sign determination)
            # Normalize Buy/Sell to uppercase for strategy detector
            buy_sell_upper = str(buy_sell).upper() if buy_sell else 'N/A'
            if buy_sell_upper in ('BUY', 'B', 'BOT'):
                buy_sell_normalized = 'BUY'
            elif buy_sell_upper in ('SELL', 'S', 'SLD'):
                buy_sell_normalized = 'SELL'
            else:
                buy_sell_normalized = buy_sell_upper
            
            legs_for_id.append({
                'Symbol': symbol,  # Use underlying symbol
                'Expiry': expiry,
                'Strike': strike,
                'Put/Call': put_call,
                'Buy/Sell': buy_sell_normalized,
                'Quantity': leg.get('Quantity', 0)  # Include quantity for sign determination
            })
        
        # Generate Strategy ID and Short Strategy String using strategy detector
        strategy_id = None
        short_strategy_str = None
        if use_detector and legs_for_id and len(legs_for_id) >= 2:
            try:
                # Ensure Expiry is in correct format (YYYYMMDD as int) for strategy detector
                legs_for_detector = []
                for leg in legs_for_id:
                    leg_copy = leg.copy()
                    # Convert Expiry to int if it's a string in YYYYMMDD format
                    expiry = leg_copy.get('Expiry')
                    if isinstance(expiry, str) and len(expiry) == 8 and expiry.isdigit():
                        leg_copy['Expiry'] = int(expiry)
                    elif isinstance(expiry, str) and '-' in expiry:
                        # Convert YYYY-MM-DD to YYYYMMDD
                        try:
                            from datetime import datetime
                            dt = datetime.strptime(expiry, '%Y-%m-%d')
                            leg_copy['Expiry'] = int(dt.strftime('%Y%m%d'))
                        except:
                            pass
                    legs_for_detector.append(leg_copy)
                
                strategy_id = detector.generate_strategy_id(legs_for_detector, underlying)
                short_strategy_str = detector.generate_short_strategy_string(legs_for_detector, underlying)
                
                # Log if generation failed
                if strategy_id == 'N/A' or short_strategy_str == 'N/A':
                    logger.debug(f"Strategy detector returned N/A for OrderID {flex_strat.get('OrderID')}, legs: {legs_for_detector}")
            except Exception as e:
                logger.debug(f"Error generating strategy ID for OrderID {flex_strat.get('OrderID')}: {e}", exc_info=True)
        
        # Fallback: Generate simple strategy description if detector failed
        if (not short_strategy_str or short_strategy_str == 'N/A') and legs_for_id:
            # Create a simple description from legs
            leg_parts = []
            for leg in sorted(legs_for_id, key=lambda x: (x.get('Expiry', ''), x.get('Strike', 0))):
                expiry = leg.get('Expiry', '')
                strike = leg.get('Strike', '')
                put_call = leg.get('Put/Call', '')
                qty = leg.get('Quantity', 0)
                sign = '+' if qty > 0 else '-'
                
                # Format expiry
                if isinstance(expiry, (int, float)) and expiry > 0:
                    expiry_str = str(int(expiry))
                    if len(expiry_str) == 8:
                        try:
                            from datetime import datetime
                            dt = datetime.strptime(expiry_str, '%Y%m%d')
                            expiry_str = dt.strftime('%b%d')  # Dec12 format
                        except:
                            pass
                else:
                    expiry_str = str(expiry)
                
                leg_parts.append(f"{sign} {expiry_str} {int(strike) if isinstance(strike, (int, float)) else strike}{put_call}")
            
            if leg_parts:
                short_strategy_str = f"{underlying} {' '.join(leg_parts)}"
        
        tws_strat = {
            'OrderID': str(flex_strat.get('OrderID', 'N/A')),
            'BAG_ID': None,  # Flex doesn't have BAG_ID
            'StrategyID': strategy_id if strategy_id and strategy_id != 'N/A' else None,  # Generated from legs
            'ShortStrategyString': short_strategy_str if short_strategy_str and short_strategy_str != 'N/A' else None,  # Generated from legs
            'NumLegs': num_legs,
            'Underlying': underlying,
            'When': dt_str,
            'PurchaseDateTime': dt_str,
            'NetCash': net_cash,
            'Commission': commission,
            'TotalCost': net_cash + commission,
            'Legs': legs_desc
        }
        
        # Add structure-specific fields
        structure = flex_strat.get('Structure', 'unknown')
        if structure == 'vertical':
            quantity = float(flex_strat.get('Quantity', 0))
            price = float(flex_strat.get('Price', 0))
            total_price = price * quantity * 100
            
            tws_strat.update({
                'SpreadType': f"{flex_strat['Symbol']} {flex_strat['Expiry']} {flex_strat['HighStrike']}/{flex_strat['LowStrike']} Bull {'Put' if flex_strat['Put/Call'] == 'P' else 'Call'}",
                'Quantity': int(quantity),
                'OpenPrice': price,
                'ClosePrice': None,
                'BuyPrice': total_price,
                'SellPrice': 0.0,
                'PnL': 0.0,
                'Price': total_price
            })
        else:
            # For non-vertical spreads, use NetCash as price
            tws_strat.update({
                'BuyPrice': net_cash,
                'SellPrice': 0.0,
                'PnL': 0.0,
                'Price': net_cash
            })
        
        tws_strategies.append(tws_strat)
    
    return tws_strategies

