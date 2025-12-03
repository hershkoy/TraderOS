#!/usr/bin/env python3
"""
IB Flex Query Multi-Leg Strategy Report Generator

Fetches trade history from Interactive Brokers Flex Query Web Service
and generates a report of multi-leg option strategies grouped by OrderID.

Usage:
    python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
    python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type csv
"""

import sys
import os
import argparse
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("Error: requests not found. Please install with: pip install requests")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Set up logging (will be reconfigured in main() based on command-line argument)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# IB Flex Query Web Service endpoints
FLEX_SEND_REQUEST_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
FLEX_GET_STATEMENT_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement"


def get_flex_query_token() -> Optional[str]:
    """Get Flex Query Web Service token from environment variable."""
    token = os.getenv("IB_FLEX_QUERY_TOKEN")
    if not token:
        logger.error("IB_FLEX_QUERY_TOKEN environment variable not set")
        logger.error("Get your token from: Client Portal -> Reports -> Flex Web Service -> Token")
    return token


def get_flex_query_id() -> Optional[str]:
    """Get Flex Query ID from environment variable."""
    query_id = os.getenv("IB_FLEX_QUERY_ID")
    if not query_id:
        logger.error("IB_FLEX_QUERY_ID environment variable not set")
        logger.error("Get your Flex Query ID from: Client Portal -> Reports -> Flex Queries")
    return query_id


def request_flex_query(token: str, query_id: str) -> Optional[str]:
    """
    Step 1: Request a Flex Query run and get Reference Code.
    
    Args:
        token: Flex Web Service token
        query_id: Flex Query ID
        
    Returns:
        Reference code if successful, None otherwise
    """
    url = f"{FLEX_SEND_REQUEST_URL}?t={token}&q={query_id}&v=3"
    
    logger.info(f"Requesting Flex Query run (Query ID: {query_id})...")
    logger.debug(f"Request URL: {FLEX_SEND_REQUEST_URL}?t=***&q={query_id}&v=3")
    
    try:
        response = requests.get(url, timeout=30)
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        response.raise_for_status()
        
        # Parse XML response
        logger.debug(f"Response content length: {len(response.text)} bytes")
        logger.debug(f"Response content preview (first 500 chars): {response.text[:500]}")
        root = ET.fromstring(response.text)
        logger.debug(f"XML root tag: {root.tag}")
        
        status = root.find("Status")
        if status is not None and status.text == "Success":
            ref_code = root.find("ReferenceCode")
            if ref_code is not None and ref_code.text:
                logger.info(f"Flex Query request successful. Reference Code: {ref_code.text}")
                return ref_code.text
            else:
                logger.error("No ReferenceCode in response")
                return None
        else:
            error_code = root.find("ErrorCode")
            error_message = root.find("ErrorMessage")
            error_msg = ""
            if error_code is not None:
                error_msg += f"Error Code: {error_code.text}"
            if error_message is not None:
                error_msg += f", Message: {error_message.text}"
            logger.error(f"Flex Query request failed: {error_msg}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting Flex Query: {e}")
        return None
    except ET.ParseError as e:
        logger.error(f"Error parsing XML response: {e}")
        logger.debug(f"Response text: {response.text}")
        return None


def download_flex_statement(token: str, reference_code: str, max_wait: int = 120) -> Optional[str]:
    """
    Step 2: Download the Flex Query result.
    
    Args:
        token: Flex Web Service token
        reference_code: Reference code from Step 1
        max_wait: Maximum seconds to wait for statement to be ready
        
    Returns:
        XML content as string if successful, None otherwise
    """
    url = f"{FLEX_GET_STATEMENT_URL}?t={token}&v=3&r={reference_code}"
    
    logger.info(f"Downloading Flex Query statement (Reference Code: {reference_code})...")
    logger.debug(f"Download URL: {FLEX_GET_STATEMENT_URL}?t=***&v=3&r={reference_code}")
    logger.debug(f"Max wait time: {max_wait} seconds")
    
    # Wait a bit for statement to be ready, with retries
    wait_time = 2
    max_attempts = max_wait // wait_time
    logger.debug(f"Will retry up to {max_attempts} times with {wait_time} second intervals")
    
    for attempt in range(max_attempts):
        try:
            logger.debug(f"Download attempt {attempt + 1}/{max_attempts}")
            response = requests.get(url, timeout=60)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response content length: {len(response.text)} bytes")
            response.raise_for_status()
            
            # Check if statement is ready (not an error response)
            if response.text and not response.text.strip().startswith("<?xml"):
                # Sometimes IB returns a simple error message
                if "error" in response.text.lower() or "not ready" in response.text.lower():
                    if attempt < max_attempts - 1:
                        logger.info(f"Statement not ready yet, waiting {wait_time} seconds... (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Statement not ready after {max_wait} seconds")
                        logger.debug(f"Response: {response.text[:500]}")
                        return None
            
            logger.info("Flex Query statement downloaded successfully")
            logger.debug(f"Statement content length: {len(response.text)} bytes")
            logger.debug(f"Statement content preview (first 1000 chars): {response.text[:1000]}")
            return response.text
            
        except requests.exceptions.RequestException as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Error downloading statement (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Error downloading Flex Query statement: {e}")
                return None
    
    logger.error(f"Failed to download statement after {max_attempts} attempts")
    return None


def read_flex_report_file(file_path: str) -> pd.DataFrame:
    """
    Read Flex Query report from a local file (CSV or XML).
    
    Args:
        file_path: Path to the Flex Query report file
        
    Returns:
        DataFrame with trade data
    """
    logger.info(f"Reading Flex Query report from file: {file_path}")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    logger.debug(f"File size: {file_path_obj.stat().st_size} bytes")
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"File read successfully, content length: {len(content)} bytes")
        logger.debug(f"Content preview (first 500 chars): {content[:500]}")
        
        # Use existing parse function
        return parse_flex_xml(content)
    except UnicodeDecodeError:
        # Try with different encoding
        logger.debug("UTF-8 failed, trying latin-1 encoding...")
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
        return parse_flex_xml(content)
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        return pd.DataFrame()


def parse_flex_xml(xml_content: str) -> pd.DataFrame:
    """
    Parse Flex Query XML and extract trade data.
    Handles both XML and CSV formats (some Flex Queries can return CSV).
    
    Args:
        xml_content: XML or CSV content from Flex Query
        
    Returns:
        DataFrame with trade data
    """
    logger.info("Parsing Flex Query data...")
    logger.debug(f"Content length: {len(xml_content)} bytes")
    logger.debug(f"Content type detection: starts with XML={xml_content.strip().startswith('<?xml')}, starts with <={xml_content.strip().startswith('<')}")
    
    # Check if content is CSV (starts with header row or is comma-separated)
    content_stripped = xml_content.strip()
    if not content_stripped.startswith('<?xml') and not content_stripped.startswith('<'):
        # Might be CSV format
        try:
            logger.info("Detected CSV format, parsing as CSV...")
            logger.debug(f"CSV content preview (first 500 chars): {xml_content[:500]}")
            # Try to read as CSV
            from io import StringIO
            df = pd.read_csv(StringIO(xml_content))
            logger.info(f"Parsed {len(df)} trade records from CSV with {len(df.columns)} columns")
            logger.debug(f"CSV columns: {list(df.columns)}")
            logger.debug(f"CSV first few rows:\n{df.head()}")
            return df
        except Exception as e:
            logger.warning(f"Failed to parse as CSV: {e}, trying XML...")
    
    # Parse as XML
    try:
        root = ET.fromstring(xml_content)
        
        # Find TradeConfirms section
        # Flex Query XML structure varies, but typically has TradeConfirms or similar
        trades = []
        
        # Try different possible XML structures
        trade_confirms = root.findall(".//TradeConfirms/TradeConfirm")
        if not trade_confirms:
            # Try alternative structure
            trade_confirms = root.findall(".//TradeConfirm")
        
        if not trade_confirms:
            # Try to find any trade-related elements
            logger.warning("Could not find TradeConfirms section, trying alternative parsing...")
            # Look for any elements that might contain trade data
            for elem in root.iter():
                if elem.tag in ['Trade', 'TradeConfirm', 'TradeData', 'Row']:
                    trade_confirms.append(elem)
        
        if not trade_confirms:
            logger.error("No trade data found in Flex Query XML")
            logger.debug(f"XML root tag: {root.tag}")
            logger.debug(f"Available top-level elements: {[child.tag for child in root]}")
            # Try to dump XML structure for debugging
            logger.debug("XML structure (first 1000 chars): " + xml_content[:1000])
            return pd.DataFrame()
        
        logger.info(f"Found {len(trade_confirms)} trade records")
        logger.debug(f"Trade confirm structure - first trade tags: {[child.tag for child in trade_confirms[0]] if trade_confirms else 'N/A'}")
        
        # Parse each trade
        for idx, trade in enumerate(trade_confirms):
            if idx == 0:
                logger.debug(f"First trade element tag: {trade.tag}, children: {[c.tag for c in trade]}")
            trade_data = {}
            
            # Extract all child elements as key-value pairs
            for child in trade:
                tag = child.tag
                text = child.text if child.text else ""
                
                # Handle numeric fields (case-insensitive matching)
                numeric_fields = ['OrderID', 'TradeID', 'Quantity', 'Strike', 'Price', 'NetCash', 'Commission',
                                 'orderid', 'tradeid', 'quantity', 'strike', 'price', 'netcash', 'commission']
                if tag in numeric_fields:
                    try:
                        trade_data[tag] = float(text) if text else None
                    except (ValueError, TypeError):
                        trade_data[tag] = None
                else:
                    trade_data[tag] = text
            
            trades.append(trade_data)
        
        if not trades:
            logger.warning("No trade data extracted from XML")
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)
        logger.info(f"Parsed {len(df)} trade records with {len(df.columns)} columns")
        logger.debug(f"Columns: {list(df.columns)}")
        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"First few rows:\n{df.head()}")
        logger.debug(f"Data types:\n{df.dtypes}")
        logger.debug(f"Null counts:\n{df.isnull().sum()}")
        
        return df
        
    except ET.ParseError as e:
        logger.error(f"Error parsing XML: {e}")
        logger.debug(f"Content preview (first 500 chars): {xml_content[:500]}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing Flex Query data: {e}", exc_info=True)
        return pd.DataFrame()


def filter_by_date(df: pd.DataFrame, since_date: Optional[datetime]) -> pd.DataFrame:
    """
    Filter DataFrame by date.
    
    Args:
        df: DataFrame with trade data
        since_date: Filter trades on or after this date
        
    Returns:
        Filtered DataFrame
    """
    if since_date is None:
        return df
    
    # Try to find date column (common names: Date, Date/Time, TradeDate, etc.)
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    logger.debug(f"Found date columns: {date_columns}")
    
    if not date_columns:
        logger.warning("No date column found, cannot filter by date")
        logger.debug(f"Available columns: {list(df.columns)}")
        return df
    
    # Use first date column found
    date_col = date_columns[0]
    logger.info(f"Filtering by date using column: {date_col}")
    logger.debug(f"Filter date: {since_date}")
    logger.debug(f"Date column sample values: {df[date_col].head().tolist()}")
    
    # Convert date column to datetime
    # Handle different date formats that IB might use
    try:
        # Try parsing as datetime with various formats
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        
        # If parsing failed for some rows, try manual parsing for common IB formats
        if df[date_col].isna().any():
            logger.debug("Some dates failed to parse, trying alternative formats...")
            # Common IB formats: YYYYMMDD, YYYYMMDD HH:MM:SS, YYYY-MM-DD, etc.
            for idx, val in df[date_col].items():
                if pd.isna(val) and pd.notna(df.loc[idx, date_col]):
                    val_str = str(df.loc[idx, date_col])
                    # Try YYYYMMDD format
                    if len(val_str) == 8 and val_str.isdigit():
                        try:
                            df.loc[idx, date_col] = pd.to_datetime(val_str, format='%Y%m%d')
                        except:
                            pass
        
        filtered = df[df[date_col] >= since_date].copy()
        logger.info(f"Filtered to {len(filtered)} trades on or after {since_date.strftime('%Y-%m-%d')} (from {len(df)} total)")
        logger.debug(f"Date range in filtered data: {filtered[date_col].min()} to {filtered[date_col].max()}")
        return filtered
    except Exception as e:
        logger.warning(f"Error filtering by date: {e}")
        return df


def group_multi_leg_strategies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Group trades by OrderID to identify multi-leg strategies.
    
    Args:
        df: DataFrame with trade data
        
    Returns:
        List of strategy dictionaries
    """
    logger.info("Grouping trades into multi-leg strategies...")
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Input columns: {list(df.columns)}")
    
    # Normalize column names (case-insensitive matching)
    df_normalized = df.copy()
    column_map = {}
    logger.debug("Normalizing column names...")
    for col in df.columns:
        col_lower = col.lower()
        # Map common column name variations
        if col_lower in ['orderid', 'order_id']:
            column_map[col] = 'OrderID'
        elif col_lower in ['put/call', 'putcall', 'put_call']:
            column_map[col] = 'Put/Call'
        elif col_lower in ['buy/sell', 'buysell', 'buy_sell']:
            column_map[col] = 'Buy/Sell'
        elif col_lower in ['netcash', 'net_cash']:
            column_map[col] = 'NetCash'
    
    # Rename columns
    if column_map:
        logger.debug(f"Column mapping: {column_map}")
        df_normalized = df_normalized.rename(columns=column_map)
    else:
        logger.debug("No column mapping needed")
    
    # Filter to only trades with valid OrderID
    # Also filter to option trades (typically have Put/Call column)
    option_trades = df_normalized.copy()
    logger.debug(f"Total trades before filtering: {len(option_trades)}")
    
    # Check if we have Put/Call column (indicates options)
    put_call_cols = [col for col in option_trades.columns if 'put' in col.lower() and 'call' in col.lower()]
    logger.debug(f"Put/Call columns found: {put_call_cols}")
    if put_call_cols:
        put_call_col = put_call_cols[0]
        before_count = len(option_trades)
        option_trades = option_trades[option_trades[put_call_col].notna()].copy()
        logger.info(f"Filtered to {len(option_trades)} option trades (using column: {put_call_col})")
        logger.debug(f"Removed {before_count - len(option_trades)} non-option trades")
        logger.debug(f"Put/Call value distribution: {option_trades[put_call_col].value_counts().to_dict()}")
    
    # Filter to trades with OrderID
    orderid_cols = [col for col in option_trades.columns if 'orderid' in col.lower() or 'order_id' in col.lower()]
    logger.debug(f"OrderID columns found: {orderid_cols}")
    if not orderid_cols:
        logger.warning("No OrderID column found, cannot group multi-leg strategies")
        logger.debug(f"Available columns: {list(option_trades.columns)}")
        return []
    
    orderid_col = orderid_cols[0]
    before_count = len(option_trades)
    legs_with_order = option_trades[option_trades[orderid_col].notna()].copy()
    logger.debug(f"Removed {before_count - len(legs_with_order)} trades without OrderID")
    
    if len(legs_with_order) == 0:
        logger.warning("No trades with OrderID found")
        return []
    
    logger.info(f"Found {len(legs_with_order)} option trades with OrderID")
    logger.debug(f"Unique OrderIDs: {legs_with_order[orderid_col].nunique()}")
    logger.debug(f"OrderID value counts (top 10):\n{legs_with_order[orderid_col].value_counts().head(10)}")
    
    # Group by OrderID
    strategies = []
    grouped = legs_with_order.groupby(orderid_col)
    logger.debug(f"Grouped into {len(grouped)} unique OrderIDs")
    
    single_leg_count = 0
    for order_id, group in grouped:
        # Only consider multi-leg strategies (2+ legs)
        if len(group) < 2:
            single_leg_count += 1
            logger.debug(f"Skipping OrderID {order_id} - only {len(group)} leg(s)")
            continue
        
        logger.debug(f"Processing OrderID {order_id} with {len(group)} legs")
        
        # Get column names (use normalized or original)
        symbol_col = 'Symbol' if 'Symbol' in group.columns else ([c for c in group.columns if 'symbol' in c.lower()] or ['Symbol'])[0]
        strike_col = 'Strike' if 'Strike' in group.columns else ([c for c in group.columns if 'strike' in c.lower()] or ['Strike'])[0]
        expiry_col = 'Expiry' if 'Expiry' in group.columns else ([c for c in group.columns if 'expiry' in c.lower()] or ['Expiry'])[0]
        put_call_col = put_call_cols[0] if put_call_cols else 'Put/Call'
        buy_sell_col = 'Buy/Sell' if 'Buy/Sell' in group.columns else ([c for c in group.columns if 'buy' in c.lower() and 'sell' in c.lower()] or ['Buy/Sell'])[0]
        quantity_col = 'Quantity' if 'Quantity' in group.columns else ([c for c in group.columns if 'quantity' in c.lower()] or ['Quantity'])[0]
        price_col = 'Price' if 'Price' in group.columns else ([c for c in group.columns if 'price' in c.lower()] or ['Price'])[0]
        
        # Sort by Symbol, Strike, Put/Call for consistent display
        sort_cols = [c for c in [symbol_col, strike_col, put_call_col] if c in group.columns]
        if sort_cols:
            group_sorted = group.sort_values(by=sort_cols)
        else:
            group_sorted = group
        
        # Build leg descriptions
        legs_desc = []
        logger.debug(f"OrderID {order_id} - sorted group columns: {list(group_sorted.columns)}")
        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            logger.debug(f"OrderID {order_id} - Leg {idx + 1} data: {dict(row)}")
            symbol = row.get(symbol_col, 'N/A')
            expiry = row.get(expiry_col, 'N/A')
            strike = row.get(strike_col, 'N/A')
            put_call = row.get(put_call_col, 'N/A')
            buy_sell = row.get(buy_sell_col, 'N/A')
            quantity = row.get(quantity_col, 0)
            price = row.get(price_col, 0)
            
            # Format expiry if it's a number (YYYYMMDD)
            if isinstance(expiry, (int, float)) and expiry > 0:
                expiry_str = str(int(expiry))
                if len(expiry_str) == 8:
                    expiry_str = f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"
            else:
                expiry_str = str(expiry)
            
            leg_desc = f"{buy_sell} {int(quantity)} x {symbol} {expiry_str} {strike}{put_call} @ {price:.2f}"
            legs_desc.append(leg_desc)
        
        # Get strategy metadata
        underlying = group_sorted[symbol_col].iloc[0] if symbol_col in group_sorted.columns else 'N/A'
        
        # Try to get date/time
        date_cols = [col for col in group_sorted.columns if 'date' in col.lower() or 'time' in col.lower()]
        when = group_sorted[date_cols[0]].iloc[0] if date_cols else 'N/A'
        if isinstance(when, pd.Timestamp):
            when = when.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate totals
        netcash_col = 'NetCash' if 'NetCash' in group_sorted.columns else ([c for c in group_sorted.columns if 'netcash' in c.lower() or 'net_cash' in c.lower()] or [None])[0]
        commission_col = 'Commission' if 'Commission' in group_sorted.columns else ([c for c in group_sorted.columns if 'commission' in c.lower()] or [None])[0]
        
        net_cash = group_sorted[netcash_col].sum() if netcash_col and netcash_col in group_sorted.columns else 0
        commission = group_sorted[commission_col].sum() if commission_col and commission_col in group_sorted.columns else 0
        
        strategy = {
            'OrderID': int(order_id) if pd.notna(order_id) else order_id,
            'NumLegs': len(group),
            'Underlying': underlying,
            'When': when,
            'Legs': legs_desc,
            'NetCash': net_cash,
            'Commission': commission,
            'TotalCost': net_cash + commission
        }
        
        strategies.append(strategy)
        logger.debug(f"OrderID {order_id} - Strategy summary: {strategy['NumLegs']} legs, NetCash={strategy['NetCash']:.2f}, Commission={strategy['Commission']:.2f}")
    
    logger.info(f"Found {len(strategies)} multi-leg strategies")
    logger.debug(f"Skipped {single_leg_count} single-leg trades")
    if strategies:
        logger.debug(f"Strategy OrderIDs: {[s['OrderID'] for s in strategies]}")
    return strategies


def generate_csv_report(strategies: List[Dict[str, Any]], output_path: str):
    """
    Generate CSV report of multi-leg strategies.
    
    Args:
        strategies: List of strategy dictionaries
        output_path: Output file path
    """
    logger.info(f"Generating CSV report: {output_path}")
    
    # Flatten strategies for CSV (one row per strategy, legs as text)
    rows = []
    for strategy in strategies:
        row = {
            'OrderID': strategy['OrderID'],
            'NumLegs': strategy['NumLegs'],
            'Underlying': strategy['Underlying'],
            'When': strategy['When'],
            'Legs': ' | '.join(strategy['Legs']),
            'NetCash': strategy['NetCash'],
            'Commission': strategy['Commission'],
            'TotalCost': strategy['TotalCost']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV report saved: {output_path}")


def generate_html_report(strategies: List[Dict[str, Any]], output_path: str):
    """
    Generate HTML report of multi-leg strategies.
    
    Args:
        strategies: List of strategy dictionaries
        output_path: Output file path
    """
    logger.info(f"Generating HTML report: {output_path}")
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>IB Multi-Leg Strategy Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .strategy {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .strategy-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .strategy-info {{
            flex: 1;
        }}
        .strategy-id {{
            font-weight: bold;
            color: #4CAF50;
            font-size: 1.1em;
        }}
        .strategy-meta {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .strategy-financials {{
            text-align: right;
        }}
        .net-cash {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        .commission {{
            color: #666;
            font-size: 0.9em;
        }}
        .legs {{
            margin-top: 10px;
        }}
        .leg {{
            padding: 8px;
            margin: 5px 0;
            background-color: #f9f9f9;
            border-left: 3px solid #4CAF50;
            font-family: 'Courier New', monospace;
        }}
        .summary {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }}
        .summary-stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Interactive Brokers Multi-Leg Strategy Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-stats">
            <div class="stat">
                <div class="stat-value">{num_strategies}</div>
                <div class="stat-label">Strategies</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_legs}</div>
                <div class="stat-label">Total Legs</div>
            </div>
            <div class="stat">
                <div class="stat-value">${total_net_cash:,.2f}</div>
                <div class="stat-label">Total Net Cash</div>
            </div>
            <div class="stat">
                <div class="stat-value">${total_commission:,.2f}</div>
                <div class="stat-label">Total Commission</div>
            </div>
        </div>
    </div>
"""
    
    # Calculate summary stats
    num_strategies = len(strategies)
    total_legs = sum(s['NumLegs'] for s in strategies)
    total_net_cash = sum(s['NetCash'] for s in strategies)
    total_commission = sum(s['Commission'] for s in strategies)
    
    html = html.format(
        num_strategies=num_strategies,
        total_legs=total_legs,
        total_net_cash=total_net_cash,
        total_commission=total_commission
    )
    
    # Add each strategy
    for strategy in strategies:
        html += f"""
    <div class="strategy">
        <div class="strategy-header">
            <div class="strategy-info">
                <div class="strategy-id">Order ID: {strategy['OrderID']}</div>
                <div class="strategy-meta">
                    {strategy['NumLegs']} legs | {strategy['Underlying']} | {strategy['When']}
                </div>
            </div>
            <div class="strategy-financials">
                <div class="net-cash">${strategy['NetCash']:,.2f}</div>
                <div class="commission">Commission: ${strategy['Commission']:,.2f}</div>
            </div>
        </div>
        <div class="legs">
"""
        
        for leg in strategy['Legs']:
            html += f'            <div class="leg">{leg}</div>\n'
        
        html += """        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"HTML report saved: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate multi-leg strategy report from IB Flex Query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate HTML report for trades since 2025-01-01
  python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
  
  # Generate CSV report for all trades
  python scripts/api/ib/ib_flex_multi_leg_report.py --type csv
  
  # Generate HTML report with custom output file
  python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --output reports/my_report.html
  
  # Generate report with longer wait time for large queries
  python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --max-wait 180
  
  # Generate report with debug logging
  python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --log-level DEBUG
  
  # Use manually downloaded Flex Query report file (no API credentials needed)
  python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report flex_report.csv --type html
  python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report flex_report.xml --since 2025-01-01 --type html

Environment Variables Required (only if not using --flex-report):
  IB_FLEX_QUERY_TOKEN: Flex Web Service token from Client Portal -> Reports -> Flex Web Service -> Token
  IB_FLEX_QUERY_ID: Flex Query ID from Client Portal -> Reports -> Flex Queries
        """
    )
    
    parser.add_argument(
        '--since',
        type=str,
        default=None,
        help='Filter trades on or after this date (YYYY-MM-DD format, e.g., 2025-01-01)'
    )
    
    parser.add_argument(
        '--type',
        choices=['html', 'csv'],
        default='html',
        help='Output format: html or csv (default: html)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated in reports/ folder)'
    )
    
    parser.add_argument(
        '--max-wait',
        type=int,
        default=120,
        help='Maximum seconds to wait for Flex Query statement to be ready (default: 120)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level: DEBUG, INFO, WARNING, or ERROR (default: INFO). Use DEBUG for detailed troubleshooting.'
    )
    
    parser.add_argument(
        '--flex-report',
        type=str,
        default=None,
        help='Path to manually downloaded Flex Query report file (CSV or XML). If provided, skips API download.'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    logger.debug(f"Logging level set to {args.log_level}")
    
    # Parse since date
    since_date = None
    if args.since:
        try:
            since_date = datetime.strptime(args.since, '%Y-%m-%d')
            logger.info(f"Filtering trades on or after: {since_date.strftime('%Y-%m-%d')}")
        except ValueError:
            logger.error(f"Invalid date format: {args.since}. Use YYYY-MM-DD format (e.g., 2025-01-01)")
            sys.exit(1)
    
    # Check if using manual file or API
    if args.flex_report:
        # Use manually downloaded file
        logger.info(f"Using manually downloaded Flex Query report: {args.flex_report}")
        df = read_flex_report_file(args.flex_report)
        if df.empty:
            logger.error("No trade data found in Flex Query file")
            sys.exit(1)
    else:
        # Use API to fetch Flex Query
        # Get credentials
        token = get_flex_query_token()
        query_id = get_flex_query_id()
        
        if not token or not query_id:
            logger.error("Missing required credentials. Please set IB_FLEX_QUERY_TOKEN and IB_FLEX_QUERY_ID environment variables.")
            logger.error("Alternatively, use --flex-report to provide a manually downloaded file.")
            sys.exit(1)
        
        # Step 1: Request Flex Query
        reference_code = request_flex_query(token, query_id)
        if not reference_code:
            logger.error("Failed to request Flex Query")
            sys.exit(1)
        
        # Step 2: Download statement
        xml_content = download_flex_statement(token, reference_code, max_wait=args.max_wait)
        if not xml_content:
            logger.error("Failed to download Flex Query statement")
            sys.exit(1)
        
        # Step 3: Parse XML
        df = parse_flex_xml(xml_content)
        if df.empty:
            logger.error("No trade data found in Flex Query")
            sys.exit(1)
    
    # Step 4 (or 1 if using file): Filter by date
    if since_date:
        df = filter_by_date(df, since_date)
        if df.empty:
            logger.warning(f"No trades found on or after {since_date.strftime('%Y-%m-%d')}")
            sys.exit(0)
    
    # Step 5: Group into multi-leg strategies
    strategies = group_multi_leg_strategies(df)
    if not strategies:
        logger.warning("No multi-leg strategies found")
        sys.exit(0)
    
    # Step 6: Generate output
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = args.type
        output_path = f"reports/ib_multi_leg_strategies_{timestamp}.{ext}"
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args.type == 'html':
        generate_html_report(strategies, str(output_file))
    else:
        generate_csv_report(strategies, str(output_file))
    
    logger.info(f"Report generated successfully: {output_path}")
    logger.info(f"Found {len(strategies)} multi-leg strategies")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

