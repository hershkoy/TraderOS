#!/usr/bin/env python3
"""
IB Multi-Leg Strategy Report Generator

Fetches trade history from Interactive Brokers and generates a report of 
multi-leg option strategies. Uses TWS API (default) for accurate combo order 
grouping, or Flex Query Web Service (fallback) for historical data.

Usage:
    # TWS API (default - matches IB's Trade History Summary exactly)
    python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
    
    # Flex Query (fallback for historical data)
    python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --data-source flex
"""

import sys
import os
import argparse
import logging
import time
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

try:
    from ib_insync import IB, Execution, ExecutionFilter, Contract
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import requests
except ImportError:
    print("Error: requests not found. Please install with: pip install requests")
    sys.exit(1)

# Import IB connection utilities
try:
    from utils.fetch_data import get_ib_connection, cleanup_ib_connection
    from utils.ib_port_detector import detect_ib_port
    from utils.strategy_detector import StrategyDetector
except ImportError:
    # Fallback if running from different directory
    def get_ib_connection(port=None, client_id=None):
        if not IB_INSYNC_AVAILABLE:
            raise ImportError("ib_insync not available")
        ib = IB()
        ib.connect('127.0.0.1', port or 4001, clientId=client_id or 2)
        return ib
    
    def cleanup_ib_connection():
        pass
    
    def detect_ib_port():
        return 4001

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
            # Exclude account data from debug output
            df_to_log = df.drop(columns=['Account'], errors='ignore')
            logger.debug(f"CSV first few rows (account data excluded):\n{df_to_log.head()}")
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
        # Exclude account data from debug output
        df_to_log = df.drop(columns=['Account'], errors='ignore')
        logger.debug(f"First few rows (account data excluded):\n{df_to_log.head()}")
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


# Create a global StrategyDetector instance for backward compatibility
_strategy_detector = None

def _get_strategy_detector():
    """Get or create the global StrategyDetector instance."""
    global _strategy_detector
    if _strategy_detector is None:
        try:
            from utils.strategy_detector import StrategyDetector
            _strategy_detector = StrategyDetector()
        except ImportError:
            # Fallback if StrategyDetector is not available
            logger.warning("StrategyDetector not available, using legacy functions")
            return None
    return _strategy_detector


def infer_leg_direction_for_combo(execution_data: List[Dict[str, Any]], parent_id: str) -> Dict[str, str]:
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
    detector = _get_strategy_detector()
    if detector:
        return detector.infer_leg_direction_for_combo(execution_data, parent_id)
    # Fallback to empty dict if detector not available
    return {}


def generate_strategy_id(legs: List[Dict[str, Any]], underlying: str = None) -> str:
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
    detector = _get_strategy_detector()
    if detector:
        return detector.generate_strategy_id(legs, underlying)
    # Fallback to "N/A" if detector not available
    return "N/A"


def fetch_tws_executions(since_date: Optional[datetime] = None, port: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch trade executions from TWS API using reqExecutions().
    Groups by parentId (combo order ID) to match IB's Trade History Summary.
    
    Args:
        since_date: Filter executions on or after this date
        port: IB API port number (optional)
        
    Returns:
        DataFrame with execution data in format compatible with Flex Query data
    """
    if not IB_INSYNC_AVAILABLE:
        raise ImportError("ib_insync not available. Install with: pip install ib_insync")
    
    logger.info("Fetching executions from TWS API...")
    
    # Get IB connection
    ib = get_ib_connection(port=port, client_id=3)  # Use different client ID for executions
    
    try:
        # Create execution filter (optional - can filter by client ID, account, etc.)
        exec_filter = ExecutionFilter()
        # Leave empty to get all executions
        
        # Check TWS connection status
        if not ib.isConnected():
            logger.error("TWS connection is not active")
            return pd.DataFrame()
        
        # Request executions with retry (IB API can be unreliable)
        logger.info("Requesting executions from TWS API...")
        logger.debug(f"Execution filter: {exec_filter}")
        logger.debug(f"TWS connection status: Connected={ib.isConnected()}")
        
        # Try requesting executions with retries (IB API can be flaky)
        executions = []
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                executions = ib.reqExecutions(exec_filter)
                logger.info(f"Received {len(executions)} executions from TWS API (attempt {attempt + 1}/{max_retries})")
                
                if executions:
                    break  # Success, exit retry loop
                elif attempt < max_retries - 1:
                    logger.warning(f"No executions returned, retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"Error requesting executions (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to get executions after {max_retries} attempts")
        
        if not executions:
            logger.warning("No executions found from TWS API after retries")
            logger.info("Possible reasons:")
            logger.info("  1. No executions in your account")
            logger.info("  2. TWS needs to sync with server (check TWS for sync status)")
            logger.info("  3. Account permissions may not allow execution history access")
            logger.info("  4. Try restarting TWS and ensuring it's fully connected")
            logger.info("  5. Check if you have any trades in TWS Account Management > Trade History")
            logger.info("  6. The IB API reqExecutions() can be unreliable - try running again in a few moments")
            return pd.DataFrame()
        
        # Log raw execution data from IB API (DEBUG level)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("=" * 80)
            logger.debug("RAW EXECUTION DATA FROM IB API (before any processing):")
            logger.debug("=" * 80)
            for idx, fill in enumerate(executions):
                contract = fill.contract
                execution = fill.execution
                commission_report = fill.commissionReport
                
                # Log contract details
                logger.debug(f"\n--- Execution {idx + 1} ---")
                logger.debug(f"Contract:")
                logger.debug(f"  secType: {contract.secType}")
                logger.debug(f"  symbol: {contract.symbol}")
                logger.debug(f"  lastTradeDateOrContractMonth: {contract.lastTradeDateOrContractMonth}")
                if contract.secType == 'OPT':
                    logger.debug(f"  strike: {contract.strike}")
                    logger.debug(f"  right: {contract.right}")
                
                # Log execution details
                logger.debug(f"Execution:")
                logger.debug(f"  execId: {execution.execId}")
                logger.debug(f"  orderId: {execution.orderId}")
                logger.debug(f"  orderRef: {execution.orderRef}")
                logger.debug(f"  side: {execution.side} ({'BUY' if execution.side == 'B' else 'SELL'})")
                logger.debug(f"  shares: {execution.shares}")
                logger.debug(f"  price: {execution.price}")
                logger.debug(f"  time: {execution.time}")
                logger.debug(f"  exchange: {execution.exchange}")
                if hasattr(execution, 'acctNumber'):
                    logger.debug(f"  acctNumber: {execution.acctNumber}")
                
                # Log fill time
                logger.debug(f"Fill time: {fill.time}")
                
                # Log commission if available
                if commission_report:
                    logger.debug(f"Commission: {commission_report.commission}")
                
                # Extract parent ID for combo orders
                parent_id_raw = None
                if execution.orderRef:
                    parent_id_raw = execution.orderRef
                elif execution.execId:
                    parts = execution.execId.split('.')
                    if len(parts) > 1:
                        parent_id_raw = parts[0]
                    else:
                        parent_id_raw = execution.execId
                logger.debug(f"Extracted ParentID: {parent_id_raw}")
                logger.debug("-" * 80)
        
        # Convert executions to DataFrame format compatible with Flex Query
        # Note: reqExecutions() returns Fill objects, which contain:
        # - contract: Contract object
        # - execution: Execution object (with side, price, shares, execId, orderRef, etc.)
        # - commissionReport: CommissionReport object
        # - time: datetime
        execution_data = []
        
        for fill in executions:
            contract = fill.contract
            execution = fill.execution  # The actual Execution object
            
            # Skip non-option trades
            if contract.secType != 'OPT':
                continue
            
            # Parse execution time
            exec_time = fill.time
            if isinstance(exec_time, str):
                try:
                    exec_time = datetime.strptime(exec_time, '%Y%m%d  %H:%M:%S')
                except:
                    try:
                        exec_time = datetime.strptime(exec_time, '%Y%m%d %H:%M:%S')
                    except:
                        exec_time = pd.to_datetime(exec_time, errors='coerce')
            
            # Filter by date if specified
            if since_date and exec_time:
                if isinstance(exec_time, datetime) and exec_time.date() < since_date.date():
                    continue
            
            # Extract option details
            symbol = contract.symbol
            expiry = contract.lastTradeDateOrContractMonth  # Format: YYYYMMDD
            strike = contract.strike
            right = contract.right  # 'P' or 'C'
            
            # Get parent ID (combo order ID) - this is the key for grouping
            # For combo orders, all legs share the same orderId, which is the combo order ID
            # This is the primary identifier that links all legs of a combo order together
            parent_id = None
            if execution.orderId and execution.orderId != 0:
                # For combo orders, orderId is the combo order ID that links all legs together
                # Use orderId as ParentID since all legs in a combo share the same orderId
                parent_id = str(execution.orderId)
            elif execution.orderRef:
                # orderRef is an alternative combo order identifier
                parent_id = execution.orderRef
            elif execution.execId:
                # Fallback: try to extract from execId format
                # Some formats: "parentId.legId" or similar
                # Note: This is less reliable than orderId, but used as last resort
                parts = execution.execId.split('.')
                if len(parts) > 1:
                    parent_id = parts[0]
                else:
                    parent_id = execution.execId
            
            # Initial buy/sell from execution side (will be updated later for combos)
            # execution.side: 'B' = Buy, 'S' = Sell
            # NOTE: For credit spreads, IB often shows all leg executions as 'S' (SELL)
            # We'll infer the actual leg direction based on strike prices and spread structure
            buy_sell_initial = 'BUY' if execution.side == 'B' else 'SELL'
            
            # NetCash calculation
            # For individual leg executions (which is what we get for combo orders):
            # - execution.price is the price per share for this individual leg
            # - execution.shares is the number of contracts
            # - NetCash for this leg = price * quantity * 100 (contract multiplier)
            # For credit spreads: SELL leg has positive price, BUY leg has negative price (or vice versa)
            # We'll aggregate all legs in the grouping function to get the combo net price
            price_per_share = execution.price
            quantity = execution.shares
            
            # Calculate NetCash for this individual leg
            # For BUY: negative (money out), for SELL: positive (money in)
            # Note: This is initial calculation, will be recalculated after direction inference
            # Use absolute price since IB API typically provides absolute prices
            net_cash_initial = abs(price_per_share * quantity * 100)
            if buy_sell_initial == 'BUY':
                net_cash = -net_cash_initial  # Money out
            else:  # SELL
                net_cash = net_cash_initial  # Money in
            
            # Commission (from commissionReport if available)
            commission = 0.0
            if fill.commissionReport:
                commission = fill.commissionReport.commission
            
            # Build execution record (Buy/Sell will be updated after inference)
            # For combo orders: all legs share the same orderId (combo order ID)
            # OrderID = orderId (the combo order ID, same for all legs in a combo)
            # ParentID = orderId (same, used for grouping legs of the same combo)
            # TradeID = execId (unique execution ID per leg)
            # BAG ID = first part of execId (format: bag_id.leg_id.leg_number.execution_id)
            order_id_value = execution.orderId if execution.orderId and execution.orderId != 0 else (execution.orderRef if execution.orderRef else parent_id)
            
            # Extract BAG ID from TradeID (execId format: bag_id.leg_id.leg_number.execution_id)
            bag_id = None
            if execution.execId:
                parts = execution.execId.split('.')
                if len(parts) > 0:
                    bag_id = parts[0]  # First part is the BAG ID
            
            exec_record = {
                'OrderID': order_id_value,
                'TradeID': execution.execId,
                'BAG_ID': bag_id,  # BAG ID extracted from TradeID
                'ParentID': parent_id,  # Combo order ID for grouping (should be orderId for combo orders)
                'Symbol': symbol,
                'Expiry': int(expiry) if expiry and expiry.isdigit() else expiry,
                'Strike': strike,
                'Put/Call': right,
                'Buy/Sell': buy_sell_initial,  # Will be updated after inference
                'Quantity': quantity if buy_sell_initial == 'BUY' else -quantity,  # Will be updated after inference
                'Price': price_per_share,
                'NetCash': net_cash,
                'Commission': commission,
                'Date/Time': exec_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(exec_time, datetime) else str(exec_time),
                'DateTime': exec_time,  # Keep as datetime for filtering
                'Exchange': execution.exchange,
                'Account': execution.acctNumber if hasattr(execution, 'acctNumber') else '',
            }
            
            execution_data.append(exec_record)
        
        # Second pass: Infer leg directions for combo orders
        # Group by ParentID to identify combos
        parent_ids = set(e.get('ParentID') for e in execution_data if e.get('ParentID'))
        leg_direction_map = {}  # Maps TradeID to inferred Buy/Sell
        
        for parent_id in parent_ids:
            inferred_directions = infer_leg_direction_for_combo(execution_data, parent_id)
            leg_direction_map.update(inferred_directions)
        
        # Update Buy/Sell and Quantity based on inferred directions
        for exec_record in execution_data:
            trade_id = exec_record.get('TradeID')
            if trade_id in leg_direction_map:
                # Update with inferred direction
                inferred_buy_sell = leg_direction_map[trade_id]
                exec_record['Buy/Sell'] = inferred_buy_sell
                # Update quantity sign based on inferred direction
                original_quantity = abs(exec_record.get('Quantity', 0))
                exec_record['Quantity'] = original_quantity if inferred_buy_sell == 'BUY' else -original_quantity
                # Recalculate NetCash based on inferred direction
                price_per_share = abs(exec_record.get('Price', 0))
                if inferred_buy_sell == 'BUY':
                    exec_record['NetCash'] = -(price_per_share * original_quantity * 100)  # Money out
                else:  # SELL
                    exec_record['NetCash'] = price_per_share * original_quantity * 100  # Money in
        
        # Log each order as JSON in debug mode (without account data)
        if logger.isEnabledFor(logging.DEBUG):
            for idx, exec_record in enumerate(execution_data):
                exec_record_log = exec_record.copy()
                exec_record_log.pop('Account', None)  # Remove account data from log
                # Convert datetime to string for JSON serialization
                if 'DateTime' in exec_record_log and isinstance(exec_record_log['DateTime'], datetime):
                    exec_record_log['DateTime'] = exec_record_log['DateTime'].isoformat()
                logger.debug(f"Execution {idx + 1}:\n{json.dumps(exec_record_log, indent=2, default=str)}")
        
        if not execution_data:
            logger.warning("No option executions found after filtering")
            return pd.DataFrame()
        
        df = pd.DataFrame(execution_data)
        logger.info(f"Converted {len(df)} executions to DataFrame format")
        logger.debug(f"Columns: {list(df.columns)}")
        
        # Log DataFrame without account data for security
        if logger.isEnabledFor(logging.DEBUG) and not df.empty:
            df_to_log = df.drop(columns=['Account'], errors='ignore')
            logger.debug(f"DataFrame shape: {df_to_log.shape}")
            logger.debug(f"First few rows (account data excluded):\n{df_to_log.head()}")
            
            # Save all trades to CSV in reports folder (without account data)
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = f"reports/tws_executions_debug_{timestamp}.csv"
                Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
                # Save without account data
                df_to_save = df.drop(columns=['Account'], errors='ignore')
                df_to_save.to_csv(csv_path, index=False)
                logger.debug(f"Saved all executions to CSV (account data excluded): {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to save debug CSV: {e}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching executions from TWS API: {e}", exc_info=True)
        raise
    finally:
        # Don't disconnect - let cleanup_ib_connection handle it
        pass


def summarize_vertical_put_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a TWS executions DataFrame, detect vertical put credit spreads per combo (ParentID) 
    and timestamp, and return IB-style summaries: quantity, price per spread, etc.
    
    Args:
        df: DataFrame with execution data from fetch_tws_executions()
        
    Returns:
        DataFrame with spread-level summaries
    """
    detector = _get_strategy_detector()
    if detector:
        return detector.summarize_vertical_put_spreads(df)
    # Fallback to empty DataFrame if detector not available
    return pd.DataFrame()


def group_tws_executions_by_combo(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Group TWS API executions by parentId (combo order ID) to match IB's Trade History Summary.
    Detects vertical put credit spreads and groups them correctly with quantity and price.
    
    Args:
        df: DataFrame with execution data from fetch_tws_executions()
        
    Returns:
        List of strategy dictionaries (same format as group_multi_leg_strategies)
    """
    detector = _get_strategy_detector()
    if detector:
        return detector.group_executions_by_combo(df)
    # Fallback to empty list if detector not available
    return []
    
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
            strategy_id = generate_strategy_id(legs_for_id, symbol)
        
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


def _group_tws_executions_raw(df: pd.DataFrame) -> List[Dict[str, Any]]:
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
            strategy_id_str = generate_strategy_id(legs_for_id, underlying)
        
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


def group_multi_leg_strategies(df: pd.DataFrame, group_by_strategy: bool = True) -> List[Dict[str, Any]]:
    """
    Group trades to identify multi-leg strategies.
    
    Args:
        df: DataFrame with trade data
        group_by_strategy: If True, group by date/time, underlying, and expiry (default).
                          If False, group by OrderID (IB assigns one OrderID per leg).
        
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
    
    # Get column names (use normalized or original)
    symbol_col = 'Symbol' if 'Symbol' in legs_with_order.columns else ([c for c in legs_with_order.columns if 'symbol' in c.lower()] or ['Symbol'])[0]
    strike_col = 'Strike' if 'Strike' in legs_with_order.columns else ([c for c in legs_with_order.columns if 'strike' in c.lower()] or ['Strike'])[0]
    expiry_col = 'Expiry' if 'Expiry' in legs_with_order.columns else ([c for c in legs_with_order.columns if 'expiry' in c.lower()] or ['Expiry'])[0]
    put_call_col = put_call_cols[0] if put_call_cols else 'Put/Call'
    buy_sell_col = 'Buy/Sell' if 'Buy/Sell' in legs_with_order.columns else ([c for c in legs_with_order.columns if 'buy' in c.lower() and 'sell' in c.lower()] or ['Buy/Sell'])[0]
    quantity_col = 'Quantity' if 'Quantity' in legs_with_order.columns else ([c for c in legs_with_order.columns if 'quantity' in c.lower()] or ['Quantity'])[0]
    price_col = 'Price' if 'Price' in legs_with_order.columns else ([c for c in legs_with_order.columns if 'price' in c.lower()] or ['Price'])[0]
    
    # Check for TradeID column to filter out summary rows
    tradeid_cols = [col for col in legs_with_order.columns if 'tradeid' in col.lower() or 'trade_id' in col.lower()]
    if tradeid_cols:
        tradeid_col = tradeid_cols[0]
        logger.debug(f"Found TradeID column: {tradeid_col}")
        logger.debug(f"Rows with TradeID: {legs_with_order[tradeid_col].notna().sum()}, without: {legs_with_order[tradeid_col].isna().sum()}")
        # Filter to only rows with TradeID (actual executions, not summary rows)
        # Summary rows (no TradeID) are duplicates of actual executions
        before_count = len(legs_with_order)
        legs_with_order = legs_with_order[legs_with_order[tradeid_col].notna()].copy()
        logger.debug(f"After TradeID filtering: {len(legs_with_order)} rows (removed {before_count - len(legs_with_order)} summary rows)")
    else:
        logger.debug("No TradeID column found, using all rows")
    
    # IB assigns separate OrderIDs to each leg of a multi-leg strategy
    # We need to group by OrderID first, then look for related OrderIDs that form strategies
    # Strategy grouping: OrderIDs with same date/time, underlying, and expiry that form a spread
    
    # First, aggregate each OrderID into a single leg (in case of multiple fills)
    orderid_legs = []
    grouped_by_order = legs_with_order.groupby(orderid_col)
    logger.debug(f"Grouped into {len(grouped_by_order)} unique OrderIDs")
    
    for order_id, group in grouped_by_order:
        # Aggregate this OrderID into a single leg
        leg_group_cols = [symbol_col, strike_col, expiry_col, put_call_col, buy_sell_col]
        leg_group_cols = [c for c in leg_group_cols if c in group.columns]
        
        if not leg_group_cols:
            continue
        
        # Group by leg characteristics (should be 1 unique leg per OrderID)
        leg_groups = group.groupby(leg_group_cols, dropna=False)
        
        for leg_key, leg_group in leg_groups:
            # Aggregate quantities and calculate weighted average price
            total_quantity = leg_group[quantity_col].sum()
            if total_quantity != 0:
                weighted_price = (leg_group[quantity_col] * leg_group[price_col]).sum() / abs(total_quantity)
            else:
                weighted_price = leg_group[price_col].mean()
            
            first_row = leg_group.iloc[0]
            
            # Get date/time
            date_cols = [col for col in group.columns if 'date' in col.lower() or 'time' in col.lower()]
            when = first_row[date_cols[0]] if date_cols else ''
            
            # Get underlying
            underlying_cols = [col for col in group.columns if 'underlying' in col.lower() and 'symbol' in col.lower()]
            underlying = first_row[underlying_cols[0]].strip() if underlying_cols and pd.notna(first_row[underlying_cols[0]]) else ''
            if not underlying:
                # Extract from symbol
                symbol_val = first_row.get(symbol_col, '')
                if isinstance(symbol_val, str):
                    underlying = symbol_val.split()[0] if symbol_val.split() else ''
            
            # Get NetCash and Commission columns
            netcash_col = 'NetCash' if 'NetCash' in group.columns else ([c for c in group.columns if 'netcash' in c.lower() or 'net_cash' in c.lower()] or [None])[0]
            commission_col = 'Commission' if 'Commission' in group.columns else ([c for c in group.columns if 'commission' in c.lower()] or [None])[0]
            
            # Track individual fills for this leg
            fills = []
            for _, fill_row in leg_group.iterrows():
                fill_qty = fill_row.get(quantity_col, 0)
                fill_price = fill_row.get(price_col, 0)
                fill_date_cols = [col for col in leg_group.columns if 'date' in col.lower() or 'time' in col.lower()]
                fill_when = fill_row[fill_date_cols[0]] if fill_date_cols else when
                fills.append({
                    'quantity': fill_qty,
                    'price': fill_price,
                    'when': fill_when
                })
            
            orderid_legs.append({
                'OrderID': order_id,
                'Symbol': first_row.get(symbol_col, 'N/A'),
                'Strike': first_row.get(strike_col, 'N/A'),
                'Expiry': first_row.get(expiry_col, 'N/A'),
                'Put/Call': first_row.get(put_call_col, 'N/A'),
                'Buy/Sell': first_row.get(buy_sell_col, 'N/A'),
                'Quantity': total_quantity,
                'Price': weighted_price,
                'When': when,
                'Underlying': underlying,
                'NetCash': group[netcash_col].sum() if netcash_col and netcash_col in group.columns else 0,
                'Commission': group[commission_col].sum() if commission_col and commission_col in group.columns else 0,
                'Fills': fills  # Store individual fills
            })
    
    logger.debug(f"Aggregated to {len(orderid_legs)} OrderID legs")
    
    df_legs = pd.DataFrame(orderid_legs)
    
    if group_by_strategy:
        # Group by strategy: date/time, underlying, and expiry
        logger.info("Grouping by strategy (date/time, underlying, expiry)")
        date_cols_legs = [col for col in df_legs.columns if 'when' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
        if not date_cols_legs:
            logger.warning("No date/time column found for strategy grouping")
            return []
        
        date_col_legs = date_cols_legs[0]
        
        # Group by date/time, underlying, and expiry
        strategy_groups = df_legs.groupby([date_col_legs, 'Underlying', 'Expiry'], dropna=False)
        logger.debug(f"Found {len(strategy_groups)} potential strategy groups (by date/time, underlying, expiry)")
    else:
        # Group by OrderID (original method - IB assigns one OrderID per leg)
        # Note: Since IB typically assigns one OrderID per leg, this will only find
        # OrderIDs that somehow have multiple unique legs (rare case)
        logger.info("Grouping by OrderID")
        
        # Go back to original data to check for OrderIDs with multiple legs
        orderid_grouped = legs_with_order.groupby(orderid_col)
        strategy_groups_list = []
        
        for order_id, group in orderid_grouped:
            # Within each OrderID, group by unique leg characteristics
            leg_group_cols = [symbol_col, strike_col, expiry_col, put_call_col, buy_sell_col]
            leg_group_cols = [c for c in leg_group_cols if c in group.columns]
            
            if not leg_group_cols:
                continue
            
            leg_groups = group.groupby(leg_group_cols, dropna=False)
            
            # Only include if this OrderID has 2+ unique legs
            if len(leg_groups) >= 2:
                # Create aggregated legs for this OrderID
                orderid_legs_for_strategy = []
                for leg_key, leg_group in leg_groups:
                    total_quantity = leg_group[quantity_col].sum()
                    if total_quantity != 0:
                        weighted_price = (leg_group[quantity_col] * leg_group[price_col]).sum() / abs(total_quantity)
                    else:
                        weighted_price = leg_group[price_col].mean()
                    
                    first_row = leg_group.iloc[0]
                    date_cols = [col for col in group.columns if 'date' in col.lower() or 'time' in col.lower()]
                    when = first_row[date_cols[0]] if date_cols else ''
                    
                    underlying_cols = [col for col in group.columns if 'underlying' in col.lower() and 'symbol' in col.lower()]
                    underlying = first_row[underlying_cols[0]].strip() if underlying_cols and pd.notna(first_row[underlying_cols[0]]) else ''
                    if not underlying:
                        symbol_val = first_row.get(symbol_col, '')
                        if isinstance(symbol_val, str):
                            underlying = symbol_val.split()[0] if symbol_val.split() else ''
                    
                    netcash_col = 'NetCash' if 'NetCash' in group.columns else ([c for c in group.columns if 'netcash' in c.lower() or 'net_cash' in c.lower()] or [None])[0]
                    commission_col = 'Commission' if 'Commission' in group.columns else ([c for c in group.columns if 'commission' in c.lower()] or [None])[0]
                    
                    orderid_legs_for_strategy.append({
                        'OrderID': order_id,
                        'Symbol': first_row.get(symbol_col, 'N/A'),
                        'Strike': first_row.get(strike_col, 'N/A'),
                        'Expiry': first_row.get(expiry_col, 'N/A'),
                        'Put/Call': first_row.get(put_call_col, 'N/A'),
                        'Buy/Sell': first_row.get(buy_sell_col, 'N/A'),
                        'Quantity': total_quantity,
                        'Price': weighted_price,
                        'When': when,
                        'Underlying': underlying,
                        'NetCash': leg_group[netcash_col].sum() if netcash_col and netcash_col in leg_group.columns else 0,
                        'Commission': leg_group[commission_col].sum() if commission_col and commission_col in leg_group.columns else 0
                    })
                
                strategy_groups_list.append((order_id, pd.DataFrame(orderid_legs_for_strategy)))
        
        logger.debug(f"Found {len(strategy_groups_list)} OrderIDs with multiple legs")
        
        # Convert to a format compatible with the loop below
        class OrderIDStrategyGroups:
            def __init__(self, groups_list):
                self.groups_list = groups_list
            
            def __iter__(self):
                for order_id, group_df in self.groups_list:
                    yield (order_id, group_df)
        
        strategy_groups = OrderIDStrategyGroups(strategy_groups_list)
    
    strategies = []
    single_leg_count = 0
    
    for strategy_key, strategy_group in strategy_groups:
        if group_by_strategy:
            when_key, underlying_key, expiry_key = strategy_key
            logger.debug(f"Processing strategy group: {when_key}, {underlying_key}, {expiry_key} with {len(strategy_group)} legs")
        else:
            # Grouping by OrderID
            order_id = strategy_key
            when_key = strategy_group['When'].iloc[0] if 'When' in strategy_group.columns and len(strategy_group) > 0 else ''
            underlying_key = strategy_group['Underlying'].iloc[0] if 'Underlying' in strategy_group.columns and len(strategy_group) > 0 else ''
            expiry_key = strategy_group['Expiry'].iloc[0] if 'Expiry' in strategy_group.columns and len(strategy_group) > 0 else ''
            logger.debug(f"Processing OrderID {order_id} with {len(strategy_group)} legs")
        
        # Only consider multi-leg strategies (2+ unique legs)
        if len(strategy_group) < 2:
            single_leg_count += 1
            logger.debug(f"Skipping strategy group - only {len(strategy_group)} leg(s)")
            continue
        
        # Build leg descriptions (simple - just show the legs)
        legs_desc = []
        leg_list = []
        order_ids = []
        
        for _, leg_row in strategy_group.iterrows():
            symbol = leg_row.get('Symbol', 'N/A')
            expiry = leg_row.get('Expiry', 'N/A')
            strike = leg_row.get('Strike', 'N/A')
            put_call = leg_row.get('Put/Call', 'N/A')
            buy_sell = leg_row.get('Buy/Sell', 'N/A')
            quantity = leg_row.get('Quantity', 0)
            price = leg_row.get('Price', 0)
            orderid = leg_row.get('OrderID', '')
            
            # Format expiry if it's a number (YYYYMMDD)
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
        
        # Sort legs by Symbol, Strike, Put/Call for consistent display
        leg_list_sorted = sorted(leg_list, key=lambda x: (x['symbol'], x['strike'], x['put_call']))
        
        for leg in leg_list_sorted:
            leg_desc = f"{leg['buy_sell']} {abs(int(leg['quantity']))} x {leg['symbol']} {leg['expiry']} {leg['strike']}{leg['put_call']}"
            legs_desc.append(leg_desc)
            if leg['orderid'] and pd.notna(leg['orderid']):
                order_ids.append(str(int(leg['orderid'])))
        
        # Format date/time for grouping
        if isinstance(when_key, pd.Timestamp):
            when = when_key.strftime('%Y-%m-%d %H:%M:%S')
        else:
            when = str(when_key) if when_key else 'N/A'
        
        # Calculate strategy-level buy/sell prices from NetCash
        # For credit spreads: prices are negative (credit received/paid)
        # For debit spreads: prices are positive (debit paid/received)
        # Find all transactions for all legs in this strategy from original data
        strategy_buys = []  # All buy transactions (preserve sign)
        strategy_sells = []  # All sell transactions (preserve sign)
        purchase_times = []  # Track purchase times for date/time
        
        # Get all OrderIDs in this strategy
        strategy_order_ids = strategy_group['OrderID'].unique() if 'OrderID' in strategy_group.columns else []
        
        # Find all transactions for legs in this strategy
        for _, leg_row in strategy_group.iterrows():
            symbol = leg_row.get('Symbol', 'N/A')
            strike = leg_row.get('Strike', 'N/A')
            expiry = leg_row.get('Expiry', 'N/A')
            put_call = leg_row.get('Put/Call', 'N/A')
            
            # Format expiry for comparison
            if isinstance(expiry, (int, float)) and expiry > 0:
                expiry_num = int(expiry)
            else:
                continue
            
            # Find all transactions for this exact option
            leg_transactions = option_trades[
                (option_trades[symbol_col] == symbol) &
                (option_trades[strike_col] == strike) &
                (option_trades[expiry_col] == expiry_num) &
                (option_trades[put_call_col] == put_call)
            ].copy()
            
            # Only include rows with TradeID (actual executions)
            if tradeid_cols:
                leg_transactions = leg_transactions[leg_transactions[tradeid_cols[0]].notna()]
            
            # Collect buys and sells based on NetCash
            # For credit spreads: buy price is negative (credit received), sell price is negative (credit paid to close)
            # For debit spreads: buy price is positive (debit paid), sell price is positive (debit received)
            
            for _, trans_row in leg_transactions.iterrows():
                trans_qty = trans_row.get(quantity_col, 0)
                trans_netcash = trans_row.get(netcash_col, 0) if netcash_col and netcash_col in trans_row.index else 0
                
                # Get transaction date/time
                date_cols = [col for col in leg_transactions.columns if 'date' in col.lower() or 'time' in col.lower()]
                trans_when = None
                if date_cols:
                    trans_when = trans_row.get(date_cols[0])
                    if pd.notna(trans_when) and trans_qty > 0:  # Only track purchase times (BUY transactions)
                        purchase_times.append(trans_when)
                
                if trans_qty > 0:  # BUY - opening the position
                    # For credit spreads: NetCash is negative (credit received)
                    # For debit spreads: NetCash is positive (debit paid)
                    strategy_buys.append(trans_netcash)  # Keep sign: negative for credit, positive for debit
                elif trans_qty < 0:  # SELL - closing the position
                    # For credit spreads: NetCash is negative (credit paid to close)
                    # For debit spreads: NetCash is positive (debit received)
                    strategy_sells.append(trans_netcash)  # Keep sign: negative for credit, positive for debit
        
        # Calculate strategy totals (preserve signs)
        total_buy_price = sum(strategy_buys) if strategy_buys else 0  # Negative for credit spreads
        total_sell_price = sum(strategy_sells) if strategy_sells else 0  # Negative for credit spreads
        
        # Calculate P&L
        # For credit spreads: buy_price is negative, sell_price is negative
        # Profit = sell_price - buy_price (e.g., -0.02 - (-0.24) = +0.22)
        # For debit spreads: buy_price is positive, sell_price is positive
        # Profit = sell_price - buy_price (e.g., 0.50 - 0.30 = +0.20)
        strategy_pnl = total_sell_price - total_buy_price  # NetCash already accounts for commissions
        
        # Get purchase date/time (earliest transaction from all legs)
        purchase_datetime = None
        if purchase_times:
            # Convert to datetime and find earliest
            purchase_datetimes = []
            for pt in purchase_times:
                if pd.notna(pt):
                    if isinstance(pt, pd.Timestamp):
                        purchase_datetimes.append(pt)
                    else:
                        try:
                            purchase_datetimes.append(pd.to_datetime(pt))
                        except:
                            pass
            if purchase_datetimes:
                purchase_datetime = min(purchase_datetimes)
        
        # Format purchase date/time for display
        purchase_datetime_str = None
        if purchase_datetime:
            if isinstance(purchase_datetime, pd.Timestamp):
                purchase_datetime_str = purchase_datetime.strftime('%Y-%m-%d %H:%M:%S')
            else:
                purchase_datetime_str = str(purchase_datetime)
        
        # Calculate totals from strategy_group (for display)
        net_cash = strategy_group['NetCash'].sum()
        commission = strategy_group['Commission'].sum()
        
        # Create strategy ID
        if group_by_strategy:
            # Multiple OrderIDs for multi-leg strategies
            strategy_id = ', '.join(order_ids) if order_ids else 'N/A'
        else:
            # Single OrderID
            strategy_id = str(int(strategy_key)) if pd.notna(strategy_key) else 'N/A'
        
        # Extract BAG_ID from any execution in the strategy group (if available)
        # Note: BAG_ID is only available for TWS API executions, not Flex Query
        bag_id = None
        if 'BAG_ID' in strategy_group.columns and not strategy_group.empty:
            bag_id = strategy_group['BAG_ID'].iloc[0] if strategy_group['BAG_ID'].notna().any() else None
        
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
            strategy_id_str = generate_strategy_id(legs_for_id, underlying_key if underlying_key else 'N/A')
        
        strategy = {
            'OrderID': strategy_id,
            'BAG_ID': bag_id,  # BAG ID from TradeID (TWS API only)
            'StrategyID': strategy_id_str,  # Human-readable strategy identifier
            'NumLegs': len(strategy_group),
            'Underlying': underlying_key if underlying_key else 'N/A',
            'When': when,
            'PurchaseDateTime': purchase_datetime_str if purchase_datetime_str else when,  # Purchase date/time
            'Legs': legs_desc,
            'BuyPrice': total_buy_price,  # Price to open (negative for credit spreads, positive for debit)
            'SellPrice': total_sell_price,  # Price to close (negative for credit spreads, positive for debit)
            'PnL': strategy_pnl,  # Profit/Loss for the strategy
            'NetCash': net_cash,
            'Commission': commission,
            'TotalCost': net_cash + commission
        }
        
        strategies.append(strategy)
        logger.debug(f"Strategy group - {len(strategy_group)} legs, NetCash={net_cash:.2f}, Commission={commission:.2f}")
    
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
            'StrategyID': strategy.get('StrategyID') or 'N/A',  # Human-readable strategy identifier (only for BAG orders)
            'Quantity': strategy.get('Quantity', strategy['NumLegs']),  # Number of spreads
            'NumLegs': strategy['NumLegs'],
            'SpreadType': strategy.get('SpreadType', ''),
            'Underlying': strategy['Underlying'],
            'When': strategy['When'],
            'PurchaseDateTime': strategy.get('PurchaseDateTime', strategy['When']),
            'Legs': ' | '.join(strategy['Legs']),
            'BuyPrice': strategy.get('BuyPrice', 0),
            'SellPrice': strategy.get('SellPrice', 0),
            'PnL': strategy.get('PnL', 0),
            'NetCash': strategy['NetCash'],
            'Commission': strategy['Commission'],
            'TotalCost': strategy['TotalCost']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by PurchaseDateTime (DateTime column) - oldest first, matching HTML report order
    if 'PurchaseDateTime' in df.columns:
        # Convert to datetime for proper sorting
        df['PurchaseDateTime'] = pd.to_datetime(df['PurchaseDateTime'], errors='coerce')
        df = df.sort_values('PurchaseDateTime', na_position='last')
        # Convert back to string for CSV output (preserve original format if possible)
        df['PurchaseDateTime'] = df['PurchaseDateTime'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ''
        )
    
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
        .strategy-details {{
            margin-top: 15px;
            padding: 15px;
            background-color: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }}
        .strategy-details h3 {{
            margin-top: 0;
            margin-bottom: 10px;
            color: #4CAF50;
            font-size: 1em;
        }}
        .strategy-detail-item {{
            margin: 8px 0;
            padding: 5px 0;
        }}
        .strategy-detail-label {{
            font-weight: bold;
            color: #666;
            margin-right: 10px;
        }}
        .strategy-detail-value {{
            color: #333;
            font-size: 1.05em;
        }}
        .pnl-positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .pnl-negative {{
            color: #f44336;
            font-weight: bold;
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
    
    # Sort strategies by date (PurchaseDateTime, matching CSV DateTime column)
    # Parse date strings to datetime for proper sorting
    def get_strategy_datetime(strategy):
        """Extract datetime from strategy for sorting (matches CSV DateTime column)."""
        # Use PurchaseDateTime (same as DateTime column in CSV)
        date_str = strategy.get('PurchaseDateTime') or strategy.get('When', '')
        if not date_str or date_str == 'N/A':
            return pd.Timestamp.max  # Put invalid dates at the end
        
        try:
            # Try parsing common date formats
            if isinstance(date_str, pd.Timestamp):
                return date_str
            # Try ISO format first
            try:
                return pd.to_datetime(date_str)
            except:
                # Try other formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                # Last resort: let pandas infer
                result = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(result):
                    return pd.Timestamp.max
                return result
        except:
            return pd.Timestamp.max
    
    # Sort strategies by datetime (oldest first, matching CSV order)
    strategies_sorted = sorted(strategies, key=get_strategy_datetime, reverse=False)
    
    # Add each strategy
    for strategy in strategies_sorted:
        # Show spread type if available (for vertical spreads)
        spread_type = strategy.get('SpreadType', '')
        quantity = strategy.get('Quantity', strategy['NumLegs'])
        
        # Use BAG_ID if available, otherwise fall back to OrderID
        display_id = strategy.get('BAG_ID') or strategy.get('OrderID', 'N/A')
        
        html += f"""
    <div class="strategy">
        <div class="strategy-header">
            <div class="strategy-info">
                <div class="strategy-id">BAG ID: {display_id}</div>
                <div class="strategy-meta">
                    {quantity} spread(s) | {strategy['Underlying']} | {strategy['When']}
                    {f' | {spread_type}' if spread_type else f' | {strategy["NumLegs"]} legs'}
                </div>
            </div>
            <div class="strategy-financials">
                <div class="net-cash">${strategy['NetCash']:,.2f}</div>
                <div class="commission">Commission: ${strategy['Commission']:,.2f}</div>
            </div>
        </div>
        <div class="legs">
"""
        
        # Show leg summaries
        for leg in strategy['Legs']:
            html += f'            <div class="leg">{leg}</div>\n'
        
        html += """        </div>
        <div class="strategy-details">
            <h3>Strategy Details</h3>
"""
        
        # Show purchase date/time
        purchase_datetime = strategy.get('PurchaseDateTime', strategy.get('When', 'N/A'))
        if purchase_datetime and purchase_datetime != 'N/A':
            html += f"""
            <div class="strategy-detail-item">
                <span class="strategy-detail-label">Purchased:</span>
                <span class="strategy-detail-value">{purchase_datetime}</span>
            </div>
"""
        
        # Show strategy-level buy/sell prices and P&L
        # For credit spreads: prices are negative (credit received/paid)
        # Opening: BOT (buy spread, receive credit) - negative price
        # Closing: SLD (sell spread, pay to close) - negative price
        quantity = strategy.get('Quantity', 1)
        open_price_per_spread = strategy.get('OpenPrice')
        close_price_per_spread = strategy.get('ClosePrice')
        buy_price = strategy.get('BuyPrice', 0)
        sell_price = strategy.get('SellPrice', 0)
        strategy_pnl = strategy.get('PnL', 0)
        
        # Show opening (BOT)
        if open_price_per_spread is not None:
            html += f"""
            <div class="strategy-detail-item">
                <span class="strategy-detail-label">BOT {quantity} @:</span>
                <span class="strategy-detail-value">${open_price_per_spread:,.2f}</span>
            </div>
"""
        elif buy_price != 0:
            html += f"""
            <div class="strategy-detail-item">
                <span class="strategy-detail-label">Bought for:</span>
                <span class="strategy-detail-value">${buy_price:,.2f}</span>
            </div>
"""
        
        # Show closing (SLD)
        if close_price_per_spread is not None:
            html += f"""
            <div class="strategy-detail-item">
                <span class="strategy-detail-label">SLD {quantity} @:</span>
                <span class="strategy-detail-value">${close_price_per_spread:,.2f}</span>
            </div>
"""
        elif sell_price != 0:
            html += f"""
            <div class="strategy-detail-item">
                <span class="strategy-detail-label">Sold for:</span>
                <span class="strategy-detail-value">${sell_price:,.2f}</span>
            </div>
"""
        
        # Show P&L
        if strategy_pnl != 0:
            pnl_class = 'pnl-positive' if strategy_pnl > 0 else 'pnl-negative'
            pnl_sign = '+' if strategy_pnl > 0 else ''
            html += f"""
            <div class="strategy-detail-item">
                <span class="strategy-detail-label">P&L:</span>
                <span class="strategy-detail-value {pnl_class}">{pnl_sign}${strategy_pnl:,.2f}</span>
            </div>
"""
        
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
  
  # Group by OrderID instead of strategy (default groups by strategy: date/time, underlying, expiry)
  python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report flex_report.csv --group-by-strategy false

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
    
    parser.add_argument(
        '--group-by-strategy',
        type=str,
        default='true',
        choices=['true', 'false', '1', '0', 'yes', 'no'],
        help='Group by strategy (date/time, underlying, expiry) instead of OrderID. Default: true. Set to false to group by OrderID instead. (Only applies to Flex Query mode)'
    )
    
    parser.add_argument(
        '--data-source',
        type=str,
        default='tws',
        choices=['tws', 'flex'],
        help='Data source: tws (TWS API - default, matches IB Trade History Summary exactly) or flex (Flex Query Web Service - fallback for historical data)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='IB API port number (for TWS API mode, default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    
    # Add file handler for DEBUG level logs
    if log_level == logging.DEBUG:
        # Create logs/api/ib directory if it doesn't exist
        log_dir = Path('logs/api/ib')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'ib_flex_multi_leg_report_{timestamp}.log'
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Debug logs will be saved to: {log_file}")
    
    logger.debug(f"Logging level set to {args.log_level}")
    
    # Suppress ib_insync debug logs to avoid exposing account data
    # ib_insync logs raw API messages that may contain sensitive information
    if log_level == logging.DEBUG:
        # Set ib_insync loggers to WARNING to prevent account data exposure
        logging.getLogger('ib_insync').setLevel(logging.WARNING)
        logging.getLogger('ib_insync.client').setLevel(logging.WARNING)
        logging.getLogger('ib_insync.ib').setLevel(logging.WARNING)
        logging.getLogger('ib_insync.wrapper').setLevel(logging.WARNING)
        logger.debug("Suppressed ib_insync debug logs to prevent account data exposure")
    
    # Parse since date
    since_date = None
    if args.since:
        try:
            since_date = datetime.strptime(args.since, '%Y-%m-%d')
            logger.info(f"Filtering trades on or after: {since_date.strftime('%Y-%m-%d')}")
        except ValueError:
            logger.error(f"Invalid date format: {args.since}. Use YYYY-MM-DD format (e.g., 2025-01-01)")
            sys.exit(1)
    
    # Determine data source
    use_tws_api = args.data_source == 'tws' and not args.flex_report
    
    if use_tws_api:
        # Use TWS API (default) - matches IB's Trade History Summary exactly
        logger.info("Using TWS API to fetch executions (matches IB Trade History Summary)")
        
        if not IB_INSYNC_AVAILABLE:
            logger.error("ib_insync not available. Install with: pip install ib_insync")
            logger.error("Alternatively, use --data-source flex for Flex Query mode")
            sys.exit(1)
        
        # Detect port if not provided
        selected_port = args.port
        if not selected_port:
            try:
                selected_port = detect_ib_port()
                if selected_port:
                    logger.info(f"Auto-detected IB port: {selected_port}")
                else:
                    logger.warning("Could not auto-detect IB port, using default 4001")
                    selected_port = 4001
            except Exception as e:
                logger.warning(f"Port auto-detection failed: {e}, using default 4001")
                selected_port = 4001
        
        # Fetch executions from TWS API
        try:
            df = fetch_tws_executions(since_date=since_date, port=selected_port)
            if df.empty:
                logger.warning("No executions found from TWS API")
                sys.exit(0)
            
            # Group by combo order (parentId) - matches IB's grouping exactly
            strategies = group_tws_executions_by_combo(df)
            if not strategies:
                logger.warning("No multi-leg strategies found in TWS executions")
                sys.exit(0)
                
        except Exception as e:
            logger.error(f"Error fetching from TWS API: {e}")
            logger.error("Falling back to Flex Query mode. Use --data-source flex to use Flex Query directly.")
            sys.exit(1)
        finally:
            cleanup_ib_connection()
    
    else:
        # Use Flex Query (fallback or explicitly requested)
        logger.info("Using Flex Query Web Service")
        
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
        
        # Filter by date
        if since_date:
            df = filter_by_date(df, since_date)
            if df.empty:
                logger.warning(f"No trades found on or after {since_date.strftime('%Y-%m-%d')}")
                sys.exit(0)
        
        # Group into multi-leg strategies
        # Convert string argument to boolean
        group_by_strategy = args.group_by_strategy.lower() in ['true', '1', 'yes']
        strategies = group_multi_leg_strategies(df, group_by_strategy=group_by_strategy)
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

