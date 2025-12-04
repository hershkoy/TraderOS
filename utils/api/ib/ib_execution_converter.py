"""
IB Execution Data Converter

Converts raw IB API Fill objects to standardized DataFrame format for reporting.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from ib_insync import Fill
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    Fill = None

logger = logging.getLogger(__name__)


class ExecutionConverter:
    """
    Converts IB API Fill objects to DataFrame format.
    
    Handles:
    - BAG executions (combo order prices)
    - OPT executions (individual leg executions)
    - Buy/Sell direction mapping
    - NetCash calculation
    - ParentID/BAG_ID extraction
    """
    
    def __init__(self):
        """Initialize the converter."""
        self.bag_prices = {}  # Store BAG execution prices by parent_id
    
    def convert_fills_to_dataframe(
        self, 
        fills: List[Fill], 
        since_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Convert IB API Fill objects to DataFrame format.
        
        Args:
            fills: List of Fill objects from ib.reqExecutions()
            since_date: Optional date filter (applied after conversion)
            
        Returns:
            DataFrame with execution data in standardized format
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError("ib_insync not available. Install with: pip install ib_insync")
        
        logger.info(f"Converting {len(fills)} fills to DataFrame format")
        
        execution_data = []
        
        # First pass: capture BAG prices and collect option executions
        for fill in fills:
            contract = fill.contract
            execution = fill.execution
            
            # Capture BAG execution prices (total combo price)
            if contract.secType == 'BAG':
                parent_id = self._extract_parent_id(execution)
                if parent_id:
                    # BAG price is the total combo price (already per contract, multiply by 100)
                    bag_price = execution.price * execution.shares * 100
                    self.bag_prices[parent_id] = bag_price
                    logger.debug(f"Captured BAG price for ParentID {parent_id}: "
                               f"{execution.price} * {execution.shares} * 100 = {bag_price}")
                continue  # Skip BAG executions from leg processing
            
            # Skip non-option trades
            if contract.secType != 'OPT':
                logger.debug(f"Skipping non-option execution: secType={contract.secType}, "
                           f"symbol={contract.symbol}")
                continue
            
            # Convert option execution to record
            exec_record = self._convert_option_execution(fill, execution, contract)
            if exec_record:
                execution_data.append(exec_record)
        
        if not execution_data:
            logger.warning("No option executions found after conversion")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(execution_data)
        
        # Filter by date if specified
        if since_date:
            df = self._filter_by_date(df, since_date)
        
        # Store BAG prices as DataFrame metadata
        if hasattr(df, 'attrs'):
            df.attrs['bag_prices'] = self.bag_prices
        else:
            # Fallback: store in instance variable
            df._bag_prices = self.bag_prices
        
        logger.info(f"Converted {len(df)} option executions to DataFrame format")
        logger.debug(f"Columns: {list(df.columns)}")
        
        return df
    
    def _convert_option_execution(
        self, 
        fill: Fill, 
        execution: Any, 
        contract: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a single option execution to a record dictionary.
        
        Args:
            fill: Fill object containing execution, contract, commissionReport, time
            execution: Execution object from fill.execution
            contract: Contract object from fill.contract
            
        Returns:
            Dictionary with execution record data, or None if should be skipped
        """
        # Extract option details
        symbol = contract.symbol
        expiry = contract.lastTradeDateOrContractMonth  # Format: YYYYMMDD
        strike = contract.strike
        right = contract.right  # 'P' or 'C'
        
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
        
        # Extract parent ID (combo order ID)
        parent_id = self._extract_parent_id(execution)
        
        # Extract BAG ID from TradeID (execId format: bag_id.leg_id.leg_number.execution_id)
        bag_id = None
        if execution.execId:
            parts = execution.execId.split('.')
            if len(parts) > 0:
                bag_id = parts[0]  # First part is the BAG ID
        
        # Map execution side to Buy/Sell
        # execution.side: 'B' = BOT (Buy), 'S' = SLD (Sell)
        # For Buy: Quantity positive, NetCash negative (money out)
        # For Sell: Quantity negative, NetCash positive (money in)
        buy_sell = 'Buy' if execution.side == 'B' else 'SELL'
        
        # Calculate quantity and NetCash
        price_per_share = execution.price
        shares = execution.shares
        
        # Quantity: positive for Buy, negative for Sell
        quantity = shares if execution.side == 'B' else -shares
        
        # NetCash: negative for Buy (money out), positive for Sell (money in)
        # NetCash = price * shares * 100 (contract multiplier)
        # For Buy: NetCash is negative (money out)
        # For Sell: NetCash is positive (money in)
        if execution.side == 'B':  # Buy
            net_cash = -(price_per_share * shares * 100)  # Negative (money out)
        else:  # Sell
            net_cash = price_per_share * shares * 100  # Positive (money in)
        
        # Commission (from commissionReport if available)
        commission = 0.0
        if fill.commissionReport:
            commission = fill.commissionReport.commission
        
        # OrderID: use parent_id (BAG ID) as OrderID for combo orders
        order_id = parent_id if parent_id else (execution.orderId if execution.orderId and execution.orderId != 0 else None)
        
        # Build execution record
        exec_record = {
            'OrderID': order_id,
            'TradeID': execution.execId,
            'BAG_ID': bag_id,
            'ParentID': parent_id,
            'Symbol': symbol,
            'Expiry': int(expiry) if expiry and expiry.isdigit() else expiry,
            'Strike': strike,
            'Put/Call': right,
            'Buy/Sell': buy_sell,
            'Quantity': quantity,
            'Price': price_per_share,
            'NetCash': net_cash,
            'Commission': commission,
            'Date/Time': exec_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(exec_time, datetime) else str(exec_time),
            'DateTime': exec_time,  # Keep as datetime for filtering
            'Exchange': execution.exchange,
            'Account': execution.acctNumber if hasattr(execution, 'acctNumber') else '',
        }
        
        return exec_record
    
    def _extract_parent_id(self, execution: Any) -> Optional[str]:
        """
        Extract parent ID (combo order ID) from execution.
        
        Args:
            execution: Execution object
            
        Returns:
            Parent ID string, or None if not found
        """
        # Try orderId first (for combo orders, all legs share the same orderId)
        if execution.orderId and execution.orderId != 0:
            return str(execution.orderId)
        
        # Try orderRef (alternative combo order identifier)
        if execution.orderRef:
            return execution.orderRef
        
        # Fallback: extract from execId format (bag_id.leg_id.leg_number.execution_id)
        if execution.execId:
            parts = execution.execId.split('.')
            if len(parts) > 1:
                return parts[0]  # First part is the BAG ID / parent ID
            else:
                return execution.execId
        
        return None
    
    def _filter_by_date(self, df: pd.DataFrame, since_date: datetime) -> pd.DataFrame:
        """
        Filter DataFrame by date.
        
        Args:
            df: DataFrame with execution data
            since_date: Filter executions on or after this date
            
        Returns:
            Filtered DataFrame
        """
        if 'DateTime' not in df.columns:
            logger.warning("No DateTime column found, cannot filter by date")
            return df
        
        # Handle timezone-aware vs timezone-naive datetime comparison
        # If DataFrame DateTime column is timezone-aware, make since_date timezone-aware too
        if not df.empty and df['DateTime'].dtype.tz is not None:
            # DataFrame has timezone-aware datetimes
            if since_date.tzinfo is None:
                # since_date is timezone-naive, make it timezone-aware (UTC)
                from datetime import timezone
                since_date = since_date.replace(tzinfo=timezone.utc)
                logger.debug(f"Converted since_date to timezone-aware (UTC): {since_date}")
        
        # Filter to executions on or after since_date
        filtered = df[df['DateTime'] >= since_date].copy()
        
        logger.info(f"Filtered to {len(filtered)} executions on or after "
                   f"{since_date.strftime('%Y-%m-%d')} (from {len(df)} total)")
        
        if not filtered.empty:
            logger.debug(f"Date range in filtered data: {filtered['DateTime'].min()} to "
                       f"{filtered['DateTime'].max()}")
        
        return filtered

