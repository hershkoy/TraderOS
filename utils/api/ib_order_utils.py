#!/usr/bin/env python3
"""
ib_order_utils.py

IB order creation, bracket orders, and order monitoring utilities.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

try:
    from ib_insync import Contract, Option, ComboLeg, Order, Trade, IB
except ImportError:
    raise ImportError("ib_insync not found. Install with: pip install ib_insync")

from ..data.fetch_data import get_ib_connection

logger = logging.getLogger(__name__)


def create_ib_bracket_order(
    spread_contract: Contract,
    parent_order: Order,
    take_profit_price: Optional[float] = None,
    stop_loss_price: Optional[float] = None,
    ib: Optional[IB] = None,
    port: Optional[int] = None
) -> Tuple[Trade, Optional[Trade], Optional[Trade]]:
    """
    Create a bracket order with optional take profit and stop loss.
    
    Args:
        spread_contract: The spread contract (BAG)
        parent_order: The parent limit order
        take_profit_price: Take profit price in IB format (optional)
        stop_loss_price: Stop loss price in IB format (optional)
        ib: IB connection (optional, will create if not provided)
        port: IB API port number (optional)
    
    Returns:
        Tuple of (parent_trade, take_profit_trade, stop_loss_trade)
        Child trades will be None if not specified
    """
    if ib is None:
        ib = get_ib_connection(port=port)
    
    # Place parent order
    parent_trade = ib.placeOrder(spread_contract, parent_order)
    parent_order_id = parent_trade.order.orderId
    
    take_profit_trade = None
    stop_loss_trade = None
    
    # Generate OCA group name for linking take profit and stop loss (One-Cancels-All)
    oca_group = f"bracket_{parent_order_id}"
    
    # Create take profit order if specified
    if take_profit_price is not None:
        tp_order = Order()
        tp_order.parentId = parent_order_id
        tp_order.action = "SELL" if parent_order.action == "BUY" else "BUY"
        tp_order.orderType = "LMT"
        tp_order.totalQuantity = parent_order.totalQuantity
        tp_order.lmtPrice = round(take_profit_price, 2)
        tp_order.tif = "GTC"
        tp_order.ocaGroup = oca_group
        tp_order.ocaType = 3
        if parent_order.account:
            tp_order.account = parent_order.account
        
        take_profit_trade = ib.placeOrder(spread_contract, tp_order)
        logger.info(
            f"Placed take profit order: {tp_order.action} @ ${take_profit_price:.2f} "
            f"(parent order ID: {parent_order_id}, OCA: {oca_group})"
        )
    
    # Create stop loss order if specified
    if stop_loss_price is not None:
        sl_order = Order()
        sl_order.parentId = parent_order_id
        sl_order.action = "SELL" if parent_order.action == "BUY" else "BUY"
        sl_order.orderType = "STP"
        sl_order.totalQuantity = parent_order.totalQuantity
        sl_order.auxPrice = round(stop_loss_price, 2)
        sl_order.tif = "GTC"
        sl_order.ocaGroup = oca_group
        sl_order.ocaType = 3
        sl_order.overridePercentageConstraints = True
        if parent_order.account:
            sl_order.account = parent_order.account
        
        stop_loss_trade = ib.placeOrder(spread_contract, sl_order)
        logger.info(
            f"Placed stop loss order: {sl_order.action} STP @ ${stop_loss_price:.2f} "
            f"(parent order ID: {parent_order_id}, OCA: {oca_group})"
        )
    
    return parent_trade, take_profit_trade, stop_loss_trade


def verify_bracket_order_structure(
    ib: IB,
    parent_order_id: int,
    expected_tp_price: Optional[float] = None,
    expected_sl_price: Optional[float] = None,
    timeout_seconds: float = 5.0
) -> bool:
    """
    Verify that bracket orders were created correctly in IB.
    
    Args:
        ib: IB connection
        parent_order_id: Parent order ID
        expected_tp_price: Expected take profit price
        expected_sl_price: Expected stop loss price
        timeout_seconds: How long to wait for orders to appear
    
    Returns:
        True if structure is as expected, False otherwise
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        ib.sleep(0.5)
        trades = ib.openTrades()
        
        parent_found = False
        tp_found = False
        sl_found = False
        
        for trade in trades:
            order = trade.order
            
            if order.orderId == parent_order_id:
                parent_found = True
                logger.info(f"  Parent order {parent_order_id}: {order.action} @ {order.lmtPrice} ({trade.orderStatus.status})")
            
            if order.parentId == parent_order_id:
                price = order.lmtPrice if order.orderType == "LMT" else order.auxPrice
                
                if expected_tp_price is not None and abs(price - expected_tp_price) < 0.01:
                    tp_found = True
                    logger.info(f"  Take profit order {order.orderId}: {order.action} @ {price} ({order.orderType}, {trade.orderStatus.status})")
                elif expected_sl_price is not None and abs(price - expected_sl_price) < 0.01:
                    sl_found = True
                    logger.info(f"  Stop loss order {order.orderId}: {order.action} @ {price} ({order.orderType}, {trade.orderStatus.status})")
                else:
                    logger.warning(f"  Unknown child order {order.orderId}: {order.action} @ {price} ({order.orderType})")
        
        tp_ok = (expected_tp_price is None) or tp_found
        sl_ok = (expected_sl_price is None) or sl_found
        
        if parent_found and tp_ok and sl_ok:
            logger.info("Bracket order structure verified successfully")
            return True
    
    if not parent_found:
        logger.error(f"Parent order {parent_order_id} not found in open trades")
    if expected_tp_price is not None and not tp_found:
        logger.error(f"Take profit order @ {expected_tp_price} not found")
    if expected_sl_price is not None and not sl_found:
        logger.error(f"Stop loss order @ {expected_sl_price} not found")
    
    return False


def calculate_bracket_prices(
    limit_price: float,
    order_action: str,
    take_profit_price_target: Optional[float] = None,
    stop_loss_multiplier: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate take profit and stop loss prices for bracket orders.
    
    Args:
        limit_price: The limit price in IB format (negative for credit spreads)
        order_action: "BUY" or "SELL"
        take_profit_price_target: The absolute price to close at for take profit (e.g., -0.02)
        stop_loss_multiplier: Stop loss multiplier on limit price (e.g., 1.5 means 1.5x the entry price)
    
    Returns:
        Tuple of (take_profit_price, stop_loss_price) in IB format, or None if not specified
    """
    take_profit_price = None
    stop_loss_price = None
    
    if take_profit_price_target is not None:
        take_profit_price = take_profit_price_target
    
    if stop_loss_multiplier is not None:
        stop_loss_price = limit_price * stop_loss_multiplier
    
    return take_profit_price, stop_loss_price


def calculate_buy_limit_price_sequence(
    initial_estimate: float,
    min_price: float = 0.21,
    price_reduction_per_minute: float = 0.01
) -> Tuple[float, float]:
    """
    Calculate the buy limit price sequence for credit spreads.
    
    Args:
        initial_estimate: Initial credit estimate (positive, e.g., 0.24)
        min_price: Minimum price to attempt (default: 0.21)
        price_reduction_per_minute: Price reduction per minute (default: 0.01)
    
    Returns:
        Tuple of (starting_price, min_price) both in IB format (negative for BUY orders)
    """
    starting_price_positive = round(initial_estimate + 0.01, 2)
    
    if starting_price_positive < min_price:
        starting_price_positive = min_price
    
    starting_price_ib = -abs(starting_price_positive)
    min_price_ib = -abs(min_price)
    
    logger.info(
        f"Price calculation: initial estimate=${initial_estimate:.2f}, "
        f"starting price=${abs(starting_price_ib):.2f} (estimate + $0.01), "
        f"minimum price=${abs(min_price_ib):.2f}"
    )
    
    return starting_price_ib, min_price_ib


def monitor_and_adjust_spread_order(
    trade: Trade,
    spread: Contract,
    initial_price: float,
    min_price: float = 0.23,
    initial_wait_minutes: int = 2,
    price_reduction_per_minute: float = 0.01,
    port: Optional[int] = None
) -> Trade:
    """
    Monitor a spread order and adjust price according to strategy.
    
    Args:
        trade: Trade object from placed order
        spread: Spread contract
        initial_price: Initial limit price (positive for SELL, negative for BUY in IB format)
        min_price: Minimum absolute price (default: 0.23)
        initial_wait_minutes: Minutes to wait at initial price (default: 2)
        price_reduction_per_minute: Price adjustment per minute after initial wait (default: 0.01)
        port: IB API port number (default: from IB_PORT env var or 4001)
    
    Returns:
        Trade object (may be filled, cancelled, or still pending)
    """
    ib = get_ib_connection(port=port)
    
    is_buy_order = trade.order.action == "BUY"
    
    if is_buy_order:
        logger.info(f"Monitoring BUY order. Initial price: ${initial_price:.2f} for {initial_wait_minutes} minutes")
        logger.info(f"After {initial_wait_minutes} minutes, will make less negative by ${price_reduction_per_minute:.2f} per minute (min: ${min_price:.2f})")
    else:
        logger.info(f"Monitoring SELL order. Initial price: ${initial_price:.2f} for {initial_wait_minutes} minutes")
        logger.info(f"After {initial_wait_minutes} minutes, will reduce by ${price_reduction_per_minute:.2f} per minute (min: ${min_price:.2f})")
    
    start_time = datetime.now()
    initial_wait_end = start_time + timedelta(minutes=initial_wait_minutes)
    current_price = initial_price
    last_price_reduction_time = initial_wait_end
    
    while True:
        ib.sleep(1)
        
        status = trade.orderStatus.status
        fill_price = trade.orderStatus.avgFillPrice
        
        current_time = datetime.now()
        
        elapsed_seconds = int((current_time - start_time).total_seconds())
        if elapsed_seconds > 0 and elapsed_seconds % 10 == 0:
            if elapsed_seconds >= 60:
                minutes = elapsed_seconds // 60
                seconds = elapsed_seconds % 60
                elapsed_str = f"{minutes}m {seconds}s"
            else:
                elapsed_str = f"{elapsed_seconds}s"
            
            logger.info(
                f"Order status: {status}, Current limit: ${current_price:.2f}, "
                f"Fill price: ${fill_price if fill_price > 0 else 'N/A'}, "
                f"Filled: {trade.orderStatus.filled}/{trade.order.totalQuantity}, "
                f"Elapsed: {elapsed_str}"
            )
        
        if status == 'Filled':
            logger.info(
                f"Order FILLED! Price: ${fill_price:.2f}, "
                f"Quantity: {trade.orderStatus.filled}/{trade.order.totalQuantity}"
            )
            return trade
        
        if status in ['Cancelled', 'ApiCancelled', 'Rejected']:
            reason = trade.orderStatus.whyHeld or 'N/A'
            if trade.log:
                error_messages = [log.message for log in trade.log if log.message]
                if error_messages:
                    reason = '; '.join(error_messages)
            logger.warning(f"Order {status.lower()}. Reason: {reason}")
            return trade
        
        if current_time >= initial_wait_end:
            minutes_since_reduction_start = (current_time - last_price_reduction_time).total_seconds() / 60.0
            
            if minutes_since_reduction_start >= 1.0:
                if is_buy_order:
                    new_price = current_price + price_reduction_per_minute
                    min_ib_price = min_price
                    if new_price > min_ib_price:
                        logger.info(
                            f"Price adjustment would go above minimum (${min_ib_price:.2f}). "
                            f"Reached minimum price. Stopping attempts."
                        )
                        ib.cancelOrder(trade.order)
                        logger.info("Order cancelled after reaching minimum price without fill.")
                        return trade
                else:
                    new_price = current_price - price_reduction_per_minute
                    if new_price < min_price:
                        logger.info(
                            f"Price reduction would go below minimum (${min_price:.2f}). "
                            f"Reached minimum price. Stopping attempts."
                        )
                        ib.cancelOrder(trade.order)
                        logger.info("Order cancelled after reaching minimum price without fill.")
                        return trade
                
                current_price = round(new_price, 2)
                trade.order.lmtPrice = current_price
                
                if is_buy_order:
                    logger.info("Adjusting limit price to $%.2f (making less negative)", current_price)
                else:
                    logger.info("Reducing limit price to $%.2f", current_price)
                
                try:
                    ib.placeOrder(spread, trade.order)
                    ib.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error modifying order: {e}")
                
                last_price_reduction_time = current_time
        
        if (current_time - start_time).total_seconds() > 3600:
            logger.warning("Order not filled after 1 hour. Cancelling...")
            ib.cancelOrder(trade.order)
            return trade


def monitor_all_orders(
    active_orders: List[Trade],
    ib: IB,
    ib_lock: threading.Lock,
    check_interval_seconds: int = 10
):
    """
    Monitor all active orders in a single loop, checking every x seconds.
    
    Args:
        active_orders: List of Trade objects to monitor (modified in place)
        ib: Shared IB connection object
        ib_lock: Thread lock for synchronizing IB access
        check_interval_seconds: How often to check/adjust orders (default: 10)
    """
    if not active_orders:
        return
    
    logger.info(f"Monitoring {len(active_orders)} active orders (checking every {check_interval_seconds}s)")
    
    order_start_times = {}
    order_last_adjustment_times = {}
    for trade in active_orders:
        order_id = trade.order.orderId if trade.order.orderId else trade.orderStatus.orderId
        if order_id:
            order_start_times[order_id] = datetime.now()
            order_last_adjustment_times[order_id] = datetime.now()
    
    last_log_time = datetime.now()
    
    while active_orders:
        time.sleep(check_interval_seconds)
        
        current_time = datetime.now()
        still_active = []
        
        for trade in active_orders:
            status = trade.orderStatus.status
            config = getattr(trade, '_monitoring_config', {})
            symbol = config.get('symbol', 'UNKNOWN') if config else 'UNKNOWN'
            order_id = trade.order.orderId if trade.order.orderId else trade.orderStatus.orderId
            
            if status == 'Filled':
                logger.info(
                    f"Order for {symbol} FILLED! "
                    f"Price: ${trade.orderStatus.avgFillPrice:.2f}, "
                    f"Quantity: {trade.orderStatus.filled}/{trade.order.totalQuantity}"
                )
                continue
            
            if status in ['Cancelled', 'ApiCancelled', 'Rejected']:
                reason = trade.orderStatus.whyHeld or 'N/A'
                logger.warning(f"Order for {symbol} {status.lower()}. Reason: {reason}")
                continue
            
            if status in ['Submitted', 'PreSubmitted', 'PendingSubmit']:
                if not config:
                    still_active.append(trade)
                    continue
                
                start_time = order_start_times.get(order_id, current_time) if order_id else current_time
                last_adjustment_time = order_last_adjustment_times.get(order_id, start_time) if order_id else start_time
                elapsed_minutes = (current_time - start_time).total_seconds() / 60.0
                minutes_since_last_adjustment = (current_time - last_adjustment_time).total_seconds() / 60.0
                
                initial_wait_minutes = config.get('initial_wait_minutes', 2)
                price_reduction_per_minute = config.get('price_reduction_per_minute', 0.01)
                order_action = config.get('order_action', 'BUY')
                min_price = config.get('min_price', 0.23)
                current_price = trade.order.lmtPrice
                
                can_modify = status == 'Submitted' or status == 'PreSubmitted'
                if can_modify and elapsed_minutes >= initial_wait_minutes and minutes_since_last_adjustment >= 1.0:
                    if order_action == "BUY":
                        new_price = current_price + price_reduction_per_minute
                        new_price = min(new_price, min_price) if min_price < 0 else max(new_price, min_price)
                    else:
                        new_price = current_price - price_reduction_per_minute
                        new_price = max(new_price, min_price)
                    
                    new_price = round(new_price, 2)
                    
                    if abs(new_price - current_price) > 0.001:
                        try:
                            with ib_lock:
                                trade.order.lmtPrice = new_price
                                ib.placeOrder(trade.contract, trade.order)
                                ib.sleep(0.5)
                            
                            actual_price = trade.order.lmtPrice
                            if abs(actual_price - new_price) > 0.01:
                                logger.warning(
                                    f"Price modification may not have been applied for {symbol}. "
                                    f"Expected: ${new_price:.2f}, Actual: ${actual_price:.2f}"
                                )
                            
                            if order_id:
                                order_last_adjustment_times[order_id] = current_time
                            
                            if new_price == min_price:
                                logger.info(f"Adjusted {symbol} order price to ${new_price:.2f} (minimum reached)")
                            else:
                                logger.info(f"Adjusted {symbol} order price to ${new_price:.2f}")
                        except Exception as e:
                            logger.error(f"Error modifying {symbol} order price: {e}", exc_info=True)
                
                still_active.append(trade)
        
        active_orders[:] = still_active
        
        if not still_active:
            logger.info("All orders completed (filled, cancelled, or rejected)")
            break
        
        if (current_time - last_log_time).total_seconds() >= 30:
            logger.info(f"Active orders: {len(still_active)}/{len(active_orders)}")
            for trade in still_active:
                config = getattr(trade, '_monitoring_config', {})
                symbol = config.get('symbol', 'UNKNOWN') if config else 'UNKNOWN'
                status = trade.orderStatus.status
                filled = trade.orderStatus.filled
                total = trade.order.totalQuantity
                logger.info(f"  {symbol}: {status}, Filled: {filled}/{total}, Price: ${trade.order.lmtPrice:.2f}")
            last_log_time = current_time

