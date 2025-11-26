#!/usr/bin/env python3
"""
options_strategy_trader.py

Read an option-chain CSV (from ib_option_chain_to_csv.py), propose spread
candidates based on selected strategy, and optionally create IB combo orders.

If --input-csv is omitted, it will automatically fetch the option chain
using ib_option_chain_to_csv functionality.

Supported strategies:
- otm_credit_spreads: Sell OTM put spread for credit
- vertical_spread_with_hedging: Put ratio spread (buy 1, sell 2)

Usage examples:
    # Just analyze a CSV with default strategy
    python scripts/options_strategy_trader.py \
        --input-csv reports/SPY_P_options_20251113_221134.csv \
        --symbol SPY --expiry 20251120 --strategy otm_credit_spreads

    # Auto-fetch chain and analyze
    python scripts/options_strategy_trader.py \
        --symbol SPY --expiry 20251120 --dte 7 --strategy otm_credit_spreads

    # Auto-select balanced spread and place order with monitoring
    python scripts/options_strategy_trader.py \
        --symbol QQQ --dte 7 \
        --strategy otm_credit_spreads \
        --risk-profile balanced \
        --quantity 1 \
        --account DU123456 \
        --create-orders-en \
        --monitor-order

Assumptions:
- CSV has columns: symbol,secType,expiry,right,strike,bid,ask,volume,iv,delta,...
- All rows in the CSV are for the same underlying + right (typically PUTs).
"""

import argparse
import csv
import logging
import math
import os
import sys
import tempfile
import time
import yaml
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import Contract, Option, ComboLeg, Order, Trade, IB
except ImportError:
    print("Error: ib_insync not found. Install with: pip install ib_insync")
    sys.exit(1)

from utils.fetch_data import get_ib_connection, cleanup_ib_connection
try:
    from utils.ib_port_detector import detect_ib_port
except ImportError:  # pragma: no cover
    from ib_port_detector import detect_ib_port  # type: ignore[import]
try:
    from utils.ib_account_detector import detect_ib_account
except ImportError:  # pragma: no cover
    from ib_account_detector import detect_ib_account  # type: ignore[import]

# Import strategy classes
from strategies.option_strategies import (
    OptionRow, VerticalSpreadCandidate, RatioSpreadCandidate,
    get_strategy, list_strategies, OTMCreditSpreadsStrategy
)

# Import fetch_options_to_csv from ib_option_chain_to_csv
try:
    from scripts.ib_option_chain_to_csv import fetch_options_to_csv
except ImportError:
    # Fallback if import path is different
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ib_option_chain_to_csv",
            os.path.join(os.path.dirname(__file__), "ib_option_chain_to_csv.py")
        )
        ib_option_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ib_option_module)
        fetch_options_to_csv = ib_option_module.fetch_options_to_csv
    except Exception as e:
        print(f"Error importing fetch_options_to_csv: {e}")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Data structures ----------
# OptionRow, VerticalSpreadCandidate, RatioSpreadCandidate imported from strategies.option_strategies

# Legacy alias for backward compatibility
SpreadCandidate = VerticalSpreadCandidate

# ---------- CSV parsing / candidate selection ----------

def parse_float_safe(x: str) -> float:
    try:
        if x == "" or x.lower() == "nan":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def validate_csv(path: str, expected_right: str = "P") -> Tuple[str, str]:
    """
    Validate CSV and extract symbol and expiry.
    
    Returns:
        Tuple of (symbol, expiry)
    
    Raises:
        ValueError if validation fails
    """
    symbols = set()
    expiries = set()
    rights = set()
    
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            symbol = r.get("symbol", "").strip()
            expiry = r.get("expiry", "").strip()
            right = r.get("right", "").strip().upper()
            
            if symbol:
                symbols.add(symbol)
            if expiry:
                expiries.add(expiry)
            if right:
                rights.add(right)
    
    # Validation checks
    if not symbols:
        raise ValueError("CSV contains no symbol data")
    
    if len(symbols) > 1:
        raise ValueError(f"CSV contains multiple symbols: {symbols}. All rows must have the same symbol.")
    
    if not expiries:
        raise ValueError("CSV contains no expiry data")
    
    if len(expiries) > 1:
        raise ValueError(f"CSV contains multiple expiries: {expiries}. All rows must have the same expiry.")
    
    if not rights:
        raise ValueError("CSV contains no option right data")
    
    if len(rights) > 1:
        raise ValueError(f"CSV contains multiple option rights: {rights}. All rows must have the same right.")
    
    actual_right = rights.pop()
    if actual_right.upper() != expected_right.upper():
        raise ValueError(f"CSV contains {actual_right} options, but expected {expected_right} options.")
    
    symbol = symbols.pop()
    expiry = expiries.pop()
    
    logger.info(f"CSV validated: symbol={symbol}, expiry={expiry}, right={actual_right}")
    return symbol, expiry

def load_option_rows(
    path: str,
    right: str,
    expiry: Optional[str] = None
) -> List[OptionRow]:
    rows: List[OptionRow] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("right", "").upper() != right.upper():
                continue
            if expiry is not None and r.get("expiry") != expiry:
                continue
            row = OptionRow(
                symbol=r["symbol"],
                expiry=r["expiry"],
                right=r["right"],
                strike=float(r["strike"]),
                bid=parse_float_safe(str(r.get("bid", ""))),
                ask=parse_float_safe(str(r.get("ask", ""))),
                delta=parse_float_safe(str(r.get("delta", ""))),
                volume=parse_float_safe(str(r.get("volume", ""))),
            )
            rows.append(row)
    if not rows:
        raise ValueError("No matching option rows found in CSV")
    return rows

def build_strike_map(rows: List[OptionRow]) -> Dict[float, OptionRow]:
    return {r.strike: r for r in rows}

def find_spread_candidates(
    rows: List[OptionRow],
    strategy_name: str = "otm_credit_spreads",
    width: float = 4.0,
    target_delta: float = 0.10,
    num_candidates: int = 3,
    min_credit: float = 0.15,
    **kwargs
) -> List:
    """
    Find spread candidates using the specified strategy.
    
    Args:
        rows: List of OptionRow objects
        strategy_name: Name of strategy to use (default: otm_credit_spreads)
        width: Width of spread (for vertical spreads)
        target_delta: Target delta for short leg
        num_candidates: Number of candidates to return
        min_credit: Minimum credit required
        **kwargs: Additional strategy-specific parameters
    
    Returns:
        List of spread candidates (type depends on strategy)
    """
    strategy = get_strategy(strategy_name)
    return strategy.find_candidates(
        rows,
        width=width,
        target_delta=target_delta,
        num_candidates=num_candidates,
        min_credit=min_credit,
        **kwargs
    )

# ---------- Reporting ----------

def describe_candidate(label: str, c, strategy_name: str = "otm_credit_spreads") -> str:
    """Describe a candidate using the appropriate strategy."""
    strategy = get_strategy(strategy_name)
    return strategy.describe_candidate(label, c)

def print_report(candidates: List, strategy_name: str = "otm_credit_spreads"):
    """Print report of spread candidates."""
    strategy = get_strategy(strategy_name)
    
    # Sort by riskiness: higher |delta| = less conservative (for vertical spreads)
    if hasattr(candidates[0], 'short') and hasattr(candidates[0].short, 'delta'):
        candidates_sorted = sorted(
            candidates,
            key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
            reverse=True,
        )
    else:
        # For ratio spreads, sort by net credit
        candidates_sorted = sorted(
            candidates,
            key=lambda c: c.net_credit if hasattr(c, 'net_credit') else 0.0,
            reverse=True,
        )
    
    label_map = [
        "Primary candidate (balanced)",
        "Slightly more conservative",
        "Extra conservative",
    ]
    
    print(f"\n=== {strategy_name} Candidates ===\n")
    for idx, c in enumerate(candidates_sorted):
        label = label_map[idx] if idx < len(label_map) else f"Candidate {idx+1}"
        print(f"[{idx+1}]")
        print(strategy.describe_candidate(label, c))
        print("-" * 50)

# ---------- IB order creation ----------

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
        # Take profit order: opposite action to close the position
        tp_order = Order()
        tp_order.parentId = parent_order_id
        tp_order.action = "SELL" if parent_order.action == "BUY" else "BUY"
        tp_order.orderType = "LMT"
        tp_order.totalQuantity = parent_order.totalQuantity
        tp_order.lmtPrice = round(take_profit_price, 2)
        tp_order.tif = "GTC"  # GTC like TWS bracket orders
        tp_order.ocaGroup = oca_group  # OCA group to cancel stop loss if TP fills
        tp_order.ocaType = 3  # 3 = reduce position with overfill protection
        if parent_order.account:
            tp_order.account = parent_order.account
        
        take_profit_trade = ib.placeOrder(spread_contract, tp_order)
        logger.info(
            f"Placed take profit order: {tp_order.action} @ ${take_profit_price:.2f} "
            f"(parent order ID: {parent_order_id}, OCA: {oca_group})"
        )
    
    # Create stop loss order if specified
    if stop_loss_price is not None:
        # Stop loss order: opposite action to close the position
        sl_order = Order()
        sl_order.parentId = parent_order_id
        sl_order.action = "SELL" if parent_order.action == "BUY" else "BUY"
        sl_order.orderType = "STP"  # Stop order like TWS bracket orders
        sl_order.totalQuantity = parent_order.totalQuantity
        sl_order.auxPrice = round(stop_loss_price, 2)  # Stop trigger price goes in auxPrice
        sl_order.tif = "GTC"  # GTC like TWS bracket orders
        sl_order.ocaGroup = oca_group  # OCA group to cancel take profit if SL fills
        sl_order.ocaType = 3  # 3 = reduce position with overfill protection
        sl_order.overridePercentageConstraints = True  # Bypass TWS 80% price deviation check
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
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        ib.sleep(0.5)
        trades = ib.openTrades()
        
        # Find parent and children
        parent_found = False
        tp_found = False
        sl_found = False
        
        for trade in trades:
            order = trade.order
            
            if order.orderId == parent_order_id:
                parent_found = True
                logger.info(f"  Parent order {parent_order_id}: {order.action} @ {order.lmtPrice} ({trade.orderStatus.status})")
            
            if order.parentId == parent_order_id:
                # This is a child order
                price = order.lmtPrice if order.orderType == "LMT" else order.auxPrice
                
                if expected_tp_price is not None and abs(price - expected_tp_price) < 0.01:
                    tp_found = True
                    logger.info(f"  Take profit order {order.orderId}: {order.action} @ {price} ({order.orderType}, {trade.orderStatus.status})")
                elif expected_sl_price is not None and abs(price - expected_sl_price) < 0.01:
                    sl_found = True
                    logger.info(f"  Stop loss order {order.orderId}: {order.action} @ {price} ({order.orderType}, {trade.orderStatus.status})")
                else:
                    logger.warning(f"  Unknown child order {order.orderId}: {order.action} @ {price} ({order.orderType})")
        
        # Check if we found what we expected
        tp_ok = (expected_tp_price is None) or tp_found
        sl_ok = (expected_sl_price is None) or sl_found
        
        if parent_found and tp_ok and sl_ok:
            logger.info("Bracket order structure verified successfully")
            return True
    
    # Timeout - report what's missing
    if not parent_found:
        logger.error(f"Parent order {parent_order_id} not found in open trades")
    if expected_tp_price is not None and not tp_found:
        logger.error(f"Take profit order @ {expected_tp_price} not found")
    if expected_sl_price is not None and not sl_found:
        logger.error(f"Stop loss order @ {expected_sl_price} not found")
    
    return False


def create_ib_spread_order(
    symbol: str,
    expiry: str,
    candidate: SpreadCandidate,
    quantity: int,
    account: str = '',
    tif: str = "DAY",
    port: Optional[int] = None
) -> Trade:
    """
    Create a SELL vertical put spread order in IB for the given candidate.
    
    Args:
        symbol: Underlying symbol
        expiry: Expiration date
        candidate: Spread candidate
        quantity: Number of spreads
        account: IB account ID (empty string for default account)
        tif: Time in force (default: DAY)
        port: IB API port number (default: from IB_PORT env var or 4001)
    
    Returns:
        Trade object
    """
    ib = get_ib_connection(port=port)
    
    # Use CBOE for all combo orders (legs and BAG must use the same exchange)
    # SMART + multi-leg combo = rejected as "riskless combination"
    exchange = "CBOE"
    
    logger.info(f"Using exchange: {exchange} for combo order (all components)")
    
    # Build option contracts for short & long legs
    # Legs MUST use the same exchange as BAG (CBOE, not SMART)
    short_opt = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=candidate.short.strike,
        right=candidate.short.right,
        exchange=exchange,  # Must match BAG exchange
    )
    long_opt = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=candidate.long.strike,
        right=candidate.long.right,
        exchange=exchange,  # Must match BAG exchange
    )
    
    qualified = ib.qualifyContracts(short_opt, long_opt)
    if len(qualified) != 2:
        raise RuntimeError("Could not qualify both legs with IB")
    
    short_q, long_q = qualified
    
    # Build combo legs - MUST use the same exchange as BAG
    # openClose=1 means this is an OPENING trade (new position)
    short_leg = ComboLeg(
        conId=short_q.conId,
        ratio=1,
        action="SELL",
        exchange=exchange,  # Must match BAG exchange
        openClose=1,  # Opening trade (required to avoid "riskless combination" rejection)
    )
    long_leg = ComboLeg(
        conId=long_q.conId,
        ratio=1,
        action="BUY",
        exchange=exchange,  # Must match BAG exchange
        openClose=1,  # Opening trade (required to avoid "riskless combination" rejection)
    )
    
    spread = Contract(
        symbol=symbol,
        secType="BAG",
        currency="USD",
        exchange=exchange,  # Must match leg exchange
    )
    spread.comboLegs = [short_leg, long_leg]
    
    # Use estimated credit as limit; user can adjust in TWS if desired
    limit_price = round(candidate.credit, 2)  # positive credit internally
    
    # For a SELL credit spread, IB expects a NEGATIVE combo price
    ib_limit_price = -abs(limit_price)
    
    order = Order()
    order.action = "SELL"
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = ib_limit_price
    order.tif = tif  # DAY as requested
    
    if account:
        order.account = account
        logger.info(f"Using account: {account}")
    
    trade = ib.placeOrder(spread, order)
    
    logger.info(
        "Placed IB order: SELL %s %s %dx %s/%s @ %.2f (credit=%.2f, TIF=%s)",
        symbol,
        expiry,
        quantity,
        candidate.short.strike,
        candidate.long.strike,
        ib_limit_price,
        limit_price,
        tif,
    )
    
    return trade


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
    
    Example for BUY at -0.25:
    - Take profit at -0.02: Close the spread for only 2 cents (big profit)
    - Stop loss at -0.375 (1.5x): Close at 1.5x entry price (loss)
    """
    take_profit_price = None
    stop_loss_price = None
    
    # Take profit: use the target price directly (e.g., -0.02 to close for 2 cents)
    if take_profit_price_target is not None:
        take_profit_price = take_profit_price_target
    
    # Stop loss: multiply the limit price by the multiplier
    if stop_loss_multiplier is not None:
        stop_loss_price = limit_price * stop_loss_multiplier
    
    return take_profit_price, stop_loss_price


def calculate_buy_limit_price_sequence(
    initial_estimate: float,
    min_price: float = 0.21,
    price_reduction_per_minute: float = 0.01
) -> tuple[float, float]:
    """
    Calculate the buy limit price sequence for credit spreads.
    
    Strategy:
    - Start at initial_estimate + 0.01 (one cent more aggressive)
    - Then reduce by price_reduction_per_minute every interval
    - Minimum price is min_price (default 0.21)
    - If minimum is reached and no fill, stop attempting
    
    Args:
        initial_estimate: Initial credit estimate (positive, e.g., 0.24)
        min_price: Minimum price to attempt (default: 0.21)
        price_reduction_per_minute: Price reduction per minute (default: 0.01)
    
    Returns:
        Tuple of (starting_price, min_price) both in IB format (negative for BUY orders)
        starting_price: First price to try (initial_estimate + 0.01, converted to IB format)
        min_price: Minimum price in IB format
    
    Example:
        If initial_estimate is 0.24:
        - Starting price: -0.25 (one cent more than estimate)
        - Minimum price: -0.21
        - Sequence: -0.25, -0.24, -0.23, -0.22, -0.21
    """
    # Start one cent more aggressive than estimate
    starting_price_positive = round(initial_estimate + 0.01, 2)
    
    # Ensure minimum is respected
    if starting_price_positive < min_price:
        starting_price_positive = min_price
    
    # Convert to IB format (negative for BUY orders)
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
    
    Pricing strategy:
    - For SELL orders: Start with initial_price, then reduce by price_reduction_per_minute
    - For BUY orders: Start with initial_price (negative), then make less negative (add price_reduction_per_minute)
    
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
        # For BUY, min_price is already in IB format (negative)
        min_display = abs(min_price) if min_price < 0 else min_price
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
        # Check order status
        ib.sleep(1)  # Wait 1 second between checks
        
        status = trade.orderStatus.status
        fill_price = trade.orderStatus.avgFillPrice
        
        current_time = datetime.now()
        
        # Log status periodically (every 10 seconds)
        elapsed_seconds = int((current_time - start_time).total_seconds())
        if elapsed_seconds > 0 and elapsed_seconds % 10 == 0:
            # Format elapsed time as "Xm Ys" or just "Xs" if less than a minute
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
        
        # Check if order is filled
        if status == 'Filled':
            logger.info(
                f"Order FILLED! Price: ${fill_price:.2f}, "
                f"Quantity: {trade.orderStatus.filled}/{trade.order.totalQuantity}"
            )
            return trade
        
        # Check if order is cancelled or rejected
        if status in ['Cancelled', 'ApiCancelled', 'Rejected']:
            reason = trade.orderStatus.whyHeld or 'N/A'
            # Check for error messages in trade log
            if trade.log:
                error_messages = [log.message for log in trade.log if log.message]
                if error_messages:
                    reason = '; '.join(error_messages)
            logger.warning(f"Order {status.lower()}. Reason: {reason}")
            return trade
        
        # Apply pricing strategy
        if current_time >= initial_wait_end:
            # Time to start adjusting price
            minutes_since_reduction_start = (current_time - last_price_reduction_time).total_seconds() / 60.0
            
            if minutes_since_reduction_start >= 1.0:
                if is_buy_order:
                    # For BUY: make price less negative (closer to zero) by adding
                    new_price = current_price + price_reduction_per_minute
                    
                    # Check minimum price (most negative we're willing to go)
                    # min_price is already in IB format (negative)
                    min_ib_price = min_price
                    if new_price > min_ib_price:
                        logger.info(
                            f"Price adjustment would go above minimum (${min_ib_price:.2f}). "
                            f"Reached minimum price. Stopping attempts."
                        )
                        logger.info(
                            f"Order at minimum price ${current_price:.2f}. "
                            f"If not filled, will stop monitoring."
                        )
                        # Cancel order if we've reached minimum and it's not filled
                        ib.cancelOrder(trade.order)
                        logger.info("Order cancelled after reaching minimum price without fill.")
                        return trade
                else:
                    # For SELL: reduce price (make less positive)
                    new_price = current_price - price_reduction_per_minute
                    
                    # Check minimum price
                    if new_price < min_price:
                        logger.info(
                            f"Price reduction would go below minimum (${min_price:.2f}). "
                            f"Reached minimum price. Stopping attempts."
                        )
                        logger.info(
                            f"Order at minimum price ${current_price:.2f}. "
                            f"If not filled, will stop monitoring."
                        )
                        # Cancel order if we've reached minimum and it's not filled
                        ib.cancelOrder(trade.order)
                        logger.info("Order cancelled after reaching minimum price without fill.")
                        return trade
                
                # Update order price
                current_price = round(new_price, 2)
                trade.order.lmtPrice = current_price
                
                if is_buy_order:
                    logger.info(
                        "Adjusting limit price to $%.2f (making less negative)",
                        current_price,
                    )
                else:
                    logger.info(
                        "Reducing limit price to $%.2f",
                        current_price,
                    )
                
                # Modify order (placeOrder with same orderId modifies existing order)
                try:
                    ib.placeOrder(spread, trade.order)
                    # Wait a moment for order modification to be processed
                    ib.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error modifying order: {e}")
                    # Continue monitoring even if modification fails
                
                last_price_reduction_time = current_time
        
        # Safety timeout: cancel order after 1 hour if not filled
        if (current_time - start_time).total_seconds() > 3600:
            logger.warning("Order not filled after 1 hour. Cancelling...")
            ib.cancelOrder(trade.order)
            return trade


def choose_candidate_by_profile(
    candidates_sorted: List[SpreadCandidate],
    profile: str
) -> SpreadCandidate:
    """
    Choose a spread candidate from sorted list based on risk profile.
    
    Args:
        candidates_sorted: List of candidates sorted by |delta| (highest first = riskiest)
        profile: Risk profile ('risky', 'balanced', 'conservative')
    
    Returns:
        Selected spread candidate
    """
    n = len(candidates_sorted)
    if n == 0:
        raise ValueError("No candidates to choose from")
    
    if profile == "risky":
        idx = 0
        description = "riskiest (highest |delta|)"
    elif profile == "conservative":
        idx = n - 1
        description = "most conservative (lowest |delta|)"
    elif profile == "balanced":
        idx = 0
        description = "balanced (primary candidate, highest |delta|)"
    else:
        raise ValueError(f"Unsupported profile: {profile}")
    
    selected = candidates_sorted[idx]
    delta_abs = abs(selected.short.delta) if selected.short.delta is not None else 0.0
    
    logger.info(
        f"Risk profile '{profile}' selected candidate #{idx+1} of {n} ({description}): "
        f"{selected.short.strike:.0f}/{selected.long.strike:.0f} (|delta|={delta_abs:.3f}, credit=${selected.credit:.2f})"
    )
    
    return selected

# ---------- Auto-fetch CSV functionality ----------

def auto_fetch_option_chain(
    symbol: str,
    expiry: Optional[str] = None,
    dte: Optional[int] = None,
    right: str = "P",
    std_dev: Optional[float] = None,
    max_strikes: int = 100,
    max_expirations: Optional[int] = None,
    port: Optional[int] = None
) -> str:
    """
    Automatically fetch option chain and return path to generated CSV.
    
    Args:
        symbol: Underlying symbol
        expiry: Expiration date (optional)
        dte: Days to expiration (optional)
        right: Option right (default: P)
        std_dev: Standard deviation filter (optional)
        max_strikes: Maximum strikes to fetch (default: 100)
        max_expirations: Maximum expirations to fetch (optional)
        port: IB API port number (optional)
    
    Returns:
        Path to the generated CSV file
    """
    logger.info(f"Auto-fetching option chain for {symbol}...")
    
    # Create a temporary file for the CSV output
    import tempfile
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"reports/{symbol}_{right}_spread_analysis_{timestamp}.csv"
    
    # Ensure reports directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Call fetch_options_to_csv
    success = fetch_options_to_csv(
        underlying_symbol=symbol,
        exchange='SMART',
        currency='USD',
        right=right,
        max_strikes=max_strikes,
        max_expirations=max_expirations,
        output_csv=csv_path,
        dte_min=None,
        dte_max=None,
        target_dte=dte,
        specific_expirations=[expiry] if expiry else None,
        std_dev=std_dev,
        port=port
    )
    
    if not success:
        raise RuntimeError(f"Failed to fetch option chain for {symbol}")
    
    logger.info(f"Option chain saved to {csv_path}")
    return csv_path

# ---------- Config file processing ----------

def load_config_file(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_single_order_direct(
    order_config: Dict,
    ib: IB,
    ib_lock: threading.Lock,
    port: int,
    account: Optional[str] = None
) -> Optional[Trade]:
    """
    Process a single order directly (without subprocess) using shared IB connection.
    
    Args:
        order_config: Order configuration dictionary
        ib: Shared IB connection object
        ib_lock: Thread lock for synchronizing IB access
        port: IB API port number
        account: IB account ID (optional)
    
    Returns:
        Trade object if order was placed successfully, None otherwise
    """
    symbol = order_config.get('symbol', 'UNKNOWN')
    strategy_name = order_config.get('strategy', 'otm_credit_spreads')
    dte = order_config.get('dte', 7)
    expiry = order_config.get('expiry')
    right = order_config.get('right', 'P')
    quantity = order_config.get('quantity', 1)
    risk_profile = order_config.get('risk_profile', 'balanced')
    spread_width = order_config.get('spread_width', 4)
    target_delta = order_config.get('target_delta', 0.10)  # Default to 0.10 if not specified
    min_credit = order_config.get('min_credit', 0.15)
    min_price = order_config.get('min_price', 0.23)
    initial_wait_minutes = order_config.get('initial_wait_minutes', 2)
    price_reduction_per_minute = order_config.get('price_reduction_per_minute', 0.01)
    order_action = order_config.get('order_action', 'BUY')
    create_orders_en = order_config.get('create_orders_en', False)
    take_profit = order_config.get('take_profit')
    stop_loss = order_config.get('stop_loss') or order_config.get('stop_loss_multiplier')
    
    logger.info(f"Using strategy: {strategy_name}")
    
    try:
        # Auto-fetch option chain
        csv_path = auto_fetch_option_chain(
            symbol=symbol,
            expiry=expiry,
            dte=dte,
            right=right,
            std_dev=None,
            max_strikes=100,
            max_expirations=1,
            port=port
        )
        
        # Validate CSV
        csv_symbol, csv_expiry = validate_csv(csv_path, expected_right=right)
        if not expiry:
            expiry = csv_expiry
        
        # Load options from CSV
        rows = load_option_rows(
            path=csv_path,
            right=right,
            expiry=expiry,
        )
        
        # Find spread candidates using strategy
        candidates = find_spread_candidates(
            rows,
            strategy_name=strategy_name,
            width=spread_width,
            target_delta=target_delta,
            num_candidates=3,
            min_credit=min_credit,
        )
        
        # Sort by riskiness
        candidates_sorted = sorted(
            candidates,
            key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
            reverse=True,
        )
        
        # Select candidate
        selected = choose_candidate_by_profile(candidates_sorted, risk_profile)
        logger.info(f"Selected spread for {symbol}: {selected.short.strike}/{selected.long.strike}")
        
        if not create_orders_en:
            logger.info(f"create_orders_en not set for {symbol}, skipping order placement")
            return None
        
        # Calculate initial price
        if order_action == "SELL":
            initial_price_raw = round(selected.credit * 1.15, 2)
            min_price_ib = min_price
        else:
            initial_price_raw, min_price_ib = calculate_buy_limit_price_sequence(
                initial_estimate=selected.credit,
                min_price=min_price,
                price_reduction_per_minute=price_reduction_per_minute
            )
        
        # Build and place order
        exchange = "CBOE"
        short_opt = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=selected.short.strike,
            right=selected.short.right,
            exchange=exchange,
        )
        long_opt = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=selected.long.strike,
            right=selected.long.right,
            exchange=exchange,
        )
        
        # Synchronize IB access
        with ib_lock:
            qualified = ib.qualifyContracts(short_opt, long_opt)
            if len(qualified) != 2:
                raise RuntimeError(f"Could not qualify both legs for {symbol}")
            
            short_q, long_q = qualified
            
            # Get live quotes
            short_ticker = ib.reqMktData(short_q, "", False, False)
            long_ticker = ib.reqMktData(long_q, "", False, False)
            ib.sleep(1.0)
            
            short_bid = short_ticker.bid
            short_ask = short_ticker.ask
            long_bid = long_ticker.bid
            long_ask = long_ticker.ask
            
            # Cancel market data
            try:
                ib.cancelMktData(short_ticker)
                ib.cancelMktData(long_ticker)
            except Exception:
                pass
        
        # Calculate combo market and validate price
        initial_price = initial_price_raw
        if (
            short_bid is not None and short_ask is not None and
            long_bid is not None and long_ask is not None and
            short_bid > 0 and short_ask > 0 and
            long_bid > 0 and long_ask > 0
        ):
            if order_action == "SELL":
                combo_bid = short_bid - long_ask
                combo_ask = short_ask - long_bid
                initial_price = max(combo_bid, min(combo_ask, initial_price_raw))
            else:
                combo_bid = long_ask - short_bid
                combo_ask = long_bid - short_ask
                if initial_price_raw > combo_bid:
                    logger.warning(f"Initial price ${initial_price_raw:.2f} above bid ${combo_bid:.2f} for {symbol}")
                elif initial_price_raw < combo_ask:
                    initial_price = combo_ask
                else:
                    initial_price = initial_price_raw
                    logger.info(f"Initial price ${initial_price_raw:.2f} is within NBBO range [${combo_ask:.2f}, ${combo_bid:.2f}]")
        
        # Convert to IB format
        if order_action == "SELL":
            ib_limit_price = -abs(initial_price)
        else:
            ib_limit_price = initial_price if initial_price < 0 else -initial_price
        
        # Build order
        short_leg = ComboLeg(
            conId=short_q.conId,
            ratio=1,
            action="SELL",
            exchange=exchange,
            openClose=1,
        )
        long_leg = ComboLeg(
            conId=long_q.conId,
            ratio=1,
            action="BUY",
            exchange=exchange,
            openClose=1,
        )
        
        spread_contract = Contract(
            symbol=symbol,
            secType="BAG",
            currency="USD",
            exchange=exchange,
        )
        spread_contract.comboLegs = [short_leg, long_leg]
        
        order = Order()
        order.action = order_action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = ib_limit_price
        order.tif = "DAY"
        if account:
            order.account = account
        
        # Calculate bracket prices if specified
        take_profit_price = None
        stop_loss_price = None
        
        if take_profit is not None or stop_loss is not None:
            # Parse stop_loss: can be "double", a number (multiplier), or None
            stop_loss_multiplier = None
            if stop_loss is not None:
                if isinstance(stop_loss, str) and stop_loss.lower() == "double":
                    stop_loss_multiplier = 2.0
                else:
                    try:
                        stop_loss_multiplier = float(stop_loss)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid stop_loss value: {stop_loss}, ignoring")
            
            take_profit_price, stop_loss_price = calculate_bracket_prices(
                limit_price=ib_limit_price,
                order_action=order_action,
                take_profit_price_target=take_profit,
                stop_loss_multiplier=stop_loss_multiplier
            )
        
        # Place order (synchronized)
        with ib_lock:
            if take_profit_price is not None or stop_loss_price is not None:
                # Use bracket order
                trade, tp_trade, sl_trade = create_ib_bracket_order(
                    spread_contract=spread_contract,
                    parent_order=order,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    ib=ib
                )
                logger.info(
                    f"Placed IB bracket order for {symbol}: {order_action} {quantity}x {selected.short.strike}/{selected.long.strike} @ {ib_limit_price:.2f}"
                )
                if take_profit_price is not None:
                    logger.info(f"  Take profit: @ ${take_profit_price:.2f}")
                if stop_loss_price is not None:
                    logger.info(f"  Stop loss: @ ${stop_loss_price:.2f}")
                
                # Verify bracket order structure
                logger.info("Verifying bracket order structure...")
                verify_bracket_order_structure(
                    ib=ib,
                    parent_order_id=trade.order.orderId,
                    expected_tp_price=take_profit_price,
                    expected_sl_price=stop_loss_price
                )
            else:
                # Regular order
                trade = ib.placeOrder(spread_contract, order)
                logger.info(
                    f"Placed IB order for {symbol}: {order_action} {quantity}x {selected.short.strike}/{selected.long.strike} @ {ib_limit_price:.2f}"
                )
        
        # Store monitoring info in trade object
        trade._monitoring_config = {
            'initial_price': ib_limit_price,
            'min_price': min_price_ib if order_action == "BUY" else min_price,
            'initial_wait_minutes': initial_wait_minutes,
            'price_reduction_per_minute': price_reduction_per_minute,
            'order_action': order_action,
            'symbol': symbol,
        }
        
        return trade
        
    except Exception as e:
        logger.error(f"Error processing order for {symbol}: {e}", exc_info=True)
        return None

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
    
    # Track start time and last adjustment time for each order using order ID as key (Trade objects are not hashable)
    order_start_times = {}
    order_last_adjustment_times = {}
    for trade in active_orders:
        order_id = trade.order.orderId if trade.order.orderId else trade.orderStatus.orderId
        if order_id:
            order_start_times[order_id] = datetime.now()
            order_last_adjustment_times[order_id] = datetime.now()  # Track when we last adjusted
    
    last_log_time = datetime.now()
    
    while active_orders:
        time.sleep(check_interval_seconds)
        
        current_time = datetime.now()
        still_active = []
        
        # Check all orders in the array
        for trade in active_orders:
            status = trade.orderStatus.status
            config = getattr(trade, '_monitoring_config', {})
            symbol = config.get('symbol', 'UNKNOWN') if config else 'UNKNOWN'
            order_id = trade.order.orderId if trade.order.orderId else trade.orderStatus.orderId
            
            # Check if filled
            if status == 'Filled':
                logger.info(
                    f"Order for {symbol} FILLED! "
                    f"Price: ${trade.orderStatus.avgFillPrice:.2f}, "
                    f"Quantity: {trade.orderStatus.filled}/{trade.order.totalQuantity}"
                )
                continue
            
            # Check if cancelled/rejected
            if status in ['Cancelled', 'ApiCancelled', 'Rejected']:
                reason = trade.orderStatus.whyHeld or 'N/A'
                logger.warning(f"Order for {symbol} {status.lower()}. Reason: {reason}")
                continue
            
            # Still active - check if we need to adjust price
            # Only modify orders that are in Submitted status (PendingSubmit orders may not accept modifications)
            if status in ['Submitted', 'PreSubmitted', 'PendingSubmit']:
                if not config:
                    still_active.append(trade)
                    continue
                
                # Get start time and last adjustment time using order ID
                start_time = order_start_times.get(order_id, current_time) if order_id else current_time
                last_adjustment_time = order_last_adjustment_times.get(order_id, start_time) if order_id else start_time
                elapsed_minutes = (current_time - start_time).total_seconds() / 60.0
                minutes_since_last_adjustment = (current_time - last_adjustment_time).total_seconds() / 60.0
                
                initial_wait_minutes = config.get('initial_wait_minutes', 2)
                price_reduction_per_minute = config.get('price_reduction_per_minute', 0.01)
                order_action = config.get('order_action', 'BUY')
                min_price = config.get('min_price', 0.23)
                current_price = trade.order.lmtPrice
                
                # Only adjust price if order is in Submitted status (modifications may not work for PendingSubmit)
                # Adjust price if initial wait period has passed AND at least 1 minute since last adjustment
                can_modify = status == 'Submitted' or status == 'PreSubmitted'
                if can_modify and elapsed_minutes >= initial_wait_minutes and minutes_since_last_adjustment >= 1.0:
                    if order_action == "BUY":
                        # Make less negative (for negative prices, we add to make it closer to zero)
                        # Adjust by one step (price_reduction_per_minute) per minute
                        new_price = current_price + price_reduction_per_minute
                        # For negative prices, min_price is also negative. Use min() to cap at minimum (most negative)
                        # e.g., min(-0.20, -0.23) = -0.23 (correct - don't go below minimum)
                        new_price = min(new_price, min_price) if min_price < 0 else max(new_price, min_price)
                    else:
                        # Reduce price (for positive prices)
                        new_price = current_price - price_reduction_per_minute
                        new_price = max(new_price, min_price)  # Don't go below minimum
                    
                    # Round to 2 decimal places to match IB precision
                    new_price = round(new_price, 2)
                    
                    if abs(new_price - current_price) > 0.001:  # Use small epsilon for float comparison
                        # Modify order
                        try:
                            with ib_lock:
                                # Ensure we're modifying the order correctly
                                trade.order.lmtPrice = new_price
                                # Place the modified order (modifies trade in place)
                                ib.placeOrder(trade.contract, trade.order)
                                # Give IB a moment to process the modification
                                ib.sleep(0.5)
                            
                            # Verify the modification was applied by checking the order
                            # Note: trade.order.lmtPrice should now reflect the new price
                            actual_price = trade.order.lmtPrice
                            if abs(actual_price - new_price) > 0.01:
                                logger.warning(
                                    f"Price modification may not have been applied for {symbol}. "
                                    f"Expected: ${new_price:.2f}, Actual: ${actual_price:.2f}"
                                )
                            
                            # Update last adjustment time
                            if order_id:
                                order_last_adjustment_times[order_id] = current_time
                            
                            if new_price == min_price:
                                logger.info(f"Adjusted {symbol} order price to ${new_price:.2f} (minimum reached)")
                            else:
                                logger.info(f"Adjusted {symbol} order price to ${new_price:.2f}")
                        except Exception as e:
                            logger.error(f"Error modifying {symbol} order price: {e}", exc_info=True)
                            # Continue monitoring even if modification fails
                    elif current_price == min_price:
                        # Already at minimum, no need to adjust
                        pass
                
                still_active.append(trade)
        
        # Update active_orders list in place
        active_orders[:] = still_active
        
        if not still_active:
            logger.info("All orders completed (filled, cancelled, or rejected)")
            break
        
        # Log status periodically (every 30 seconds)
        if (current_time - last_log_time).total_seconds() >= 30:
            logger.info(f"Active orders: {len(still_active)}/{len(active_orders)}")
            for trade in still_active:
                config = getattr(trade, '_monitoring_config', {})
                symbol = config.get('symbol', 'UNKNOWN') if config else 'UNKNOWN'
                status = trade.orderStatus.status
                filled = trade.orderStatus.filled
                total = trade.order.totalQuantity  # totalQuantity is on Order, not OrderStatus
                logger.info(f"  {symbol}: {status}, Filled: {filled}/{total}, Price: ${trade.order.lmtPrice:.2f}")
            last_log_time = current_time

def process_single_order_from_config(order_config: Dict, script_path: str, port: Optional[int] = None) -> Tuple[str, int, str]:
    """
    Process a single order from config by running spreads_trader.py as subprocess.
    
    Args:
        order_config: Order configuration dictionary
        script_path: Path to spreads_trader.py script
        port: IB API port number (detected in main process and passed here)
    
    Returns:
        Tuple of (symbol, return_code, output)
    """
    symbol = order_config.get('symbol', 'UNKNOWN')
    
    # Build command line arguments
    # Use -u flag for unbuffered output so logs appear immediately
    cmd = [
        sys.executable,
        '-u',  # Unbuffered output
        script_path,
        '--symbol', symbol,
    ]
    
    # Add port (from main process detection or config, in that order)
    if port is not None:
        cmd.extend(['--port', str(port)])
    elif 'port' in order_config:
        cmd.extend(['--port', str(order_config['port'])])
    
    # Add optional parameters
    if 'dte' in order_config:
        cmd.extend(['--dte', str(order_config['dte'])])
    if 'expiry' in order_config:
        cmd.extend(['--expiry', str(order_config['expiry'])])
    if 'quantity' in order_config:
        cmd.extend(['--quantity', str(order_config['quantity'])])
    if 'risk_profile' in order_config:
        cmd.extend(['--risk-profile', str(order_config['risk_profile'])])
    if 'right' in order_config:
        cmd.extend(['--right', str(order_config['right'])])
    if 'spread_width' in order_config:
        cmd.extend(['--spread-width', str(order_config['spread_width'])])
    if 'target_delta' in order_config:
        cmd.extend(['--target-delta', str(order_config['target_delta'])])
    if 'min_credit' in order_config:
        cmd.extend(['--min-credit', str(order_config['min_credit'])])
    if 'min_price' in order_config:
        cmd.extend(['--min-price', str(order_config['min_price'])])
    if 'order_action' in order_config:
        cmd.extend(['--order-action', str(order_config['order_action'])])
    if order_config.get('create_orders_en', False):
        cmd.append('--create-orders-en')
    if order_config.get('monitor_order', True):
        cmd.append('--monitor-order')
    if order_config.get('no_monitor_order', False):
        cmd.append('--no-monitor-order')
    if 'account' in order_config:
        cmd.extend(['--account', str(order_config['account'])])
    if order_config.get('live_en', False):
        cmd.append('--live-en')
    
    # Run subprocess with output going directly to stdout/stderr
    # This ensures all output is captured by the batch file's log redirection
    try:
        result = subprocess.run(
            cmd,
            text=True,
            timeout=3600,  # 1 hour timeout per order
            # Don't capture output - let it go to stdout/stderr so batch file captures it
            # But we still need return code, so we'll capture and also print
            stdout=None,  # Let it go to parent's stdout
            stderr=subprocess.STDOUT  # Merge stderr to stdout
        )
        # Capture output for error reporting, but it's already been printed
        return (symbol, result.returncode, "")
    except subprocess.TimeoutExpired:
        error_msg = f"Timeout after 1 hour for {symbol}"
        print(error_msg, flush=True)
        return (symbol, -1, error_msg)
    except Exception as e:
        error_msg = f"Error running subprocess: {str(e)}"
        print(error_msg, flush=True)
        return (symbol, -1, error_msg)

def process_config_file(config_path: str, max_workers: Optional[int] = None):
    """
    Process multiple orders from YAML config file sequentially, then monitor all orders together.
    
    Args:
        config_path: Path to YAML config file
        max_workers: Not used (kept for compatibility)
    """
    config = load_config_file(config_path)
    orders = config.get('orders', [])
    
    if not orders:
        logger.error("No orders found in config file")
        return
    
    logger.info(f"Processing {len(orders)} orders from config file: {config_path}")
    
    # Detect IB port
    detected_port = None
    if 'port' in config:
        detected_port = config['port']
        logger.info(f"Using port from config: {detected_port}")
    else:
        try:
            detected_port = detect_ib_port()
            if detected_port is None:
                logger.error("Unable to auto-detect an IB port. Please specify one in config file or with --port.")
                return
            logger.info(f"Auto-detected IB port: {detected_port}")
        except Exception as e:
            logger.error(f"Error detecting IB port: {e}")
            return
    
    # Detect account
    detected_account = None
    try:
        detected_account = detect_ib_account(port=detected_port)
        if detected_account:
            logger.info(f"Auto-detected IB account: {detected_account}")
    except Exception as e:
        logger.warning(f"Could not auto-detect IB account: {e}")
    
    # Create single IB connection
    ib = get_ib_connection(port=detected_port)
    ib_lock = threading.Lock()  # Shared lock for IB operations
    
    # Process orders sequentially (strategy qualification and placing initial orders)
    active_orders = []  # List of Trade objects
    errors = []
    
    for order in orders:
        symbol = order.get('symbol', 'UNKNOWN')
        print(f"\n{'='*60}", flush=True)
        print(f"Processing order for {symbol}", flush=True)
        print(f"{'='*60}", flush=True)
        
        try:
            trade = process_single_order_direct(
                order_config=order,
                ib=ib,
                ib_lock=ib_lock,
                port=detected_port,
                account=detected_account or order.get('account')
            )
            
            if trade:
                active_orders.append(trade)
                print(f"Order for {symbol} placed successfully", flush=True)
            else:
                errors.append((symbol, "Order not placed (create_orders_en may be False)"))
                print(f"Order for {symbol} not placed", flush=True)
        except Exception as e:
            errors.append((symbol, str(e)))
            logger.error(f"Error processing order for {symbol}: {e}", exc_info=True)
            print(f"Order for {symbol} failed: {e}", flush=True)
        finally:
            print(f"{'='*60}\n", flush=True)
    
    # Monitor all active orders in a single loop
    if active_orders:
        logger.info(f"Starting monitoring loop for {len(active_orders)} active orders...")
        monitor_all_orders(active_orders, ib, ib_lock, check_interval_seconds=10)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total orders: {len(orders)}")
    print(f"Placed: {len(active_orders)}")
    print(f"Failed: {len(errors)}")
    print("=" * 60)
    
    if errors:
        print("\nFailed orders:")
        for symbol, error_msg in errors:
            print(f"  - {symbol}: {error_msg[:200]}")

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze CSV option chain and trade put-credit spreads."
    )
    
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to CSV produced by ib_option_chain_to_csv.py (if omitted, will auto-fetch)",
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Underlying symbol (e.g., SPY, QQQ, SPX). If omitted and --input-csv provided, will be extracted from CSV.",
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="otm_credit_spreads",
        choices=list_strategies(),
        help=f"Strategy to use for spread selection. Available: {', '.join(list_strategies())} (default: otm_credit_spreads)",
    )
    
    parser.add_argument(
        "--expiry",
        type=str,
        default=None,
        help="Expiration in IB format (e.g., 20251120). If omitted and --input-csv provided, will be extracted from CSV.",
    )
    
    parser.add_argument(
        "--dte",
        type=int,
        default=None,
        help="Target DTE for auto-fetch (e.g., 7). Used when --input-csv is omitted.",
    )
    
    parser.add_argument(
        "--right",
        default="P",
        choices=["P", "C"],
        help="Option right to use for spreads (default: P)",
    )
    
    parser.add_argument(
        "--spread-width",
        type=float,
        default=4.0,
        help="Width of vertical spread in strike units (default: 4.0)",
    )
    
    parser.add_argument(
        "--target-delta",
        type=float,
        default=0.10,
        help="Target absolute delta for short leg (default: 0.10)",
    )
    
    parser.add_argument(
        "--min-credit",
        type=float,
        default=0.10,
        help="Minimum acceptable credit for candidate (default: 0.10)",
    )
    
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of candidate spreads to propose (default: 3)",
    )
    
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Number of spreads to trade if creating orders (default: 1)",
    )
    
    parser.add_argument(
        "--order-action",
        type=str,
        choices=["SELL", "BUY"],
        default="BUY",
        help="Order action: BUY to open debit spread or close credit spread (default), SELL to open credit spread",
    )
    
    parser.add_argument(
        "--create-orders-en",
        action="store_true",
        help="If set, create IB DAY orders for the chosen spread.",
    )
    
    parser.add_argument(
        "--place-order",
        action="store_true",
        help="Alias for --create-orders-en (for convenience).",
    )
    
    parser.add_argument(
        "--risk-profile",
        choices=["interactive", "conservative", "balanced", "risky"],
        default="interactive",
        help=(
            "How to choose the spread. "
            "'interactive' = ask user; "
            "'conservative' = lowest |delta|; "
            "'balanced' = middle |delta|; "
            "'risky' = highest |delta|."
        ),
    )
    
    parser.add_argument(
        "--account",
        type=str,
        default="",
        help="IB account ID for order placement (e.g., DU123456 for paper trading, U123456 for live). Empty string auto-detects from managed accounts.",
    )
    
    parser.add_argument(
        "--live-en",
        action="store_true",
        help="Enable trading in live accounts (accounts starting with 'U'). Required when placing orders in real accounts.",
    )
    
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.23,
        help="Minimum price to reduce order to (default: 0.23)",
    )
    
    parser.add_argument(
        "--initial-wait-minutes",
        type=int,
        default=2,
        help="Minutes to wait at initial price before reducing (default: 2)",
    )
    
    parser.add_argument(
        "--price-reduction-per-minute",
        type=float,
        default=0.01,
        help="Price reduction per minute after initial wait (default: 0.01)",
    )
    
    parser.add_argument(
        "--monitor-order",
        action="store_true",
        help="Monitor order and adjust price according to strategy (mid+15% initial, then reduce by --price-reduction-per-minute/min, min --min-price). Enabled by default when --create-orders-en is set.",
    )
    
    parser.add_argument(
        "--no-monitor-order",
        action="store_true",
        help="Disable order monitoring (just place order and exit).",
    )
    
    parser.add_argument(
        "--std-dev",
        type=float,
        default=None,
        help="Filter strikes by standard deviation when auto-fetching (e.g., 2.0 for '2 SD')",
    )
    
    parser.add_argument(
        "--max-strikes",
        type=int,
        default=100,
        help="Maximum strikes to fetch when auto-fetching (default: 100)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="IB API port number (overrides auto-detect & IB_PORT env).",
    )
    
    parser.add_argument(
        "--conf-file",
        type=str,
        default=None,
        help="Path to YAML config file with multiple orders. When provided, processes all orders in parallel and ignores other arguments.",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers when using --conf-file (default: number of orders).",
    )
    
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit in dollars (credit). If specified, creates a bracket order with take profit. Example: 0.02 for 2 cents.",
    )
    
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss multiplier or absolute price. If 'double' or 2.0, uses double the limit price. If a number, uses that as multiplier. If absolute price specified, uses that directly.",
    )
    
    args = parser.parse_args()
    
    # If config file is provided, process it and exit
    if args.conf_file:
        if not os.path.exists(args.conf_file):
            parser.error(f"Config file not found: {args.conf_file}")
        process_config_file(args.conf_file, max_workers=args.max_workers)
        return
    
    selected_port = args.port
    if selected_port:
        logger.info("Using IB port override: %s", selected_port)
    else:
        selected_port = detect_ib_port()
        if selected_port is None:
            parser.error(
                "Unable to auto-detect an IB port. Please specify one with --port."
            )
        logger.info("Auto-detected IB port: %s", selected_port)
    os.environ["IB_PORT"] = str(selected_port)
    
    # Auto-detect account if not provided
    if not args.account:
        detected_account = detect_ib_account(port=selected_port)
        if detected_account:
            args.account = detected_account
            logger.info("Auto-detected IB account: %s", args.account)
        else:
            logger.warning("Could not auto-detect IB account. Order will use default account.")

    # Determine CSV path - either use provided or auto-fetch
    csv_path = args.input_csv
    
    if csv_path is None:
        # Auto-fetch option chain
        if not args.symbol:
            parser.error("--symbol is required when --input-csv is omitted")
        if not args.expiry and not args.dte:
            parser.error("Either --expiry or --dte must be provided when --input-csv is omitted")
        
        csv_path = auto_fetch_option_chain(
            symbol=args.symbol,
            expiry=args.expiry,
            dte=args.dte,
            right=args.right,
            std_dev=args.std_dev,
            max_strikes=args.max_strikes,
            max_expirations=1 if args.expiry else None,  # If expiry specified, only fetch that one
            port=selected_port
        )
        
        # Extract symbol and expiry from CSV if not provided
        csv_symbol, csv_expiry = validate_csv(csv_path, expected_right=args.right)
        if not args.symbol:
            args.symbol = csv_symbol
            logger.info(f"Using symbol from CSV: {args.symbol}")
        if not args.expiry:
            args.expiry = csv_expiry
            logger.info(f"Using expiry from CSV: {args.expiry}")
    else:
        # CSV provided - validate and extract symbol/expiry if not provided
        csv_symbol, csv_expiry = validate_csv(csv_path, expected_right=args.right)
        
        if not args.symbol:
            args.symbol = csv_symbol
            logger.info(f"Using symbol from CSV: {args.symbol}")
        else:
            # Verify provided symbol matches CSV
            if args.symbol.upper() != csv_symbol.upper():
                parser.error(f"Provided symbol '{args.symbol}' does not match CSV symbol '{csv_symbol}'")
        
        if not args.expiry:
            args.expiry = csv_expiry
            logger.info(f"Using expiry from CSV: {args.expiry}")
        else:
            # Verify provided expiry matches CSV
            if args.expiry != csv_expiry:
                parser.error(f"Provided expiry '{args.expiry}' does not match CSV expiry '{csv_expiry}'")
    
    # Load options from CSV
    rows = load_option_rows(
        path=csv_path,
        right=args.right,
        expiry=args.expiry,
    )
    
    # Propose spread candidates using strategy
    logger.info(f"Using strategy: {args.strategy}")
    candidates = find_spread_candidates(
        rows,
        strategy_name=args.strategy,
        width=args.spread_width,
        target_delta=args.target_delta,
        num_candidates=args.num_candidates,
        min_credit=args.min_credit,
    )
    
    # Sort candidates by riskiness: higher |delta| = less conservative
    if hasattr(candidates[0], 'short') and hasattr(candidates[0].short, 'delta'):
        candidates_sorted = sorted(
            candidates,
            key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
            reverse=True,
        )
    else:
        candidates_sorted = candidates  # For other strategies
    
    print_report(candidates_sorted, strategy_name=args.strategy)
    
    # Selection: auto vs interactive
    if args.risk_profile != "interactive":
        selected = choose_candidate_by_profile(candidates_sorted, args.risk_profile)
        print("\nAuto-selected spread:\n")
        print(describe_candidate("Chosen spread", selected, strategy_name=args.strategy))
    else:
        # Interactive selection
        choice = input(
            "\nEnter the number of the spread you want to trade (or press Enter to exit): "
        ).strip()
        
        if not choice:
            print("No spread selected, exiting.")
            return
        
        try:
            idx = int(choice) - 1
        except ValueError:
            print("Invalid choice, exiting.")
            return
        
        if idx < 0 or idx >= len(candidates_sorted):
            print("Choice out of range, exiting.")
            return
        
        selected = candidates_sorted[idx]
        
        print("\nYou selected:\n")
        print(describe_candidate("Chosen spread", selected, strategy_name=args.strategy))
    
    # Support --place-order as alias for --create-orders-en
    if args.place_order and not args.create_orders_en:
        args.create_orders_en = True
    
    # Enable monitoring by default when placing orders (unless explicitly disabled)
    if args.create_orders_en and not args.no_monitor_order:
        args.monitor_order = True
    elif args.no_monitor_order:
        args.monitor_order = False
    
    if not args.create_orders_en:
        print(
            "\n(create-orders-en not set) No orders were sent. "
            "Re-run with --create-orders-en to place IB orders."
        )
        return
    
    # Safeguard: Prevent orders in live accounts without --live-en flag
    if args.account and args.account.startswith("U") and not args.live_en:
        error_msg = (
            f"SAFETY CHECK FAILED: Attempting to place order in live account '{args.account}' "
            f"without --live-en flag. This is blocked to prevent accidental live trading.\n"
            f"Add --live-en to your command if you intend to trade in a live account."
        )
        logger.error(error_msg)
        print(f"\nERROR: {error_msg}")
        sys.exit(1)
    
    # Calculate initial price using new strategy
    # For SELL: credit received = short.mid - long.mid (positive, add 15% markup)
    # For BUY: use new strategy - start at estimate + 1 cent, then reduce
    min_price_ib = None  # Will be set for BUY orders
    if args.order_action == "SELL":
        initial_price_raw = round(selected.credit * 1.15, 2)  # Positive credit
        logger.info(
            f"Mid credit: ${selected.credit:.2f}, "
            f"Raw initial order price (mid + 15%): ${initial_price_raw:.2f}"
        )
    else:
        # For BUY: use new price calculation strategy
        # Start at estimate + 1 cent, then reduce by 1 cent per interval
        # Minimum is args.min_price (default 0.21)
        initial_price_raw, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=selected.credit,
            min_price=args.min_price,
            price_reduction_per_minute=args.price_reduction_per_minute
        )
        logger.info(
            f"Mid credit: ${selected.credit:.2f}, "
            f"Starting price (estimate + $0.01): ${abs(initial_price_raw):.2f} (IB: ${initial_price_raw:.2f})"
        )
    
    # Create IB order
    print("\nCreating IB DAY order via API...")
    
    # Use CBOE for all combo orders (legs and BAG must use the same exchange)
    # SMART + multi-leg combo = rejected as "riskless combination"
    exchange = "CBOE"
    
    logger.info(f"Using exchange: {exchange} for combo order (all components)")
    
    ib = get_ib_connection(port=selected_port)
    
    # --- Build & qualify legs ---
    # Legs MUST use the same exchange as BAG (CBOE, not SMART)
    short_opt = Option(
        symbol=args.symbol,
        lastTradeDateOrContractMonth=args.expiry,
        strike=selected.short.strike,
        right=selected.short.right,
        exchange=exchange,  # Must match BAG exchange
    )
    long_opt = Option(
        symbol=args.symbol,
        lastTradeDateOrContractMonth=args.expiry,
        strike=selected.long.strike,
        right=selected.long.right,
        exchange=exchange,  # Must match BAG exchange
    )
    
    qualified = ib.qualifyContracts(short_opt, long_opt)
    if len(qualified) != 2:
        raise RuntimeError("Could not qualify both legs with IB")
    
    short_q, long_q = qualified
    
    # Build combo legs - MUST use the same exchange as BAG
    # openClose=1 means this is an OPENING trade (new position)
    short_leg = ComboLeg(
        conId=short_q.conId,
        ratio=1,
        action="SELL",
        exchange=exchange,  # Must match BAG exchange
        openClose=1,  # Opening trade (required to avoid "riskless combination" rejection)
    )
    long_leg = ComboLeg(
        conId=long_q.conId,
        ratio=1,
        action="BUY",
        exchange=exchange,  # Must match BAG exchange
        openClose=1,  # Opening trade (required to avoid "riskless combination" rejection)
    )
    
    spread_contract = Contract(
        symbol=args.symbol,
        secType="BAG",
        currency="USD",
        exchange=exchange,  # Must match leg exchange
    )
    spread_contract.comboLegs = [short_leg, long_leg]
    
    # --- Get live quotes to avoid "riskless combination" rejection ---
    logger.info("Requesting live quotes for spread legs to validate pricing...")
    short_ticker = ib.reqMktData(short_q, "", False, False)
    long_ticker = ib.reqMktData(long_q, "", False, False)
    ib.sleep(1.0)  # give IB a moment to populate quotes
    
    short_bid = short_ticker.bid
    short_ask = short_ticker.ask
    long_bid = long_ticker.bid
    long_ask = long_ticker.ask
    
    logger.info(
        f"Legs: short_bid={short_bid}, short_ask={short_ask}, "
        f"long_bid={long_bid}, long_ask={long_ask}"
    )
    
    initial_price = initial_price_raw
    
    combo_bid = None
    combo_ask = None
    
    if (
        short_bid is not None and short_ask is not None and
        long_bid is not None and long_ask is not None and
        short_bid > 0 and short_ask > 0 and
        long_bid > 0 and long_ask > 0
    ):
        # Calculate combo market based on order action:
        # For SELL (opening credit spread): short_bid - long_ask (positive credit received)
        # For BUY (opening credit spread): long_ask - short_bid (negative, paying to buy)
        if args.order_action == "SELL":
            # SELL: we receive credit = short_bid - long_ask (positive)
            combo_bid = short_bid - long_ask
            combo_ask = short_ask - long_bid
            logger.info(
                f"Synthetic combo market (SELL): bid={combo_bid:.2f}, ask={combo_ask:.2f}, "
                f"mid={(combo_bid + combo_ask) / 2:.2f}"
            )
        else:
            # BUY: we pay debit = long_ask - short_bid (negative for credit spread)
            combo_bid = long_ask - short_bid  # Negative (buying at ask, selling at bid)
            combo_ask = long_bid - short_ask  # More negative
            logger.info(
                f"Synthetic combo market (BUY): bid={combo_bid:.2f}, ask={combo_ask:.2f}, "
                f"mid={(combo_bid + combo_ask) / 2:.2f}"
            )
        
        # Clamp initial price into [bid, ask] to satisfy NBBO rules
        # For SELL: must be between bid and ask (both positive)
        # For BUY: must be between bid and ask (both negative, bid > ask)
        if args.order_action == "SELL":
            min_allowed = combo_bid
            max_allowed = combo_ask
            if initial_price < min_allowed:
                logger.info(
                    f"Initial price ${initial_price:.2f} is below synthetic bid "
                    f"${min_allowed:.2f}. Raising to bid."
                )
                initial_price = min_allowed
            if initial_price > max_allowed:
                logger.info(
                    f"Initial price ${initial_price:.2f} is above synthetic ask "
                    f"${max_allowed:.2f}. Lowering to ask."
                )
                initial_price = max_allowed
        else:
            # For BUY: bid > ask (both negative), so min_allowed is ask, max_allowed is bid
            # With new strategy, we want to use calculated price (estimate + 1 cent)
            # Only warn if outside NBBO, but try the calculated price anyway
            min_allowed = combo_ask  # More negative
            max_allowed = combo_bid  # Less negative
            if initial_price > max_allowed:
                logger.warning(
                    f"Initial price ${initial_price:.2f} is above synthetic bid "
                    f"${max_allowed:.2f}. Using calculated price anyway (may be rejected by IB)."
                )
                # Don't lower it - use the calculated price
            elif initial_price < min_allowed:
                logger.warning(
                    f"Initial price ${initial_price:.2f} is below synthetic ask "
                    f"${min_allowed:.2f}. Raising to ask to satisfy NBBO."
                )
                initial_price = min_allowed
            else:
                logger.info(
                    f"Initial price ${initial_price:.2f} is within NBBO range "
                    f"[${min_allowed:.2f}, ${max_allowed:.2f}]"
                )
    else:
        logger.warning(
            "Could not compute synthetic combo market from leg quotes; "
            "proceeding with raw initial price (may be rejected)."
        )
    
    # Respect your script's minimum credit floor (only for SELL orders)
    if args.order_action == "SELL" and initial_price < args.min_price:
        logger.info(
            f"Initial price ${initial_price:.2f} is below min-price ${args.min_price:.2f}. "
            f"Raising up to min-price."
        )
        initial_price = round(args.min_price, 2)
    
    initial_price = round(initial_price, 2)
    
    # Convert to IB combo price based on order action:
    # SELL credit spread → negative price (IB convention)
    # BUY credit spread → negative price (IB convention, we're buying the combo)
    if args.order_action == "SELL":
        # For SELL: IB expects negative price for credit spread
        ib_limit_price = -abs(initial_price)  # Negative
        price_description = "credit"
        display_price = abs(initial_price)  # Positive for logs
    else:
        # For BUY: IB expects negative price (buying credit spread = paying negative amount)
        ib_limit_price = initial_price  # Already negative or will be made negative
        if ib_limit_price > 0:
            # If somehow positive, make it negative
            ib_limit_price = -ib_limit_price
        price_description = "credit (buying combo)"
        display_price = abs(ib_limit_price)  # Positive for logs
    
    logger.info(
        "Final initial %s: $%.2f, sending IB combo limit: %.2f",
        price_description,
        display_price,
        ib_limit_price,
    )
    
    # Cancel market data subscriptions before placing order
    # (Optional - harmless if skipped, but cleaner to cancel)
    try:
        ib.cancelMktData(short_ticker)
        ib.cancelMktData(long_ticker)
    except Exception:
        pass  # Ignore cancel errors - harmless
    
    # --- Create the order ---
    order = Order()
    order.action = args.order_action  # Use user-specified action (SELL or BUY)
    order.orderType = "LMT"
    order.totalQuantity = args.quantity
    order.lmtPrice = ib_limit_price
    order.tif = "DAY"
    
    if args.account:
        order.account = args.account
        logger.info("Using account: %s", args.account)
    
    # Calculate bracket prices if specified
    take_profit_price = None
    stop_loss_price = None
    
    if args.take_profit is not None or args.stop_loss is not None:
        # Parse stop_loss: can be "double", a number (multiplier), or None
        stop_loss_multiplier = None
        if args.stop_loss is not None:
            if isinstance(args.stop_loss, str) and args.stop_loss.lower() == "double":
                stop_loss_multiplier = 2.0
            else:
                try:
                    stop_loss_multiplier = float(args.stop_loss)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid stop_loss value: {args.stop_loss}, ignoring")
        
        take_profit_price, stop_loss_price = calculate_bracket_prices(
            limit_price=ib_limit_price,
            order_action=args.order_action,
            take_profit_price_target=args.take_profit,
            stop_loss_multiplier=stop_loss_multiplier
        )
        
        if take_profit_price is not None or stop_loss_price is not None:
            # Use bracket order
            trade, tp_trade, sl_trade = create_ib_bracket_order(
                spread_contract=spread_contract,
                parent_order=order,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                ib=ib
            )
            
            logger.info(
                "Placed IB bracket order: %s %s %s %dx %s/%s @ %.2f (%s=%.2f, TIF=DAY)",
                args.order_action,
                args.symbol,
                args.expiry,
                args.quantity,
                selected.short.strike,
                selected.long.strike,
                ib_limit_price,
                price_description,
                display_price,
            )
            
            if take_profit_price is not None:
                logger.info(f"  Take profit: @ ${take_profit_price:.2f}")
            if stop_loss_price is not None:
                logger.info(f"  Stop loss: @ ${stop_loss_price:.2f}")
            
            # Verify bracket order structure
            logger.info("Verifying bracket order structure...")
            verify_bracket_order_structure(
                ib=ib,
                parent_order_id=trade.order.orderId,
                expected_tp_price=take_profit_price,
                expected_sl_price=stop_loss_price
            )
        else:
            # No bracket orders, just place regular order
            trade = ib.placeOrder(spread_contract, order)
            logger.info(
                "Placed IB order: %s %s %s %dx %s/%s @ %.2f (%s=%.2f, TIF=DAY)",
                args.order_action,
                args.symbol,
                args.expiry,
                args.quantity,
                selected.short.strike,
                selected.long.strike,
                ib_limit_price,
                price_description,
                display_price,
            )
    else:
        # No bracket orders, just place regular order
        trade = ib.placeOrder(spread_contract, order)
        logger.info(
            "Placed IB order: %s %s %s %dx %s/%s @ %.2f (%s=%.2f, TIF=DAY)",
            args.order_action,
            args.symbol,
            args.expiry,
            args.quantity,
            selected.short.strike,
            selected.long.strike,
            ib_limit_price,
            price_description,
            display_price,
        )
    
    if args.monitor_order:
        logger.info("Monitoring order and adjusting price according to strategy...")
        # For BUY orders, pass min_price in IB format (negative)
        if args.order_action == "BUY":
            min_price_for_monitor = min_price_ib if args.order_action == "BUY" else args.min_price
        else:
            min_price_for_monitor = args.min_price
        trade = monitor_and_adjust_spread_order(
            trade=trade,
            spread=spread_contract,
            initial_price=ib_limit_price,  # IB price format (negative for both SELL and BUY credit spreads)
            min_price=min_price_for_monitor,  # For BUY: already in IB format, for SELL: positive
            initial_wait_minutes=args.initial_wait_minutes,
            price_reduction_per_minute=args.price_reduction_per_minute,
            port=selected_port
        )
        
        if trade.orderStatus.status == 'Filled':
            logger.info("Order successfully filled!")
        else:
            logger.warning(f"Order status: {trade.orderStatus.status}")
    else:
        print("Order sent to IB (check TWS).")
        print("(Use --monitor-order to automatically adjust price and monitor until filled)")
    
    # Cleanup connection
    cleanup_ib_connection()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cleanup_ib_connection()

