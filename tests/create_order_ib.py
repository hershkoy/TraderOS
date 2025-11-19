#!/usr/bin/env python3
"""
create_order_ib.py

Simple test script to debug IB order creation.
Auto-detects account and port, then creates a test order.

Usage:
    # Combo order (limit price required)
    python tests/create_order_ib.py --symbol QQQ --expiry 20251126 --strike-short 560 --strike-long 556 --limit-price 0.33
    
    # Single-leg order (limit price optional, uses market if not provided)
    python tests/create_order_ib.py --symbol QQQ --expiry 20251126 --strike 560 --right P --action SELL --limit-price 1.50
"""

import argparse
import logging
import os
import sys
from typing import Optional

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ib_insync import Contract, Option, ComboLeg, Order, Trade

from utils.fetch_data import get_ib_connection, cleanup_ib_connection
try:
    from utils.ib_port_detector import detect_ib_port
except ImportError:
    from ib_port_detector import detect_ib_port
try:
    from utils.ib_account_detector import detect_ib_account
except ImportError:
    from ib_account_detector import detect_ib_account

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_simple_option_order(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    action: str,
    quantity: int,
    account: str,
    port: Optional[int] = None,
    limit_price: Optional[float] = None
) -> Trade:
    """
    Create a simple single-leg option order for testing.
    
    Args:
        symbol: Underlying symbol
        expiry: Expiration date (IB format, e.g., 20251126)
        strike: Strike price
        right: Option right ('P' or 'C')
        action: Order action ('BUY' or 'SELL')
        quantity: Number of contracts
        account: IB account ID
        port: IB API port number
        limit_price: Optional limit price (if None, uses market order)
    
    Returns:
        Trade object
    """
    ib = get_ib_connection(port=port)
    
    INDEX_SYMBOLS = {"SPX", "RUT", "NDX", "VIX", "DJX"}
    exchange = "CBOE" if symbol.upper() in INDEX_SYMBOLS else "SMART"
    
    # Build option contract
    opt = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
        exchange=exchange,
    )
    
    qualified = ib.qualifyContracts(opt)
    if not qualified:
        raise RuntimeError(f"Could not qualify option contract: {opt}")
    
    opt_q = qualified[0]
    logger.info(f"Qualified contract: {opt_q}")
    
    # Create order
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.account = account
    
    if limit_price is not None:
        order.orderType = "LMT"
        order.lmtPrice = limit_price
    else:
        order.orderType = "MKT"
    
    order.tif = "DAY"
    
    logger.info(f"Placing order: {action} {quantity} {symbol} {expiry} {strike} {right} @ {limit_price if limit_price else 'MKT'}")
    
    trade = ib.placeOrder(opt_q, order)
    
    logger.info(f"Order placed: {trade}")
    logger.info(f"Order status: {trade.orderStatus}")
    
    return trade


def create_combo_order(
    symbol: str,
    expiry: str,
    strike_short: float,
    strike_long: float,
    right: str,
    quantity: int,
    account: str,
    limit_price: float,
    port: Optional[int] = None
) -> Trade:
    """
    Create a vertical spread combo order (SELL short leg, BUY long leg).
    
    Args:
        symbol: Underlying symbol
        expiry: Expiration date (IB format, e.g., 20251126)
        strike_short: Strike price for short leg (higher strike for puts)
        strike_long: Strike price for long leg (lower strike for puts)
        right: Option right ('P' or 'C')
        quantity: Number of spreads
        account: IB account ID
        limit_price: Limit price for the spread (required)
        port: IB API port number
    
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
        strike=strike_short,
        right=right,
        exchange=exchange,  # Must match BAG exchange
    )
    long_opt = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=strike_long,
        right=right,
        exchange=exchange,  # Must match BAG exchange
    )
    
    qualified = ib.qualifyContracts(short_opt, long_opt)
    if len(qualified) != 2:
        raise RuntimeError("Could not qualify both legs with IB")
    
    short_q, long_q = qualified
    logger.info(f"Qualified short leg: {short_q}")
    logger.info(f"Qualified long leg: {long_q}")
    
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
    
    # Create order
    order = Order()
    order.action = "SELL"  # SELL the spread (credit spread)
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = round(limit_price, 2)
    order.tif = "DAY"
    order.account = account
    
    logger.info(
        f"Placing combo order: SELL {quantity}x {symbol} {expiry} "
        f"{strike_short}/{strike_long} {right} @ ${limit_price:.2f}"
    )
    
    trade = ib.placeOrder(spread, order)
    
    logger.info(f"Order placed: {trade}")
    logger.info(f"Order status: {trade.orderStatus}")
    
    # Log any error messages
    if trade.log:
        for log_entry in trade.log:
            if log_entry.message:
                logger.info(f"Order log: {log_entry.status} - {log_entry.message}")
    
    return trade


def main():
    parser = argparse.ArgumentParser(
        description="Test IB order creation with auto-detected account and port."
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Underlying symbol (e.g., QQQ, SPY)",
    )
    
    parser.add_argument(
        "--expiry",
        type=str,
        required=True,
        help="Expiration date in IB format (e.g., 20251126)",
    )
    
    parser.add_argument(
        "--right",
        type=str,
        default="P",
        choices=["P", "C"],
        help="Option right (default: P)",
    )
    
    parser.add_argument(
        "--strike-short",
        type=float,
        help="Strike price for short leg (required for combo orders)",
    )
    
    parser.add_argument(
        "--strike-long",
        type=float,
        help="Strike price for long leg (required for combo orders)",
    )
    
    parser.add_argument(
        "--strike",
        type=float,
        help="Strike price for single-leg orders",
    )
    
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Number of contracts/spreads (default: 1)",
    )
    
    parser.add_argument(
        "--limit-price",
        type=float,
        default=None,
        help="Limit price (required for combo orders, optional for single-leg - uses market if not provided)",
    )
    
    parser.add_argument(
        "--action",
        type=str,
        default="SELL",
        choices=["BUY", "SELL"],
        help="Order action for single-leg orders (default: SELL)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="IB API port number (overrides auto-detect)",
    )
    
    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="IB account ID (overrides auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Auto-detect port
    selected_port = args.port
    if selected_port:
        logger.info(f"Using IB port override: {selected_port}")
    else:
        selected_port = detect_ib_port()
        if selected_port is None:
            parser.error("Unable to auto-detect an IB port. Please specify one with --port.")
        logger.info(f"Auto-detected IB port: {selected_port}")
    os.environ["IB_PORT"] = str(selected_port)
    
    # Auto-detect account
    selected_account = args.account
    if selected_account:
        logger.info(f"Using IB account override: {selected_account}")
    else:
        selected_account = detect_ib_account(port=selected_port)
        if not selected_account:
            parser.error("Unable to auto-detect an IB account. Please specify one with --account.")
        logger.info(f"Auto-detected IB account: {selected_account}")
    
    # Determine order type and create order
    if args.strike_short and args.strike_long:
        # Combo order - limit price is required
        if args.limit_price is None:
            parser.error("--limit-price is required for combo orders")
        logger.info("Creating combo order (vertical spread)...")
        trade = create_combo_order(
            symbol=args.symbol,
            expiry=args.expiry,
            strike_short=args.strike_short,
            strike_long=args.strike_long,
            right=args.right,
            quantity=args.quantity,
            account=selected_account,
            limit_price=args.limit_price,
            port=selected_port
        )
    elif args.strike:
        # Single-leg order
        logger.info("Creating single-leg option order...")
        trade = create_simple_option_order(
            symbol=args.symbol,
            expiry=args.expiry,
            strike=args.strike,
            right=args.right,
            action=args.action,
            quantity=args.quantity,
            account=selected_account,
            port=selected_port,
            limit_price=args.limit_price
        )
    else:
        parser.error("Either --strike (for single-leg) or --strike-short and --strike-long (for combo) must be provided")
    
    # Monitor order status briefly
    logger.info("Monitoring order status for 5 seconds...")
    ib = get_ib_connection(port=selected_port)
    for i in range(5):
        ib.sleep(1)
        status = trade.orderStatus.status
        logger.info(f"Order status after {i+1}s: {status}")
        if status in ['Filled', 'Cancelled', 'Rejected', 'ApiCancelled']:
            break
    
    # Print final status
    logger.info(f"\nFinal order status: {trade.orderStatus.status}")
    if trade.orderStatus.status == 'Filled':
        logger.info(f"Fill price: ${trade.orderStatus.avgFillPrice:.2f}")
        logger.info(f"Filled: {trade.orderStatus.filled}/{trade.order.totalQuantity}")
    elif trade.orderStatus.status in ['Cancelled', 'Rejected', 'ApiCancelled']:
        if trade.log:
            for log_entry in trade.log:
                if log_entry.message:
                    logger.error(f"Order error: {log_entry.message}")
    
    # Cleanup
    cleanup_ib_connection()
    
    return 0 if trade.orderStatus.status == 'Filled' else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        cleanup_ib_connection()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        cleanup_ib_connection()
        sys.exit(1)

