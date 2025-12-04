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
    python scripts/trading/options_strategy_trader.py \
        --input-csv reports/SPY_P_options_20251113_221134.csv \
        --symbol SPY --expiry 20251120 --strategy otm_credit_spreads

    # Auto-fetch chain and analyze
    python scripts/trading/options_strategy_trader.py \
        --symbol SPY --expiry 20251120 --dte 7 --strategy otm_credit_spreads

    # Auto-select balanced spread and place order with monitoring
    python scripts/trading/options_strategy_trader.py \
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
import logging
import os
import sys

# Make project root importable
# Go up 2 levels from scripts/trading/ to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from ib_insync import Contract, Option, ComboLeg, Order, Trade, IB
except ImportError:
    print("Error: ib_insync not found. Install with: pip install ib_insync")
    sys.exit(1)

from utils.data.fetch_data import get_ib_connection, cleanup_ib_connection
from utils.api.ib.ib_port_detector import detect_ib_port
from utils.api.ib.ib_account_detector import detect_ib_account

# Import from refactored modules
from utils.options.option_csv_utils import validate_csv, load_option_rows
from utils.api.ib.ib_order_utils import (
    create_ib_bracket_order,
    verify_bracket_order_structure,
    calculate_bracket_prices,
    calculate_buy_limit_price_sequence,
    monitor_and_adjust_spread_order,
)
from utils.config.config_processor import (
    auto_fetch_option_chain,
    choose_candidate_by_profile,
    process_config_file,
)

# Import strategy classes
from strategies.option_strategies import (
    OptionRow, VerticalSpreadCandidate, RatioSpreadCandidate,
    get_strategy, list_strategies,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Legacy alias for backward compatibility
SpreadCandidate = VerticalSpreadCandidate


def find_spread_candidates(rows, strategy_name="otm_credit_spreads", **kwargs):
    """Find spread candidates using the specified strategy."""
    strategy = get_strategy(strategy_name)
    return strategy.find_candidates(rows, **kwargs)


def describe_candidate(label, c, strategy_name="otm_credit_spreads"):
    """Describe a candidate using the appropriate strategy."""
    strategy = get_strategy(strategy_name)
    return strategy.describe_candidate(label, c)


def print_report(candidates, strategy_name="otm_credit_spreads"):
    """Print report of spread candidates."""
    strategy = get_strategy(strategy_name)
    
    if hasattr(candidates[0], 'short') and hasattr(candidates[0].short, 'delta'):
        candidates_sorted = sorted(
            candidates,
            key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
            reverse=True,
        )
    else:
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
    logger.info("=== %s Candidates ===", strategy_name)
    logger.info("Found %d candidate(s)", len(candidates_sorted))
    
    for idx, c in enumerate(candidates_sorted):
        label = label_map[idx] if idx < len(label_map) else f"Candidate {idx+1}"
        print(f"[{idx+1}]")
        candidate_description = strategy.describe_candidate(label, c)
        print(candidate_description)
        print("-" * 50)
        
        # Log candidate details
        logger.info("Candidate [%d]: %s", idx + 1, label)
        # Log each line of the description
        for line in candidate_description.split('\n'):
            if line.strip():
                logger.info("  %s", line)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CSV option chain and trade put-credit spreads."
    )
    
    parser.add_argument("--input-csv", type=str, default=None,
        help="Path to CSV produced by ib_option_chain_to_csv.py (if omitted, will auto-fetch)")
    parser.add_argument("--symbol", type=str, default=None,
        help="Underlying symbol (e.g., SPY, QQQ, SPX)")
    parser.add_argument("--strategy", type=str, default="otm_credit_spreads", choices=list_strategies(),
        help=f"Strategy to use. Available: {', '.join(list_strategies())}")
    parser.add_argument("--expiry", type=str, default=None,
        help="Expiration in IB format (e.g., 20251120)")
    parser.add_argument("--dte", type=int, default=None,
        help="Target DTE for auto-fetch (e.g., 7)")
    parser.add_argument("--right", default="P", choices=["P", "C"],
        help="Option right to use for spreads (default: P)")
    parser.add_argument("--spread-width", type=float, default=4.0,
        help="Width of vertical spread in strike units (default: 4.0)")
    parser.add_argument("--target-delta", type=float, default=0.10,
        help="Target absolute delta for short leg (default: 0.10)")
    parser.add_argument("--min-credit", type=float, default=0.10,
        help="Minimum acceptable credit for candidate (default: 0.10)")
    parser.add_argument("--num-candidates", type=int, default=3,
        help="Number of candidate spreads to propose (default: 3)")
    parser.add_argument("--quantity", type=int, default=1,
        help="Number of spreads to trade if creating orders (default: 1)")
    parser.add_argument("--order-action", type=str, choices=["SELL", "BUY"], default="BUY",
        help="Order action: BUY or SELL (default: BUY)")
    parser.add_argument("--create-orders-en", action="store_true",
        help="If set, create IB DAY orders for the chosen spread.")
    parser.add_argument("--place-order", action="store_true",
        help="Alias for --create-orders-en")
    parser.add_argument("--risk-profile", choices=["interactive", "conservative", "balanced", "risky"], default="interactive",
        help="How to choose the spread")
    parser.add_argument("--account", type=str, default="",
        help="IB account ID for order placement")
    parser.add_argument("--live-en", action="store_true",
        help="Enable trading in live accounts (accounts starting with 'U')")
    parser.add_argument("--min-price", type=float, default=0.23,
        help="Minimum price to reduce order to (default: 0.23)")
    parser.add_argument("--initial-wait-minutes", type=int, default=2,
        help="Minutes to wait at initial price before reducing (default: 2)")
    parser.add_argument("--price-reduction-per-minute", type=float, default=0.01,
        help="Price reduction per minute after initial wait (default: 0.01)")
    parser.add_argument("--monitor-order", action="store_true",
        help="Monitor order and adjust price according to strategy")
    parser.add_argument("--no-monitor-order", action="store_true",
        help="Disable order monitoring")
    parser.add_argument("--transmit-only", action="store_true",
        help="Transmit order and exit immediately (no monitoring or auto-adjustment)")
    parser.add_argument("--std-dev", type=float, default=None,
        help="Filter strikes by standard deviation when auto-fetching")
    parser.add_argument("--max-strikes", type=int, default=100,
        help="Maximum strikes to fetch when auto-fetching (default: 100)")
    parser.add_argument("--port", type=int, default=None,
        help="IB API port number (overrides auto-detect & IB_PORT env)")
    parser.add_argument("--conf-file", type=str, default=None,
        help="Path to YAML config file with multiple orders")
    parser.add_argument("--max-workers", type=int, default=None,
        help="Maximum number of parallel workers when using --conf-file")
    parser.add_argument("--take-profit", type=float, default=None,
        help="Take profit price (e.g., -0.02 for 2 cents)")
    parser.add_argument("--stop-loss", type=float, default=None,
        help="Stop loss multiplier or absolute price")
    
    args = parser.parse_args()
    
    # Log command line parameters
    logger.info("=== Command Line Parameters ===")
    for key, value in vars(args).items():
        logger.info("  %s: %s", key, value)
    logger.info("=== End Command Line Parameters ===")
    
    # If config file is provided, process it and exit
    if args.conf_file:
        if not os.path.exists(args.conf_file):
            parser.error(f"Config file not found: {args.conf_file}")
        process_config_file(args.conf_file, max_workers=args.max_workers)
        return
    
    # Detect port
    selected_port = args.port
    if selected_port:
        logger.info("Using IB port override: %s", selected_port)
    else:
        selected_port = detect_ib_port()
        if selected_port is None:
            parser.error("Unable to auto-detect an IB port. Please specify one with --port.")
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
            max_expirations=1 if args.expiry else None,
            port=selected_port
        )
        
        csv_symbol, csv_expiry = validate_csv(csv_path, expected_right=args.right)
        if not args.symbol:
            args.symbol = csv_symbol
        if not args.expiry:
            args.expiry = csv_expiry
    else:
        csv_symbol, csv_expiry = validate_csv(csv_path, expected_right=args.right)
        
        if not args.symbol:
            args.symbol = csv_symbol
        else:
            if args.symbol.upper() != csv_symbol.upper():
                parser.error(f"Provided symbol '{args.symbol}' does not match CSV symbol '{csv_symbol}'")
        
        if not args.expiry:
            args.expiry = csv_expiry
        else:
            if args.expiry != csv_expiry:
                parser.error(f"Provided expiry '{args.expiry}' does not match CSV expiry '{csv_expiry}'")
    
    # Load options from CSV
    rows = load_option_rows(path=csv_path, right=args.right, expiry=args.expiry)
    
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
    
    # Sort candidates
    if hasattr(candidates[0], 'short') and hasattr(candidates[0].short, 'delta'):
        candidates_sorted = sorted(
            candidates,
            key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
            reverse=True,
        )
    else:
        candidates_sorted = candidates
    
    print_report(candidates_sorted, strategy_name=args.strategy)
    
    # Selection: auto vs interactive
    if args.risk_profile != "interactive":
        selected = choose_candidate_by_profile(candidates_sorted, args.risk_profile)
        selected_idx = candidates_sorted.index(selected) if selected in candidates_sorted else None
        print("\nAuto-selected spread:\n")
        selected_description = describe_candidate("Chosen spread", selected, strategy_name=args.strategy)
        print(selected_description)
        
        # Log the selected candidate
        logger.info("=== SELECTED CANDIDATE (Auto-selected via %s profile) ===", args.risk_profile)
        if selected_idx is not None:
            logger.info("Selected candidate index: [%d]", selected_idx + 1)
        for line in selected_description.split('\n'):
            if line.strip():
                logger.info("  %s", line)
        
        # Log compact summary
        if hasattr(selected, 'short') and hasattr(selected, 'long'):
            logger.info("SELECTED SUMMARY: %s %s/%s %s spread, Credit: $%.2f, Delta: %.3f",
                       args.symbol, selected.short.strike, selected.long.strike,
                       selected.short.right, selected.credit,
                       abs(selected.short.delta) if selected.short.delta is not None else 0.0)
    else:
        choice = input("\nEnter the number of the spread you want to trade (or press Enter to exit): ").strip()
        
        if not choice:
            print("No spread selected, exiting.")
            logger.info("No spread selected by user, exiting")
            return
        
        try:
            idx = int(choice) - 1
        except ValueError:
            print("Invalid choice, exiting.")
            logger.warning("Invalid choice entered: %s", choice)
            return
        
        if idx < 0 or idx >= len(candidates_sorted):
            print("Choice out of range, exiting.")
            logger.warning("Choice out of range: %d (valid range: 1-%d)", idx + 1, len(candidates_sorted))
            return
        
        selected = candidates_sorted[idx]
        print("\nYou selected:\n")
        selected_description = describe_candidate("Chosen spread", selected, strategy_name=args.strategy)
        print(selected_description)
        
        # Log the selected candidate
        logger.info("=== SELECTED CANDIDATE (Manually selected by user) ===")
        logger.info("Selected candidate index: [%d]", idx + 1)
        for line in selected_description.split('\n'):
            if line.strip():
                logger.info("  %s", line)
        
        # Log compact summary
        if hasattr(selected, 'short') and hasattr(selected, 'long'):
            logger.info("SELECTED SUMMARY: %s %s/%s %s spread, Credit: $%.2f, Delta: %.3f",
                       args.symbol, selected.short.strike, selected.long.strike,
                       selected.short.right, selected.credit,
                       abs(selected.short.delta) if selected.short.delta is not None else 0.0)
    
    # Support --place-order as alias
    if args.place_order and not args.create_orders_en:
        args.create_orders_en = True
    
    # Handle transmit-only flag (overrides other monitoring flags)
    if args.transmit_only:
        args.monitor_order = False
        args.no_monitor_order = True
        logger.info("--transmit-only flag set: order will be transmitted without monitoring or auto-adjustment")
    
    # Enable monitoring by default when placing orders (unless transmit-only or no-monitor-order is set)
    if args.create_orders_en and not args.no_monitor_order and not args.transmit_only:
        args.monitor_order = True
    elif args.no_monitor_order or args.transmit_only:
        args.monitor_order = False
    
    if not args.create_orders_en:
        print("\n(create-orders-en not set) No orders were sent. Re-run with --create-orders-en to place IB orders.")
        return
    
    # Safeguard: Prevent orders in live accounts without --live-en flag
    if args.account and args.account.startswith("U") and not args.live_en:
        error_msg = (
            f"SAFETY CHECK FAILED: Attempting to place order in live account '{args.account}' "
            f"without --live-en flag. Add --live-en to your command if you intend to trade in a live account."
        )
        logger.error(error_msg)
        print(f"\nERROR: {error_msg}")
        sys.exit(1)
    
    # Calculate initial price
    min_price_ib = None
    if args.order_action == "SELL":
        initial_price_raw = round(selected.credit * 1.15, 2)
        logger.info(f"Mid credit: ${selected.credit:.2f}, Raw initial order price (mid + 15%): ${initial_price_raw:.2f}")
    else:
        initial_price_raw, min_price_ib = calculate_buy_limit_price_sequence(
            initial_estimate=selected.credit,
            min_price=args.min_price,
            price_reduction_per_minute=args.price_reduction_per_minute
        )
        logger.info(f"Mid credit: ${selected.credit:.2f}, Starting price (estimate + $0.01): ${abs(initial_price_raw):.2f}")
    
    # Create IB order
    print("\nCreating IB DAY order via API...")
    
    exchange = "CBOE"
    logger.info(f"Using exchange: {exchange} for combo order (all components)")
    
    ib = get_ib_connection(port=selected_port)
    
    # Build & qualify legs
    short_opt = Option(
        symbol=args.symbol,
        lastTradeDateOrContractMonth=args.expiry,
        strike=selected.short.strike,
        right=selected.short.right,
        exchange=exchange,
    )
    long_opt = Option(
        symbol=args.symbol,
        lastTradeDateOrContractMonth=args.expiry,
        strike=selected.long.strike,
        right=selected.long.right,
        exchange=exchange,
    )
    
    qualified = ib.qualifyContracts(short_opt, long_opt)
    if len(qualified) != 2:
        raise RuntimeError("Could not qualify both legs with IB")
    
    short_q, long_q = qualified
    
    # Get live quotes
    logger.info("Requesting live quotes for spread legs to validate pricing...")
    short_ticker = ib.reqMktData(short_q, "", False, False)
    long_ticker = ib.reqMktData(long_q, "", False, False)
    ib.sleep(1.0)
    
    short_bid = short_ticker.bid
    short_ask = short_ticker.ask
    long_bid = long_ticker.bid
    long_ask = long_ticker.ask
    
    logger.info(f"Legs: short_bid={short_bid}, short_ask={short_ask}, long_bid={long_bid}, long_ask={long_ask}")
    
    initial_price = initial_price_raw
    
    if (short_bid is not None and short_ask is not None and
        long_bid is not None and long_ask is not None and
        short_bid > 0 and short_ask > 0 and long_bid > 0 and long_ask > 0):
        
        if args.order_action == "SELL":
            combo_bid = short_bid - long_ask
            combo_ask = short_ask - long_bid
            logger.info(f"Synthetic combo market (SELL): bid={combo_bid:.2f}, ask={combo_ask:.2f}")
            
            if initial_price < combo_bid:
                logger.info(f"Initial price ${initial_price:.2f} below bid. Raising to bid.")
                initial_price = combo_bid
            if initial_price > combo_ask:
                logger.info(f"Initial price ${initial_price:.2f} above ask. Lowering to ask.")
                initial_price = combo_ask
        else:
            combo_bid = long_ask - short_bid
            combo_ask = long_bid - short_ask
            logger.info(f"Synthetic combo market (BUY): bid={combo_bid:.2f}, ask={combo_ask:.2f}")
            
            min_allowed = combo_ask
            max_allowed = combo_bid
            if initial_price > max_allowed:
                logger.warning(f"Initial price ${initial_price:.2f} above bid ${max_allowed:.2f}.")
            elif initial_price < min_allowed:
                logger.warning(f"Initial price ${initial_price:.2f} below ask. Raising to ask.")
                initial_price = min_allowed
            else:
                logger.info(f"Initial price ${initial_price:.2f} is within NBBO range")
    else:
        logger.warning("Could not compute synthetic combo market; proceeding with raw initial price.")
    
    if args.order_action == "SELL" and initial_price < args.min_price:
        logger.info(f"Initial price ${initial_price:.2f} below min-price. Raising to min-price.")
        initial_price = round(args.min_price, 2)
    
    initial_price = round(initial_price, 2)
    
    # Convert to IB format
    if args.order_action == "SELL":
        ib_limit_price = -abs(initial_price)
        price_description = "credit"
        display_price = abs(initial_price)
    else:
        ib_limit_price = initial_price if initial_price < 0 else -initial_price
        price_description = "credit (buying combo)"
        display_price = abs(ib_limit_price)
    
    logger.info(f"Final initial {price_description}: ${display_price:.2f}, sending IB combo limit: {ib_limit_price:.2f}")
    
    # Cancel market data subscriptions
    try:
        ib.cancelMktData(short_ticker)
        ib.cancelMktData(long_ticker)
    except Exception:
        pass
    
    # Build combo legs
    short_leg = ComboLeg(conId=short_q.conId, ratio=1, action="SELL", exchange=exchange, openClose=1)
    long_leg = ComboLeg(conId=long_q.conId, ratio=1, action="BUY", exchange=exchange, openClose=1)
    
    spread_contract = Contract(symbol=args.symbol, secType="BAG", currency="USD", exchange=exchange)
    spread_contract.comboLegs = [short_leg, long_leg]
    
    # Create the order
    order = Order()
    order.action = args.order_action
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
            trade, tp_trade, sl_trade = create_ib_bracket_order(
                spread_contract=spread_contract,
                parent_order=order,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                ib=ib
            )
            
            logger.info(
                f"Placed IB bracket order: {args.order_action} {args.symbol} {args.expiry} "
                f"{args.quantity}x {selected.short.strike}/{selected.long.strike} @ {ib_limit_price:.2f}"
            )
            
            if take_profit_price is not None:
                logger.info(f"  Take profit: @ ${take_profit_price:.2f}")
            if stop_loss_price is not None:
                logger.info(f"  Stop loss: @ ${stop_loss_price:.2f}")
            
            logger.info("Verifying bracket order structure...")
            verify_bracket_order_structure(
                ib=ib,
                parent_order_id=trade.order.orderId,
                expected_tp_price=take_profit_price,
                expected_sl_price=stop_loss_price
            )
        else:
            trade = ib.placeOrder(spread_contract, order)
            logger.info(
                f"Placed IB order: {args.order_action} {args.symbol} {args.expiry} "
                f"{args.quantity}x {selected.short.strike}/{selected.long.strike} @ {ib_limit_price:.2f}"
            )
    else:
        trade = ib.placeOrder(spread_contract, order)
        logger.info(
            f"Placed IB order: {args.order_action} {args.symbol} {args.expiry} "
            f"{args.quantity}x {selected.short.strike}/{selected.long.strike} @ {ib_limit_price:.2f}"
        )
    
    if args.monitor_order:
        logger.info("Monitoring order and adjusting price according to strategy...")
        min_price_for_monitor = min_price_ib if args.order_action == "BUY" else args.min_price
        trade = monitor_and_adjust_spread_order(
            trade=trade,
            spread=spread_contract,
            initial_price=ib_limit_price,
            min_price=min_price_for_monitor,
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
