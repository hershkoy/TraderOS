#!/usr/bin/env python3
"""
config_processor.py

Configuration file processing and multi-order execution.
"""

import logging
import os
import sys
import subprocess
import threading
import yaml
from typing import Dict, List, Optional, Tuple

try:
    from ib_insync import Contract, Option, ComboLeg, Order, Trade, IB
except ImportError:
    raise ImportError("ib_insync not found. Install with: pip install ib_insync")

from ..data.fetch_data import get_ib_connection
from ..api.ib.ib_port_detector import detect_ib_port
from ..api.ib.ib_account_detector import detect_ib_account
from ..options.option_csv_utils import validate_csv, load_option_rows
from ..api.ib.ib_order_utils import (
    create_ib_bracket_order,
    verify_bracket_order_structure,
    calculate_bracket_prices,
    calculate_buy_limit_price_sequence,
    monitor_all_orders,
)
from strategies.option_strategies import get_strategy, VerticalSpreadCandidate

logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    from datetime import datetime
    
    # Import fetch_options_to_csv
    try:
        from scripts.api.ib.ib_option_chain_to_csv import fetch_options_to_csv
    except ImportError:
        import importlib.util
        # File is at scripts/api/ib/ib_option_chain_to_csv.py
        project_root = os.path.dirname(os.path.dirname(__file__))
        script_path = os.path.join(project_root, "scripts", "api", "ib", "ib_option_chain_to_csv.py")
        spec = importlib.util.spec_from_file_location(
            "ib_option_chain_to_csv",
            script_path
        )
        ib_option_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ib_option_module)
        fetch_options_to_csv = ib_option_module.fetch_options_to_csv
    
    logger.info(f"Auto-fetching option chain for {symbol}...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"reports/options/spreads/{symbol}_{right}_spread_analysis_{timestamp}.csv"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
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


def choose_candidate_by_profile(
    candidates_sorted: List[VerticalSpreadCandidate],
    profile: str
) -> VerticalSpreadCandidate:
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
    target_delta = order_config.get('target_delta', 0.10)
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
        
        csv_symbol, csv_expiry = validate_csv(csv_path, expected_right=right)
        if not expiry:
            expiry = csv_expiry
        
        rows = load_option_rows(
            path=csv_path,
            right=right,
            expiry=expiry,
        )
        
        # Find spread candidates using strategy
        strategy = get_strategy(strategy_name)
        candidates = strategy.find_candidates(
            rows,
            width=spread_width,
            target_delta=target_delta,
            num_candidates=3,
            min_credit=min_credit,
        )
        
        candidates_sorted = sorted(
            candidates,
            key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
            reverse=True,
        )
        
        selected = choose_candidate_by_profile(candidates_sorted, risk_profile)
        logger.info(f"Selected spread for {symbol}: {selected.short.strike}/{selected.long.strike}")
        
        if not create_orders_en:
            logger.info(f"create_orders_en not set for {symbol}, skipping order placement")
            return None
        
        if order_action == "SELL":
            initial_price_raw = round(selected.credit * 1.15, 2)
            min_price_ib = min_price
        else:
            initial_price_raw, min_price_ib = calculate_buy_limit_price_sequence(
                initial_estimate=selected.credit,
                min_price=min_price,
                price_reduction_per_minute=price_reduction_per_minute
            )
        
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
        
        with ib_lock:
            qualified = ib.qualifyContracts(short_opt, long_opt)
            if len(qualified) != 2:
                raise RuntimeError(f"Could not qualify both legs for {symbol}")
            
            short_q, long_q = qualified
            
            short_ticker = ib.reqMktData(short_q, "", False, False)
            long_ticker = ib.reqMktData(long_q, "", False, False)
            ib.sleep(1.0)
            
            short_bid = short_ticker.bid
            short_ask = short_ticker.ask
            long_bid = long_ticker.bid
            long_ask = long_ticker.ask
            
            try:
                ib.cancelMktData(short_ticker)
                ib.cancelMktData(long_ticker)
            except Exception:
                pass
        
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
        
        if order_action == "SELL":
            ib_limit_price = -abs(initial_price)
        else:
            ib_limit_price = initial_price if initial_price < 0 else -initial_price
        
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
        
        take_profit_price = None
        stop_loss_price = None
        
        if take_profit is not None or stop_loss is not None:
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
        
        with ib_lock:
            if take_profit_price is not None or stop_loss_price is not None:
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
                    f"Placed IB order for {symbol}: {order_action} {quantity}x {selected.short.strike}/{selected.long.strike} @ {ib_limit_price:.2f}"
                )
        
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


def process_single_order_from_config(order_config: Dict, script_path: str, port: Optional[int] = None) -> Tuple[str, int, str]:
    """
    Process a single order from config by running options_strategy_trader.py as subprocess.
    
    Args:
        order_config: Order configuration dictionary
        script_path: Path to options_strategy_trader.py script
        port: IB API port number (detected in main process and passed here)
    
    Returns:
        Tuple of (symbol, return_code, output)
    """
    symbol = order_config.get('symbol', 'UNKNOWN')
    
    cmd = [
        sys.executable,
        '-u',
        script_path,
        '--symbol', symbol,
    ]
    
    if port is not None:
        cmd.extend(['--port', str(port)])
    elif 'port' in order_config:
        cmd.extend(['--port', str(order_config['port'])])
    
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
    
    try:
        result = subprocess.run(
            cmd,
            text=True,
            timeout=3600,
            stdout=None,
            stderr=subprocess.STDOUT
        )
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
    
    detected_account = None
    try:
        detected_account = detect_ib_account(port=detected_port)
        if detected_account:
            logger.info(f"Auto-detected IB account: {detected_account}")
    except Exception as e:
        logger.warning(f"Could not auto-detect IB account: {e}")
    
    ib = get_ib_connection(port=detected_port)
    ib_lock = threading.Lock()
    
    active_orders = []
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
    
    if active_orders:
        logger.info(f"Starting monitoring loop for {len(active_orders)} active orders...")
        monitor_all_orders(active_orders, ib, ib_lock, check_interval_seconds=10)
    
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

