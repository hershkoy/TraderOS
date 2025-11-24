#!/usr/bin/env python3
"""
spreads_trader.py

Read an option-chain CSV (from ib_option_chain_to_csv.py), propose a few
put-credit-spread candidates around a target delta, and optionally create
IB combo orders for the chosen spread.

If --input-csv is omitted, it will automatically fetch the option chain
using ib_option_chain_to_csv functionality.

Usage examples:
    # Just analyze a CSV, no orders
    python scripts/spreads_trader.py \
        --input-csv reports/SPY_P_options_20251113_221134.csv \
        --symbol SPY --expiry 20251120

    # Auto-fetch chain and analyze (no CSV file needed)
    python scripts/spreads_trader.py \
        --symbol SPY --expiry 20251120 --dte 7

    # Analyze and create a DAY order for 2 spreads
    python scripts/spreads_trader.py \
        --input-csv reports/SPY_P_options_20251113_221134.csv \
        --symbol SPY --expiry 20251120 \
        --quantity 2 \
        --create-orders-en

    # Auto-select balanced spread and place order with monitoring
    python scripts/spreads_trader.py \
        --input-csv reports/QQQ_P_options.csv \
        --symbol QQQ --expiry 20251120 \
        --risk-profile balanced \
        --quantity 1 \
        --account DU123456 \
        --create-orders-en \
        --monitor-order

    # Auto-fetch, auto-select, and place order
    python scripts/spreads_trader.py \
        --symbol QQQ --dte 7 \
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

@dataclass
class OptionRow:
    symbol: str
    expiry: str
    right: str
    strike: float
    bid: float
    ask: float
    delta: Optional[float]
    volume: Optional[float]

    @property
    def mid(self) -> Optional[float]:
        if math.isnan(self.bid) or math.isnan(self.ask):
            return None
        if self.bid == "" or self.ask == "":
            return None
        try:
            return (float(self.bid) + float(self.ask)) / 2.0
        except Exception:
            return None

@dataclass
class SpreadCandidate:
    short: OptionRow
    long: OptionRow
    width: float
    credit: float  # estimated mid credit

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
    width: float = 4.0,
    target_delta: float = 0.10,
    num_candidates: int = 3,
    min_credit: float = 0.15
) -> List[SpreadCandidate]:
    """
    Find num_candidates vertical put spreads with short leg around target_delta.
    Short put delta is negative -> we use abs(delta).
    Long put is width points lower in strike.
    """
    # Ensure target_delta has a valid value
    if target_delta is None:
        target_delta = 0.10
    
    strike_map = build_strike_map(rows)
    
    # Filter rows with usable greeks + quotes
    usable = []
    for r in rows:
        if r.delta is None or math.isnan(r.delta):
            continue
        if r.mid is None:
            continue
        d_abs = abs(r.delta)
        if d_abs < 0.02 or d_abs > 0.35:  # arbitrary sanity range
            continue
        usable.append(r)
    
    if not usable:
        # Provide diagnostic info
        total_with_delta = sum(1 for r in rows if r.delta is not None and not math.isnan(r.delta))
        total_with_mid = sum(1 for r in rows if r.mid is not None)
        raise ValueError(
            f"No usable options with delta and quotes. "
            f"Found {total_with_delta} options with delta, {total_with_mid} with mid prices, "
            f"but none in delta range [0.02, 0.35]"
        )
    
    logger.info(f"Found {len(usable)} usable options with delta and quotes")
    
    # Sort by closeness to target delta (absolute)
    # Defensive check: filter out any rows with None delta that might have slipped through
    usable = [r for r in usable if r.delta is not None and not math.isnan(r.delta)]
    usable.sort(key=lambda r: abs(abs(r.delta) - target_delta))
    
    candidates: List[SpreadCandidate] = []
    skipped_no_long = 0
    skipped_low_credit = 0
    
    for r in usable:
        # Try exact width first
        long_strike = r.strike - width
        long_row = strike_map.get(long_strike)
        actual_width = width  # Default to requested width
        
        # If exact strike not found, try to find closest available strike
        if not long_row or long_row.mid is None:
            # Find available strikes below, sorted by how close they are to desired width
            available_strikes = [s for s in strike_map.keys() if s < r.strike]
            if available_strikes:
                # Sort by how close the width is to desired width
                available_strikes.sort(key=lambda s: abs((r.strike - s) - width))
                # Use the strike that gives width closest to desired, but at least width/2 away
                for candidate_long_strike in available_strikes:
                    actual_width_candidate = r.strike - candidate_long_strike
                    if actual_width_candidate >= width * 0.5:  # At least half the width
                        long_row = strike_map.get(candidate_long_strike)
                        if long_row and long_row.mid is not None:
                            long_strike = candidate_long_strike
                            actual_width = actual_width_candidate
                            if abs(actual_width - width) > 0.5:  # Only log if significantly different
                                logger.debug(f"Using adjusted width: {r.strike}/{long_strike} (width={actual_width:.1f} instead of {width:.1f})")
                            break
        
        if not long_row or long_row.mid is None:
            skipped_no_long += 1
            continue
        
        credit = r.mid - long_row.mid
        if credit < min_credit:
            skipped_low_credit += 1
            logger.debug(f"Skipping {r.strike}/{long_strike} spread: credit ${credit:.2f} < min ${min_credit:.2f}")
            continue
        
        # Use actual width (may differ from requested if we adjusted)
        candidates.append(SpreadCandidate(short=r, long=long_row, width=actual_width, credit=credit))
        if len(candidates) >= num_candidates:
            break
    
    if not candidates:
        # Provide detailed diagnostic info
        logger.warning(f"Skipped {skipped_no_long} candidates: long strike not found")
        logger.warning(f"Skipped {skipped_low_credit} candidates: credit too low (min: ${min_credit:.2f})")
        
        # Try to find what the actual credit range is
        sample_credits = []
        for r in usable[:5]:  # Check first 5 usable options
            long_strike = r.strike - width
            long_row = strike_map.get(long_strike)
            if long_row and long_row.mid is not None:
                credit = r.mid - long_row.mid
                sample_credits.append(credit)
        
        if sample_credits:
            max_credit = max(sample_credits)
            min_credit_found = min(sample_credits)
            raise ValueError(
                f"No spread candidates met the filters. "
                f"Found {len(usable)} usable short legs, but:\n"
                f"  - {skipped_no_long} skipped: long strike not found (width={width})\n"
                f"  - {skipped_low_credit} skipped: credit < ${min_credit:.2f}\n"
                f"  - Sample credit range: ${min_credit_found:.2f} - ${max_credit:.2f}\n"
                f"  - Try: --min-credit {max(0.05, min_credit_found - 0.05):.2f} or --spread-width {width + 1}"
            )
        else:
            raise ValueError(
                f"No spread candidates met the filters. "
                f"Found {len(usable)} usable short legs, but none had matching long strikes. "
                f"Try adjusting --spread-width (current: {width})"
            )
    
    return candidates

# ---------- Reporting ----------

def describe_candidate(label: str, c: SpreadCandidate) -> str:
    d_abs = abs(c.short.delta) if c.short.delta is not None and not math.isnan(c.short.delta) else None
    lines = [
        f"{label}",
        "",
        f"Sell {c.short.strike:.0f} {c.short.right} / Buy {c.long.strike:.0f} {c.long.right} "
        f"({int(c.width)}-wide)",
    ]
    if d_abs is not None:
        lines.append(f"Delta ~ {c.short.delta:.3f} (|delta| ~ {d_abs:.3f})")
    lines.append(f"Credit ~ ${c.credit:.2f}")
    if c.short.volume and not math.isnan(c.short.volume):
        lines.append(f"Short leg volume: {int(c.short.volume)}")
    return "\n".join(lines)

def print_report(candidates: List[SpreadCandidate]):
    # Sort by riskiness: higher |delta| = less conservative
    candidates_sorted = sorted(
        candidates,
        key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
        reverse=True,
    )
    
    label_map = [
        "Primary candidate (balanced)",
        "Slightly more conservative",
        "Extra conservative",
    ]
    
    print("\n=== Spread Candidates ===\n")
    for idx, c in enumerate(candidates_sorted):
        label = label_map[idx] if idx < len(label_map) else f"Candidate {idx+1}"
        print(f"[{idx+1}]")
        print(describe_candidate(label, c))
        print("-" * 50)

# ---------- IB order creation ----------

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
        
        # Find spread candidates
        candidates = find_spread_candidates(
            rows,
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
        
        # Place order (synchronized)
        with ib_lock:
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
    
    # Propose spread candidates
    candidates = find_spread_candidates(
        rows,
        width=args.spread_width,
        target_delta=args.target_delta,
        num_candidates=args.num_candidates,
        min_credit=args.min_credit,
    )
    
    # Sort candidates by riskiness: higher |delta| = less conservative
    candidates_sorted = sorted(
        candidates,
        key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
        reverse=True,
    )
    
    print_report(candidates_sorted)
    
    # Selection: auto vs interactive
    if args.risk_profile != "interactive":
        selected = choose_candidate_by_profile(candidates_sorted, args.risk_profile)
        print("\nAuto-selected spread:\n")
        print(describe_candidate("Chosen spread", selected))
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
        print(describe_candidate("Chosen spread", selected))
    
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

