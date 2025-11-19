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
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import Contract, Option, ComboLeg, Order, Trade
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
        lines.append(f"Delta ≈ {c.short.delta:.3f} (|Δ| ≈ {d_abs:.3f})")
    lines.append(f"Credit ≈ ${c.credit:.2f}")
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
    limit_price = round(candidate.credit, 2)
    
    order = Order()
    order.action = "SELL"
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = limit_price
    order.tif = tif  # DAY as requested
    
    if account:
        order.account = account
        logger.info(f"Using account: {account}")
    
    trade = ib.placeOrder(spread, order)
    
    logger.info(
        "Placed IB order: SELL %s %s %dx %s/%s @ %.2f (TIF=%s)",
        symbol,
        expiry,
        quantity,
        candidate.short.strike,
        candidate.long.strike,
        limit_price,
        tif,
    )
    
    return trade


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
    
    Pricing strategy (for SELL orders):
    - Start with initial_price for initial_wait_minutes
    - Then reduce price by price_reduction_per_minute every minute
    - Stop if price would go below min_price
    
    Args:
        trade: Trade object from placed order
        spread: Spread contract
        initial_price: Initial limit price
        min_price: Minimum price to go down to (default: 0.23)
        initial_wait_minutes: Minutes to wait at initial price (default: 2)
        price_reduction_per_minute: Price reduction per minute after initial wait (default: 0.01)
        port: IB API port number (default: from IB_PORT env var or 4001)
    
    Returns:
        Trade object (may be filled, cancelled, or still pending)
    """
    ib = get_ib_connection(port=port)
    
    logger.info(f"Monitoring order. Initial price: ${initial_price:.2f} for {initial_wait_minutes} minutes")
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
            logger.info(
                f"Order status: {status}, Current limit: ${current_price:.2f}, "
                f"Fill price: ${fill_price if fill_price > 0 else 'N/A'}, "
                f"Filled: {trade.orderStatus.filled}/{trade.order.totalQuantity}"
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
            # Time to start reducing price
            minutes_since_reduction_start = (current_time - last_price_reduction_time).total_seconds() / 60.0
            
            if minutes_since_reduction_start >= 1.0:
                # Reduce price by price_reduction_per_minute
                new_price = current_price - price_reduction_per_minute
                
                # Check minimum price
                if new_price < min_price:
                    logger.info(f"Price reduction would go below minimum (${min_price:.2f}). Stopping price reduction.")
                    logger.info(f"Order will remain at ${current_price:.2f} until filled or cancelled")
                    # Continue monitoring but don't reduce price further
                    last_price_reduction_time = current_time  # Reset to prevent further reductions
                    continue
                
                # Update order price
                current_price = round(new_price, 2)
                trade.order.lmtPrice = current_price
                
                logger.info(f"Reducing limit price to ${current_price:.2f}")
                
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
        idx = n // 2
        description = "balanced (middle |delta|)"
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
    
    args = parser.parse_args()
    
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
    
    # Calculate initial price (mid credit + 15% for SELL orders)
    initial_price_raw = round(selected.credit * 1.15, 2)
    logger.info(
        f"Mid credit: ${selected.credit:.2f}, "
        f"Raw initial order price (mid + 15%): ${initial_price_raw:.2f}"
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
        # Synthetic combo market: credit spread = short_put - long_put
        combo_bid = short_bid - long_ask
        combo_ask = short_ask - long_bid
        logger.info(
            f"Synthetic combo market: bid={combo_bid:.2f}, ask={combo_ask:.2f}, "
            f"mid={(combo_bid + combo_ask) / 2:.2f}"
        )
        
        # Clamp initial price into [bid, ask] to satisfy NBBO rules
        # For a SELL, we must not be below bid or above ask.
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
        logger.warning(
            "Could not compute synthetic combo market from leg quotes; "
            "proceeding with raw initial price (may be rejected)."
        )
    
    # Respect your script's minimum credit floor
    if initial_price < args.min_price:
        logger.info(
            f"Initial price ${initial_price:.2f} is below min-price ${args.min_price:.2f}. "
            f"Raising up to min-price."
        )
        initial_price = round(args.min_price, 2)
    
    initial_price = round(initial_price, 2)
    logger.info(f"Final initial limit price: ${initial_price:.2f}")
    
    # Cancel market data subscriptions before placing order
    # (Optional - harmless if skipped, but cleaner to cancel)
    try:
        ib.cancelMktData(short_ticker)
        ib.cancelMktData(long_ticker)
    except Exception:
        pass  # Ignore cancel errors - harmless
    
    # --- Create the order ---
    order = Order()
    order.action = "SELL"
    order.orderType = "LMT"
    order.totalQuantity = args.quantity
    order.lmtPrice = initial_price
    order.tif = "DAY"
    
    if args.account:
        order.account = args.account
        logger.info("Using account: %s", args.account)
    
    trade = ib.placeOrder(spread_contract, order)
    
    logger.info(
        "Placed IB order: SELL %s %s %dx %s/%s @ %.2f (TIF=DAY)",
        args.symbol,
        args.expiry,
        args.quantity,
        selected.short.strike,
        selected.long.strike,
        initial_price,
    )
    
    if args.monitor_order:
        logger.info("Monitoring order and adjusting price according to strategy...")
        trade = monitor_and_adjust_spread_order(
            trade=trade,
            spread=spread_contract,
            initial_price=initial_price,
            min_price=args.min_price,
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

