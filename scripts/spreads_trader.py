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
from typing import List, Optional, Dict

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import Contract, Option, ComboLeg, Order
except ImportError:
    print("Error: ib_insync not found. Install with: pip install ib_insync")
    sys.exit(1)

from utils.fetch_data import get_ib_connection, cleanup_ib_connection

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
        raise ValueError("No usable options with delta and quotes")
    
    # Sort by closeness to target delta (absolute)
    usable.sort(key=lambda r: abs(abs(r.delta) - target_delta))
    
    candidates: List[SpreadCandidate] = []
    for r in usable:
        long_strike = r.strike - width
        long_row = strike_map.get(long_strike)
        if not long_row or long_row.mid is None:
            continue
        
        credit = r.mid - long_row.mid
        if credit < min_credit:
            continue
        
        candidates.append(SpreadCandidate(short=r, long=long_row, width=width, credit=credit))
        if len(candidates) >= num_candidates:
            break
    
    if not candidates:
        raise ValueError("No spread candidates met the credit/min-delta filters")
    
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
    tif: str = "DAY"
):
    """
    Create a SELL vertical put spread order in IB for the given candidate.
    """
    ib = get_ib_connection()
    
    try:
        INDEX_SYMBOLS = {"SPX", "RUT", "NDX", "VIX", "DJX"}
        exchange = "CBOE" if symbol.upper() in INDEX_SYMBOLS else "SMART"
        
        # Build option contracts for short & long legs
        short_opt = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=candidate.short.strike,
            right=candidate.short.right,
            exchange=exchange,
        )
        long_opt = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=candidate.long.strike,
            right=candidate.long.right,
            exchange=exchange,
        )
        
        qualified = ib.qualifyContracts(short_opt, long_opt)
        if len(qualified) != 2:
            raise RuntimeError("Could not qualify both legs with IB")
        
        short_q, long_q = qualified
        
        short_leg = ComboLeg(
            conId=short_q.conId,
            ratio=1,
            action="SELL",
            exchange=exchange,
        )
        long_leg = ComboLeg(
            conId=long_q.conId,
            ratio=1,
            action="BUY",
            exchange=exchange,
        )
        
        spread = Contract(
            symbol=symbol,
            secType="BAG",
            currency="USD",
            exchange=exchange,
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
    finally:
        # Don't cleanup connection here - let caller handle it
        pass

# ---------- Auto-fetch CSV functionality ----------

def auto_fetch_option_chain(
    symbol: str,
    expiry: Optional[str] = None,
    dte: Optional[int] = None,
    right: str = "P",
    std_dev: Optional[float] = None,
    max_strikes: int = 100,
    max_expirations: Optional[int] = None
) -> str:
    """
    Automatically fetch option chain and return path to generated CSV.
    
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
        std_dev=std_dev
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
        required=True,
        help="Underlying symbol (e.g., SPY, QQQ, SPX)",
    )
    
    parser.add_argument(
        "--expiry",
        type=str,
        default=None,
        help="Expiration in IB format (e.g., 20251120). Required if --input-csv is provided, optional if auto-fetching.",
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
        default=0.18,
        help="Minimum acceptable credit for candidate (default: 0.18)",
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
    
    args = parser.parse_args()
    
    # Determine CSV path - either use provided or auto-fetch
    csv_path = args.input_csv
    
    if csv_path is None:
        # Auto-fetch option chain
        if not args.expiry and not args.dte:
            parser.error("Either --expiry or --dte must be provided when --input-csv is omitted")
        
        csv_path = auto_fetch_option_chain(
            symbol=args.symbol,
            expiry=args.expiry,
            dte=args.dte,
            right=args.right,
            std_dev=args.std_dev,
            max_strikes=args.max_strikes,
            max_expirations=1 if args.expiry else None  # If expiry specified, only fetch that one
        )
        
        # Extract expiry from CSV if not provided
        if not args.expiry:
            # Read first row to get expiry
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                first_row = next(reader, None)
                if first_row:
                    args.expiry = first_row.get("expiry")
                    logger.info(f"Using expiry from fetched data: {args.expiry}")
    else:
        # CSV provided - expiry is required
        if not args.expiry:
            parser.error("--expiry is required when --input-csv is provided")
    
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
    
    print_report(candidates)
    
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
    
    candidates_sorted = sorted(
        candidates,
        key=lambda c: abs(c.short.delta) if c.short.delta is not None else 0.0,
        reverse=True,
    )
    
    if idx < 0 or idx >= len(candidates_sorted):
        print("Choice out of range, exiting.")
        return
    
    selected = candidates_sorted[idx]
    
    print("\nYou selected:\n")
    print(describe_candidate("Chosen spread", selected))
    
    if not args.create_orders_en:
        print(
            "\n(create-orders-en not set) No orders were sent. "
            "Re-run with --create-orders-en to place IB orders."
        )
        return
    
    # Create IB order
    print("\nCreating IB DAY order via API...")
    create_ib_spread_order(
        symbol=args.symbol,
        expiry=args.expiry,
        candidate=selected,
        quantity=args.quantity,
        tif="DAY",
    )
    
    print("Order sent to IB (check TWS).")
    
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

