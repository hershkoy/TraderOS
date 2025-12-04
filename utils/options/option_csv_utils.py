#!/usr/bin/env python3
"""
option_csv_utils.py

CSV parsing and option data loading utilities.
"""

import csv
import logging
from typing import Dict, List, Optional, Tuple

from strategies.option_strategies import OptionRow

logger = logging.getLogger(__name__)


def parse_float_safe(x: str) -> float:
    """Safely parse a string to float, returning NaN on failure."""
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
    """
    Load option rows from CSV file.
    
    Args:
        path: Path to CSV file
        right: Option right to filter (P or C)
        expiry: Optional expiry to filter
    
    Returns:
        List of OptionRow objects
    """
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
    """Build a dictionary mapping strikes to OptionRow objects."""
    return {r.strike: r for r in rows}

