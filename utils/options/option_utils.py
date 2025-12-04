"""
Option utilities for LEAPS strategy
Handles option ID generation and other option-related functions
"""

from datetime import date
from typing import Union


def build_option_id(underlying: str, expiration: Union[date, str], strike: float, option_right: str) -> str:
    """
    Build a deterministic option ID for consistent database lookups.
    
    Args:
        underlying: The underlying symbol (e.g., 'QQQ')
        expiration: Expiration date as date object or 'YYYY-MM-DD' string
        strike: Strike price as float
        option_right: Option right as 'C' or 'P'
    
    Returns:
        Deterministic option ID in format: UNDERLYING_YYYY-MM-DD_STRIKECENTS_RIGHT
        
    Example:
        >>> build_option_id('QQQ', '2025-06-20', 350.0, 'C')
        'QQQ_2025-06-20_000350C'
    """
    # Convert expiration to string if it's a date object
    if isinstance(expiration, date):
        expiration_str = expiration.strftime('%Y-%m-%d')
    else:
        expiration_str = str(expiration)
    
    # Convert strike to cents and zero-pad to 6 digits
    strike_cents = int(round(strike * 100))
    strike_cents_str = f"{strike_cents:06d}"
    
    # Validate option right
    if option_right not in ['C', 'P']:
        raise ValueError(f"option_right must be 'C' or 'P', got '{option_right}'")
    
    # Build the option ID
    option_id = f"{underlying}_{expiration_str}_{strike_cents_str}{option_right}"
    
    return option_id


def parse_option_id(option_id: str) -> dict:
    """
    Parse an option ID back into its components.
    
    Args:
        option_id: Option ID in format UNDERLYING_YYYY-MM-DD_STRIKECENTS_RIGHT
    
    Returns:
        Dictionary with parsed components
        
    Example:
        >>> parse_option_id('QQQ_2025-06-20_000350C')
        {'underlying': 'QQQ', 'expiration': '2025-06-20', 'strike': 350.0, 'option_right': 'C'}
    """
    try:
        # Split by underscore
        parts = option_id.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid option_id format: {option_id}")
        
        underlying = parts[0]
        expiration = parts[1]
        strike_and_right = parts[2]
        
        # Extract strike and right from the last part
        if len(strike_and_right) < 7:  # 6 digits + 1 character
            raise ValueError(f"Invalid strike/right format in: {strike_and_right}")
        
        strike_cents_str = strike_and_right[:-1]
        option_right = strike_and_right[-1]
        
        # Convert strike cents back to dollars
        strike_cents = int(strike_cents_str)
        strike = strike_cents / 100.0
        
        return {
            'underlying': underlying,
            'expiration': expiration,
            'strike': strike,
            'option_right': option_right
        }
    except Exception as e:
        raise ValueError(f"Failed to parse option_id '{option_id}': {e}")


def validate_option_id(option_id: str) -> bool:
    """
    Validate that an option ID has the correct format.
    
    Args:
        option_id: Option ID to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_option_id(option_id)
        return True
    except ValueError:
        return False
