"""
Black-Scholes Greeks Calculator for Options

This module provides functions to calculate option Greeks (delta, gamma, theta, vega)
and implied volatility using the Black-Scholes model. It's used as a fallback when
Greeks are not available from the data provider.
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return norm.cdf(x)

def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return norm.pdf(x)

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        q: Dividend yield (default 0.0)
    
    Returns:
        Call option price
    """
    if T <= 0:
        return max(0, S - K)
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_price = S * math.exp(-q * T) * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    return call_price

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Black-Scholes put option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        q: Dividend yield (default 0.0)
    
    Returns:
        Put option price
    """
    if T <= 0:
        return max(0, K - S)
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    put_price = K * math.exp(-r * T) * normal_cdf(-d2) - S * math.exp(-q * T) * normal_cdf(-d1)
    return put_price

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> Dict[str, float]:
    """
    Calculate all Greeks for an option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        option_type: 'C' for call, 'P' for put
        q: Dividend yield (default 0.0)
    
    Returns:
        Dictionary with delta, gamma, theta, vega
    """
    if T <= 0:
        # At expiration, Greeks are not meaningful
        return {
            'delta': 1.0 if (option_type == 'C' and S > K) or (option_type == 'P' and S < K) else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Gamma is the same for calls and puts
    gamma = math.exp(-q * T) * normal_pdf(d1) / (S * sigma * math.sqrt(T))
    
    # Vega is the same for calls and puts
    vega = S * math.exp(-q * T) * normal_pdf(d1) * math.sqrt(T)
    
    if option_type.upper() == 'C':
        delta = math.exp(-q * T) * normal_cdf(d1)
        theta = (-S * math.exp(-q * T) * normal_pdf(d1) * sigma / (2 * math.sqrt(T)) 
                - r * K * math.exp(-r * T) * normal_cdf(d2) 
                + q * S * math.exp(-q * T) * normal_cdf(d1))
    else:  # Put
        delta = math.exp(-q * T) * (normal_cdf(d1) - 1)
        theta = (-S * math.exp(-q * T) * normal_pdf(d1) * sigma / (2 * math.sqrt(T)) 
                + r * K * math.exp(-r * T) * normal_cdf(-d2) 
                - q * S * math.exp(-q * T) * normal_cdf(-d1))
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

def implied_volatility(price: float, S: float, K: float, T: float, r: float, option_type: str, q: float = 0.0, 
                      tolerance: float = 1e-5, max_iterations: int = 100) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        price: Option price (bid/ask mid)
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        option_type: 'C' for call, 'P' for put
        q: Dividend yield (default 0.0)
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
    
    Returns:
        Implied volatility or None if not found
    """
    if T <= 0:
        return None
    
    # Initial guess for volatility
    sigma = 0.3
    
    for i in range(max_iterations):
        if option_type.upper() == 'C':
            price_guess = black_scholes_call(S, K, T, r, sigma, q)
        else:
            price_guess = black_scholes_put(S, K, T, r, sigma, q)
        
        # Calculate vega for Newton-Raphson
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.exp(-q * T) * normal_pdf(d1) * math.sqrt(T)
        
        if abs(vega) < 1e-10:
            break
        
        # Newton-Raphson update
        sigma_new = sigma - (price_guess - price) / vega
        
        # Ensure volatility is positive
        sigma_new = max(0.001, sigma_new)
        
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new
        
        sigma = sigma_new
    
    logger.warning(f"Implied volatility calculation did not converge after {max_iterations} iterations")
    return None

def calculate_option_greeks_with_iv(underlying_price: float, strike: float, days_to_exp: int, 
                                   option_price: float, option_type: str, risk_free_rate: float = 0.0, 
                                   dividend_yield: float = 0.0) -> Dict[str, Any]:
    """
    Calculate option Greeks with implied volatility.
    
    Args:
        underlying_price: Current underlying price
        strike: Strike price
        days_to_exp: Days to expiration
        option_price: Option price (bid/ask mid)
        option_type: 'C' for call, 'P' for put
        risk_free_rate: Risk-free interest rate (default 0.0)
        dividend_yield: Dividend yield (default 0.0)
    
    Returns:
        Dictionary with implied_volatility and Greeks
    """
    T = days_to_exp / 365.0
    
    # Calculate implied volatility
    iv = implied_volatility(option_price, underlying_price, strike, T, risk_free_rate, option_type, dividend_yield)
    
    if iv is None:
        return {
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'theta': None,
            'vega': None
        }
    
    # Calculate Greeks
    greeks = calculate_greeks(underlying_price, strike, T, risk_free_rate, iv, option_type, dividend_yield)
    
    return {
        'implied_volatility': iv,
        **greeks
    }

def update_option_quotes_greeks(conn, ts: str, option_id: str, underlying_price: float, 
                               option_price: float, option_type: str, days_to_exp: int,
                               risk_free_rate: float = 0.0, dividend_yield: float = 0.0) -> bool:
    """
    Update option quotes with calculated Greeks.
    
    Args:
        conn: Database connection
        ts: Timestamp
        option_id: Option identifier
        underlying_price: Underlying price
        option_price: Option price (bid/ask mid)
        option_type: 'C' for call, 'P' for put
        days_to_exp: Days to expiration
        risk_free_rate: Risk-free interest rate
        dividend_yield: Dividend yield
    
    Returns:
        True if update was successful
    """
    try:
        # Extract strike from option_id (format: QQQ_2025-06-20_000350C)
        parts = option_id.split('_')
        if len(parts) != 3:
            logger.error(f"Invalid option_id format: {option_id}")
            return False
        
        strike_cents = int(parts[2][:-1])  # Remove the last character (C/P)
        strike = strike_cents / 100.0
        
        # Calculate Greeks
        greeks_result = calculate_option_greeks_with_iv(
            underlying_price, strike, days_to_exp, option_price, option_type, 
            risk_free_rate, dividend_yield
        )
        
        if greeks_result['implied_volatility'] is None:
            logger.warning(f"Could not calculate IV for {option_id}")
            return False
        
        # Update database
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE option_quotes 
                SET iv = %s, delta = %s, gamma = %s, theta = %s, vega = %s
                WHERE option_id = %s AND ts = %s
            """, (
                greeks_result['implied_volatility'],
                greeks_result['delta'],
                greeks_result['gamma'],
                greeks_result['theta'],
                greeks_result['vega'],
                option_id,
                ts
            ))
            
            if cursor.rowcount > 0:
                logger.info(f"Updated Greeks for {option_id} at {ts}")
                return True
            else:
                logger.warning(f"No rows updated for {option_id} at {ts}")
                return False
                
    except Exception as e:
        logger.error(f"Error updating Greeks for {option_id}: {e}")
        return False

def batch_update_greeks(conn, quotes_data: list, risk_free_rate: float = 0.0, dividend_yield: float = 0.0) -> Dict[str, int]:
    """
    Batch update Greeks for multiple option quotes.
    
    Args:
        conn: Database connection
        quotes_data: List of dictionaries with quote data
        risk_free_rate: Risk-free interest rate
        dividend_yield: Dividend yield
    
    Returns:
        Dictionary with success and failure counts
    """
    success_count = 0
    failure_count = 0
    
    for quote in quotes_data:
        success = update_option_quotes_greeks(
            conn,
            quote['ts'],
            quote['option_id'],
            quote['underlying_price'],
            quote['option_price'],
            quote['option_type'],
            quote['days_to_exp'],
            risk_free_rate,
            dividend_yield
        )
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    return {
        'success_count': success_count,
        'failure_count': failure_count
    }
