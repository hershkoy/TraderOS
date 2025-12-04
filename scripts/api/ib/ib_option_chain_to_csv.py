#!/usr/bin/env python3
"""
IB Option Chain to CSV Fetcher

Fetches option chain data from Interactive Brokers and writes to CSV.
Uses the project's shared IB API connection manager.

Usage:
    python scripts/api/ib/ib_option_chain_to_csv.py --symbol QQQ --right P --max-strikes 20 --max-expirations 3
"""

import sys
import os
import csv
import argparse
import logging
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Add project root to path for imports
# Go up 4 levels from scripts/api/ib/ib_option_chain_to_csv.py to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from ib_insync import Contract, Option, Stock, Index, Order, Trade
except ImportError:
    print("Error: ib_insync not found. Please install with: pip install ib_insync")
    sys.exit(1)

from utils.data.fetch_data import get_ib_connection, cleanup_ib_connection
try:
    from utils.api.ib.ib_port_detector import detect_ib_port
except ImportError:  # pragma: no cover
    from ib_port_detector import detect_ib_port  # type: ignore[import]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out unwanted IBKR messages
ib_logger = logging.getLogger('ib_insync.wrapper')
ib_logger.setLevel(logging.WARNING)


def get_underlying_and_chain(
    underlying_symbol: str,
    exchange: str = 'SMART',
    currency: str = 'USD',
    port: int = None
) -> Tuple[Contract, object]:
    """
    Get qualified underlying contract and option chain information.
    
    Handles special cases like SPXW (weekly options) which map to SPX as underlying
    but use SPXW as the trading class.
    
    Returns:
        Tuple of (qualified_underlying_contract, chain_object)
    """
    ib = get_ib_connection(port=port)
    
    # Map weekly option classes to their underlying symbols
    # SPXW, SPXQ are option trading classes, not underlying symbols
    WEEKLY_OPTION_CLASSES = {'SPXW', 'SPXQ'}
    original_symbol = underlying_symbol.upper()
    actual_underlying_symbol = underlying_symbol.upper()
    
    # Map weekly classes to underlying
    if original_symbol == 'SPXW':
        logger.info("Detected SPXW weekly options. Using SPX as underlying.")
        actual_underlying_symbol = 'SPX'
    elif original_symbol == 'SPXQ':
        logger.info("Detected SPXQ quarterly options. Using SPX as underlying.")
        actual_underlying_symbol = 'SPX'
    
    # Determine security type and exchange for indices
    # Index options typically trade on CBOE
    INDEX_SYMBOLS = {'SPX', 'RUT', 'NDX', 'VIX', 'DJX'}
    if actual_underlying_symbol in INDEX_SYMBOLS:
        # For indices, use Index class with CBOE exchange
        logger.info(f"Detected index symbol {actual_underlying_symbol}, using Index contract with CBOE exchange")
        underlying = Index(symbol=actual_underlying_symbol, exchange='CBOE', currency=currency)
    else:
        # For stocks/ETFs, use Stock class
        underlying = Stock(symbol=actual_underlying_symbol, exchange=exchange, currency=currency)
    
    # Qualify contract
    logger.info(f"Qualifying underlying contract for {actual_underlying_symbol}...")
    qualified_underlying = ib.qualifyContracts(underlying)
    if not qualified_underlying:
        raise ValueError(f"No qualified contracts found for {actual_underlying_symbol}")
    
    underlying = qualified_underlying[0]
    logger.info(f"Underlying contract qualified: {actual_underlying_symbol} -> conId: {underlying.conId}")
    
    # Get option chain parameters
    logger.info(f"Requesting option chain parameters for {actual_underlying_symbol}...")
    chains = ib.reqSecDefOptParams(
        actual_underlying_symbol,
        '',   # futFopExchange - usually empty for stocks/ETFs
        underlying.secType,
        underlying.conId
    )
    
    if not chains:
        raise ValueError(f'No option chains found for {actual_underlying_symbol}')
    
    # If original symbol was a weekly class (e.g., SPXW), find the matching chain
    if original_symbol in WEEKLY_OPTION_CLASSES:
        # Find chain with matching tradingClass
        matching_chain = None
        for chain in chains:
            chain_trading_class = getattr(chain, 'tradingClass', None)
            if chain_trading_class == original_symbol:
                matching_chain = chain
                logger.info(f"Found chain with tradingClass={original_symbol}")
                break
        
        if matching_chain:
            chain = matching_chain
        else:
            # Fallback to first chain if no match found
            logger.warning(f"Could not find chain with tradingClass={original_symbol}, using first available chain")
            chain = chains[0]
            if hasattr(chain, 'tradingClass'):
                logger.info(f"Using chain with tradingClass={chain.tradingClass}")
    else:
        # For regular symbols, prefer chain where tradingClass matches the symbol
        # (e.g., for TSLA, prefer tradingClass='TSLA' over '2TSLA' which is LEAPS)
        matching_chain = None
        for c in chains:
            chain_trading_class = getattr(c, 'tradingClass', None)
            if chain_trading_class == actual_underlying_symbol:
                matching_chain = c
                logger.info(f"Found chain with tradingClass={chain_trading_class} matching symbol")
                break
        
        if matching_chain:
            chain = matching_chain
        else:
            # Fallback to first chain
            chain = chains[0]
            if hasattr(chain, 'tradingClass'):
                logger.info(f"Using chain with tradingClass={chain.tradingClass}")
    
    return underlying, chain


def calculate_dte(expiry_str: str) -> int:
    """
    Calculate Days To Expiration from IB expiry string.
    
    IB expiry format is typically 'YYYYMMDD' or 'YYYYMMDD 16:00:00'
    """
    try:
        # Parse the expiry string - IB format is usually YYYYMMDD
        if ' ' in expiry_str:
            expiry_date = datetime.strptime(expiry_str.split()[0], '%Y%m%d')
        else:
            expiry_date = datetime.strptime(expiry_str, '%Y%m%d')
        
        # Calculate DTE (days to expiration)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dte = (expiry_date - today).days
        return dte
    except Exception as e:
        logger.warning(f"Could not parse expiry {expiry_str}: {e}")
        return None


def list_expirations(
    underlying_symbol: str,
    exchange: str = 'SMART',
    currency: str = 'USD',
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
    port: int = None
) -> List[Tuple[str, int]]:
    """
    List available expirations with Days To Expiration (DTE).
    
    Args:
        underlying_symbol: The underlying symbol
        exchange: Exchange for the contract
        currency: Currency
        dte_min: Minimum DTE to include (optional)
        dte_max: Maximum DTE to include (optional)
        port: IB Gateway/TWS port number (optional)
    
    Returns:
        List of tuples: [(expiry_string, dte), ...]
    """
    try:
        underlying, chain = get_underlying_and_chain(underlying_symbol, exchange, currency, port=port)
        
        logger.info(f"Found {len(chain.expirations)} available expirations")
        
        # Calculate DTE for each expiration
        expirations_with_dte = []
        for expiry in sorted(chain.expirations):
            dte = calculate_dte(expiry)
            if dte is not None:
                # Apply DTE filters if specified
                if dte_min is not None and dte < dte_min:
                    continue
                if dte_max is not None and dte > dte_max:
                    continue
                expirations_with_dte.append((expiry, dte))
        
        return expirations_with_dte
        
    except Exception as e:
        logger.error(f"Error listing expirations: {e}", exc_info=True)
        return []


def calculate_strike_std_dev(
    underlying_price: float,
    volatility: float,
    days_to_expiration: int
) -> float:
    """
    Calculate one standard deviation of price movement for option strikes.
    
    Formula: S * σ * sqrt(T)
    where:
        S = current underlying price
        σ = implied volatility (as decimal, e.g., 0.20 for 20%)
        T = time to expiration in years
    
    Args:
        underlying_price: Current underlying price
        volatility: Implied volatility as decimal (e.g., 0.20 for 20%)
        days_to_expiration: Days to expiration
    
    Returns:
        One standard deviation in price terms
    """
    T = days_to_expiration / 365.0
    if T <= 0:
        return 0.0
    return underlying_price * volatility * math.sqrt(T)


def filter_strikes_by_std_dev(
    all_strikes: List[float],
    underlying_price: float,
    volatility: float,
    days_to_expiration: int,
    std_dev_multiplier: float = 2.0
) -> List[float]:
    """
    Filter strikes within N standard deviations of current price.
    
    Args:
        all_strikes: List of all available strikes
        underlying_price: Current underlying price
        volatility: Implied volatility as decimal
        days_to_expiration: Days to expiration
        std_dev_multiplier: Number of standard deviations (default: 2.0 for "2 SD")
    
    Returns:
        Filtered list of strikes within the range
    """
    if not all_strikes or underlying_price <= 0 or volatility <= 0:
        return all_strikes
    
    std_dev = calculate_strike_std_dev(underlying_price, volatility, days_to_expiration)
    lower_bound = underlying_price - (std_dev_multiplier * std_dev)
    upper_bound = underlying_price + (std_dev_multiplier * std_dev)
    
    filtered = [s for s in all_strikes if lower_bound <= s <= upper_bound]
    return sorted(filtered)


def filter_expirations(
    all_expirations: List[str],
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
    target_dte: Optional[int] = None,
    specific_expirations: Optional[List[str]] = None
) -> List[str]:
    """
    Filter expirations based on DTE range, target DTE, or specific list.
    
    Args:
        all_expirations: List of all available expirations
        dte_min: Minimum DTE to include
        dte_max: Maximum DTE to include
        target_dte: Target DTE - finds expiration closest to this value (overrides dte_min/dte_max)
        specific_expirations: Specific expirations to use (overrides all DTE filtering)
    
    Returns:
        Filtered list of expirations
    """
    if specific_expirations:
        # Use specific expirations, but validate they exist
        valid_expirations = []
        for exp in specific_expirations:
            if exp in all_expirations:
                valid_expirations.append(exp)
            else:
                logger.warning(f"Expiration {exp} not found in available expirations")
        return sorted(valid_expirations)
    
    # If target_dte is specified, find the expiration closest to that DTE
    if target_dte is not None:
        best_expiry = None
        best_dte_diff = float('inf')
        
        for expiry in sorted(all_expirations):
            dte = calculate_dte(expiry)
            if dte is not None:
                dte_diff = abs(dte - target_dte)
                if dte_diff < best_dte_diff:
                    best_dte_diff = dte_diff
                    best_expiry = expiry
        
        if best_expiry:
            actual_dte = calculate_dte(best_expiry)
            logger.info(f"Found expiration closest to {target_dte} DTE: {best_expiry} (DTE: {actual_dte})")
            return [best_expiry]
        else:
            logger.warning(f"No expiration found with DTE close to {target_dte}")
            return []
    
    # Filter by DTE range if specified
    if dte_min is not None or dte_max is not None:
        filtered = []
        for expiry in sorted(all_expirations):
            dte = calculate_dte(expiry)
            if dte is not None:
                if dte_min is not None and dte < dte_min:
                    continue
                if dte_max is not None and dte > dte_max:
                    continue
                filtered.append(expiry)
        return filtered
    
    # No filtering - return all
    return sorted(all_expirations)


def fetch_options_for_right(
    ib,
    underlying,
    underlying_symbol: str,
    chain,
    right: str,
    expirations: list,
    selected_strikes: list
):
    """
    Fetch options for a specific right (P or C) and return the data rows.
    
    Args:
        ib: IB connection
        underlying: Underlying contract
        underlying_symbol: Underlying symbol
        chain: Option chain object from reqSecDefOptParams
        right: Option right ('P' or 'C')
        expirations: List of expiration dates
        selected_strikes: List of strike prices
    
    Returns:
        List of dictionaries containing option data
    """
    # Get trading class from chain
    trading_class = getattr(chain, 'tradingClass', None)
    
    # Use underlying symbol from qualified contract (not input symbol which might be SPXW)
    # The underlying contract will have the correct underlying symbol (e.g., SPX)
    option_symbol = underlying.symbol
    
    # Try to determine the correct exchange from the chain or underlying
    # For some symbols, we need to use a specific exchange or leave it empty
    exchange = ''  # Empty string lets IB find the contract automatically
    if hasattr(chain, 'exchange') and chain.exchange:
        exchange = chain.exchange
    elif hasattr(underlying, 'primaryExchange') and underlying.primaryExchange:
        # For QQQ, options might trade on the same exchange as underlying
        # But often options trade on CBOE even if stock is on NASDAQ
        pass  # Keep exchange empty for now
    
    logger.info(
        f"Building {len(expirations) * len(selected_strikes)} {right} option contracts "
        f"(symbol={option_symbol}, exchange='{exchange or 'auto'}', tradingClass={trading_class})..."
    )
    
    option_contracts = []
    for expiry in expirations:
        for strike in selected_strikes:
            opt = Option(
                symbol=option_symbol,  # Use underlying symbol (SPX), not input symbol (SPXW)
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,
                exchange=exchange if exchange else '',  # Empty string for auto-discovery
                tradingClass=trading_class  # This will be SPXW if input was SPXW
            )
            option_contracts.append(opt)
    
    # Qualify option contracts - try in smaller batches to avoid overwhelming IB
    logger.info(f"Qualifying {len(option_contracts)} {right} option contracts...")
    qualified_contracts = []
    
    # Qualify in batches of 50 to avoid issues
    batch_size = 50
    for i in range(0, len(option_contracts), batch_size):
        batch = option_contracts[i:i+batch_size]
        try:
            qualified_batch = ib.qualifyContracts(*batch)
            qualified_contracts.extend(qualified_batch)
            logger.debug(f"Qualified batch {i//batch_size + 1}: {len(qualified_batch)}/{len(batch)} contracts")
        except Exception as e:
            logger.warning(f"Error qualifying batch {i//batch_size + 1}: {e}")
            # Try qualifying one at a time for this batch
            for contract in batch:
                try:
                    qualified = ib.qualifyContracts(contract)
                    if qualified:
                        qualified_contracts.extend(qualified)
                except Exception as e2:
                    logger.debug(f"Could not qualify contract {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.right} {contract.strike}: {e2}")
    logger.info(f"Successfully qualified {len(qualified_contracts)} {right} option contracts")
    
    if not qualified_contracts:
        logger.warning(f"No qualified {right} option contracts found")
        return []
    
    # Request market data in batches to respect IB's 100 instrument limit
    # Use 45 to leave significant safety margin for any other active subscriptions
    MARKET_DATA_BATCH_SIZE = 45
    total_contracts = len(qualified_contracts)
    logger.info(f"Requesting market data for {total_contracts} {right} contracts in batches of {MARKET_DATA_BATCH_SIZE}...")
    
    rows = []
    data_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Process contracts in batches
    num_batches = (total_contracts + MARKET_DATA_BATCH_SIZE - 1) // MARKET_DATA_BATCH_SIZE
    for batch_idx in range(0, total_contracts, MARKET_DATA_BATCH_SIZE):
        batch = qualified_contracts[batch_idx:batch_idx + MARKET_DATA_BATCH_SIZE]
        batch_num = batch_idx // MARKET_DATA_BATCH_SIZE + 1
        
        logger.info(f"Processing batch {batch_num}/{num_batches}: {len(batch)} contracts (total: {batch_idx + len(batch)}/{total_contracts})")
        
        # Request market data for this batch
        tickers = [ib.reqMktData(c, '', False, False) for c in batch]
        
        # Wait a bit for data to come in
        ib.sleep(3)
        
        # Collect data for this batch
        if batch_num == 1:
            logger.info(f"Collecting market data at {data_timestamp}")
        
        for contract, t in zip(batch, tickers):
            # OptionGreeks lives in tickers[i].modelGreeks
            greeks = t.modelGreeks
            
            # Get market data type info (if available)
            market_data_type = 'Live'  # Default since we request type 1
            # Check if we can determine actual data type from ticker
            # Note: ib_insync doesn't expose this directly, but we log what we requested
            
            row = {
                'symbol': contract.symbol,
                'secType': contract.secType,
                'expiry': contract.lastTradeDateOrContractMonth,
                'right': contract.right,
                'strike': contract.strike,
                'bid': t.bid if t.bid is not None else '',
                'ask': t.ask if t.ask is not None else '',
                'last': t.last if t.last is not None else '',
                'volume': t.volume if t.volume is not None else '',
                'iv': greeks.impliedVol if greeks and greeks.impliedVol is not None else '',
                'delta': greeks.delta if greeks and greeks.delta is not None else '',
                'gamma': greeks.gamma if greeks and greeks.gamma is not None else '',
                'vega': greeks.vega if greeks and greeks.vega is not None else '',
                'theta': greeks.theta if greeks and greeks.theta is not None else '',
                'data_timestamp': data_timestamp,
                'market_data_type': market_data_type
            }
            
            rows.append(row)
        
        # CRITICAL: Cancel ALL market data subscriptions for this batch BEFORE starting next batch
        logger.info(f"Cancelling market data subscriptions for batch {batch_num} before next batch...")
        cancelled_count = 0
        for t in tickers:
            try:
                # Cancel using both ticker and contract to ensure cancellation
                # Some versions of ib_insync prefer one or the other
                try:
                    ib.cancelMktData(t)
                except:
                    pass
                try:
                    ib.cancelMktData(t.contract)
                except:
                    pass
                cancelled_count += 1
            except Exception as e:
                logger.debug(f"Error cancelling market data for {t.contract}: {e}")
        
        logger.info(f"Cancelled {cancelled_count}/{len(tickers)} subscriptions for batch {batch_num}")
        
        # Wait longer after cancellation to ensure IB fully processes it before next batch
        if batch_idx + MARKET_DATA_BATCH_SIZE < total_contracts:
            logger.info("Waiting 3 seconds for cancellations to fully process before next batch...")
            ib.sleep(3.0)  # Increased delay to ensure cancellations are fully processed by IB
    
    logger.info(f"Successfully collected market data for {len(rows)} {right} contracts")
    return rows


def fetch_options_to_csv(
    underlying_symbol: str,
    exchange: str = 'SMART',
    currency: str = 'USD',
    right: str = 'P',           # 'P' for puts, 'C' for calls, 'BOTH' for both
    max_strikes: int = 20,      # limit how many strikes
    max_expirations: Optional[int] = 3,   # limit how many expirations (None = all)
    output_csv: Optional[str] = None,
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
    target_dte: Optional[int] = None,
    specific_expirations: Optional[List[str]] = None,
    std_dev: Optional[float] = None,  # Filter strikes by standard deviation (e.g., 2.0 for "2 SD")
    port: int = None  # IB Gateway/TWS port number
):
    """
    Pulls option chain for the given underlying and writes bid/ask, volume, IV, delta to CSV.
    
    Args:
        underlying_symbol: The underlying symbol (e.g., 'QQQ', 'SPX', 'RUT')
        exchange: Exchange for the contract (default: 'SMART')
        currency: Currency (default: 'USD')
        right: Option type - 'P' for puts, 'C' for calls, 'BOTH' for both (default: 'P')
        max_strikes: Maximum number of strikes to fetch (default: 20)
        max_expirations: Maximum number of expirations to fetch (None = all, default: 3)
        output_csv: Output CSV file path (default: auto-generated)
        dte_min: Minimum Days To Expiration to include
        dte_max: Maximum Days To Expiration to include
        target_dte: Target DTE - finds expiration closest to this value (overrides dte_min/dte_max)
        specific_expirations: List of specific expirations to fetch (overrides DTE filtering)
        std_dev: Filter strikes by standard deviation (e.g., 2.0 for "2 SD", None = no filtering)
        port: IB Gateway/TWS port number (optional)
    """
    ib = get_ib_connection(port=port)
    
    try:
        # Request market data - try live first, fallback to delayed if not available
        # MarketDataType 1 = Live, 2 = Frozen, 3 = Delayed, 4 = Delayed-Frozen
        try:
            ib.reqMarketDataType(1)
            logger.info("Requested live market data (MarketDataType 1)")
        except Exception as e:
            logger.warning(f"Could not request live market data: {e}. Trying delayed data...")
            try:
                ib.reqMarketDataType(3)  # Delayed data
                logger.info("Requested delayed market data (MarketDataType 3)")
            except Exception as e2:
                logger.warning(f"Could not request delayed market data either: {e2}. Continuing anyway...")
        # Determine which rights to fetch
        if right.upper() == 'BOTH':
            rights_to_fetch = ['P', 'C']
            right_suffix = 'BOTH'
        else:
            rights_to_fetch = [right.upper()]
            right_suffix = right.upper()
        
        # Default output to reports/options/chains folder
        if output_csv is None:
            dt = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = f'reports/options/chains/{underlying_symbol}_{right_suffix}_options_{dt}.csv'
        elif not os.path.isabs(output_csv) and not output_csv.startswith('reports/'):
            # If relative path doesn't start with reports/, prepend it
            output_csv = f'reports/options/chains/{output_csv}'
        
        # Ensure output directory exists
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get underlying contract and chain
        underlying, chain = get_underlying_and_chain(underlying_symbol, exchange, currency, port=port)
        logger.info(f"Found option chain with {len(chain.expirations)} expirations and {len(chain.strikes)} strikes")
        
        # Filter expirations based on DTE or specific list
        filtered_expirations = filter_expirations(
            list(chain.expirations),
            dte_min=dte_min,
            dte_max=dte_max,
            target_dte=target_dte,
            specific_expirations=specific_expirations
        )
        
        # Apply max_expirations limit if specified
        if max_expirations is not None and len(filtered_expirations) > max_expirations:
            expirations = sorted(filtered_expirations)[:max_expirations]
            logger.info(f"Limited to {max_expirations} expirations (from {len(filtered_expirations)} filtered)")
        else:
            expirations = sorted(filtered_expirations)
        
        if not expirations:
            logger.error("No expirations match the specified criteria")
            return False
        
        # Log expiration details with DTE
        logger.info(f"Selected {len(expirations)} expirations:")
        for exp in expirations[:10]:  # Show first 10
            dte = calculate_dte(exp)
            logger.info(f"  {exp} (DTE: {dte})")
        if len(expirations) > 10:
            logger.info(f"  ... and {len(expirations) - 10} more")
        
        all_strikes = sorted(chain.strikes)
        
        # Optional: Filter out "weird" strikes (adjusted series, etc.)
        # Keep only normal strikes (integer values, typically 1 or 5 increments)
        def is_normal_strike(s: float) -> bool:
            # Must be an integer (or very close to it)
            if abs(round(s) - s) > 1e-6:
                return False
            # Keep all integers (they're all valid)
            return True
        
        original_strike_count = len(all_strikes)
        all_strikes = [s for s in all_strikes if is_normal_strike(s)]
        if len(all_strikes) < original_strike_count:
            logger.info(f"Filtered out {original_strike_count - len(all_strikes)} non-standard strikes (kept {len(all_strikes)} normal strikes)")
        
        # Get current underlying price and IV for filtering
        logger.info(f"Requesting current market price for {underlying_symbol}...")
        ticker = ib.reqMktData(underlying, '', False, False)
        ib.sleep(3)  # give more time to receive price
        
        # Try multiple price sources in order of preference
        underlying_price = None
        if ticker.last is not None and ticker.last > 0:
            underlying_price = ticker.last
        elif ticker.close is not None and ticker.close > 0:
            underlying_price = ticker.close
        elif ticker.bid is not None and ticker.ask is not None and ticker.bid > 0 and ticker.ask > 0:
            underlying_price = (ticker.bid + ticker.ask) / 2.0
            logger.info(f"Using bid/ask mid for price: {underlying_price}")
        
        if underlying_price is None or underlying_price <= 0:
            logger.warning(f"Could not get current price for {underlying_symbol} (last={ticker.last}, close={ticker.close}, bid={ticker.bid}, ask={ticker.ask})")
            logger.warning("Will proceed without price filtering - strikes will be selected without std dev filtering")
            underlying_price = None  # Will skip std dev filtering
        else:
            logger.info(f"Underlying {underlying_symbol} current price: {underlying_price}")
        
        # Cancel underlying market data subscription to free up slot for option contracts
        try:
            ib.cancelMktData(underlying)
            ib.sleep(0.5)  # Give IB time to process cancellation
            logger.debug("Cancelled underlying market data subscription")
        except Exception as e:
            logger.debug(f"Error cancelling underlying market data: {e}")
        
        # Filter strikes based on std dev if requested and we have a price
        if std_dev is not None and underlying_price is not None:
            # Get ATM IV from the first expiration to use for std dev calculation
            # We'll use the first expiration's DTE for the calculation
            first_expiry = expirations[0] if expirations else None
            if first_expiry:
                dte = calculate_dte(first_expiry)
                if dte is not None and dte > 0:
                    # Try to get actual ATM IV by fetching an ATM call option
                    iv_to_use = None
                    # Find closest strike to ATM
                    atm_strike = min(all_strikes, key=lambda s: abs(s - underlying_price))
                    
                    atm_ticker = None
                    qualified_atm_contract = None
                    try:
                        # Try to get IV from ATM call option using SMART exchange
                        # Use underlying symbol from qualified contract (not input symbol which might be SPXW)
                        chain_trading_class = getattr(chain, 'tradingClass', None)
                        atm_call = Option(
                            symbol=underlying.symbol,  # Use underlying symbol (SPX), not input symbol (SPXW)
                            lastTradeDateOrContractMonth=first_expiry,
                            strike=atm_strike,
                            right='C',
                            exchange='SMART',
                            tradingClass=chain_trading_class  # This will be SPXW if input was SPXW
                        )
                        qualified_atm = ib.qualifyContracts(atm_call)
                        if qualified_atm:
                            qualified_atm_contract = qualified_atm[0]
                            # Ensure we're requesting live data for ATM IV lookup
                            ib.reqMarketDataType(1)
                            atm_ticker = ib.reqMktData(qualified_atm_contract, '', False, False)
                            ib.sleep(1)
                            if atm_ticker.modelGreeks and atm_ticker.modelGreeks.impliedVol:
                                iv_to_use = atm_ticker.modelGreeks.impliedVol
                                logger.info(f"Got ATM IV from option: {iv_to_use*100:.2f}%")
                    except Exception as e:
                        logger.debug(f"Could not get ATM IV: {e}")
                    finally:
                        # Always cancel ATM option subscription
                        if qualified_atm_contract:
                            try:
                                ib.cancelMktData(qualified_atm_contract)
                                ib.sleep(0.5)  # Give IB time to process cancellation
                                logger.debug("Cancelled ATM option market data subscription")
                            except Exception as e:
                                logger.debug(f"Error cancelling ATM option market data: {e}")
                    
                    # Fallback to default IV if we couldn't get it
                    if iv_to_use is None or iv_to_use <= 0:
                        iv_to_use = 0.20  # 20% default
                        logger.info(f"Using default IV {iv_to_use*100:.1f}% for std dev calculation")
                    
                    logger.info(f"Filtering strikes by {std_dev} standard deviations (IV: {iv_to_use*100:.2f}%, DTE: {dte})")
                    
                    # Filter strikes by std dev
                    filtered_strikes = filter_strikes_by_std_dev(
                        all_strikes,
                        underlying_price,
                        iv_to_use,
                        dte,
                        std_dev
                    )
                    
                    if filtered_strikes:
                        std_dev_value = calculate_strike_std_dev(underlying_price, iv_to_use, dte)
                        logger.info(f"Filtered to {len(filtered_strikes)} strikes within {std_dev} std dev (range: {underlying_price - std_dev*std_dev_value:.2f} to {underlying_price + std_dev*std_dev_value:.2f}, from {len(all_strikes)} total)")
                        all_strikes = filtered_strikes
                    else:
                        logger.warning(f"No strikes found within {std_dev} std dev, using all strikes")
        
        # Select strikes - use only strikes that exist in the chain
        if len(all_strikes) > max_strikes:
            # Choose strikes closest to ATM if we have price, otherwise just take first N
            if underlying_price is not None:
                strikes_sorted_by_distance = sorted(
                    all_strikes,
                    key=lambda s: abs(s - underlying_price)
                )
                selected_strikes = sorted(strikes_sorted_by_distance[:max_strikes])
                logger.info(f"Selected {len(selected_strikes)} strikes closest to ATM (from {len(all_strikes)} available)")
            else:
                # No price available, just take first N strikes
                selected_strikes = sorted(all_strikes[:max_strikes])
                logger.info(f"Selected first {len(selected_strikes)} strikes (from {len(all_strikes)} available, no price for ATM selection)")
        else:
            selected_strikes = all_strikes
            logger.info(f"Using all {len(selected_strikes)} available strikes")
        
        logger.info(f'Using {len(selected_strikes)} strikes (showing first 10): {selected_strikes[:10]}')
        
        # Cancel market data for underlying
        ib.cancelMktData(underlying)
        
        # 4) Fetch options for each right type
        all_rows = []
        for right_type in rights_to_fetch:
            logger.info(f"Fetching {right_type} options...")
            rows = fetch_options_for_right(
                ib, underlying, underlying_symbol, chain,
                right_type, expirations, selected_strikes
            )
            all_rows.extend(rows)
            logger.info(f"Fetched {len(rows)} {right_type} options")
        
        if not all_rows:
            logger.error("No option data collected")
            return False
        
        # 5) Write all data to CSV
        fieldnames = [
            'symbol',
            'secType',
            'expiry',
            'right',
            'strike',
            'bid',
            'ask',
            'last',
            'volume',
            'iv',
            'delta',
            'gamma',
            'vega',
            'theta',
            'data_timestamp',
            'market_data_type'
        ]
        
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        logger.info(f"Wrote {len(all_rows)} options ({len(rights_to_fetch)} type(s)) to {output_csv}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fetching options: {e}", exc_info=True)
        return False


def calculate_mid_price(bid: float, ask: float) -> Optional[float]:
    """
    Calculate mid price from bid and ask.
    
    Args:
        bid: Bid price
        ask: Ask price
    
    Returns:
        Mid price, or None if either bid or ask is missing
    """
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def place_and_monitor_option_order(
    contract: Contract,
    quantity: int,
    initial_price: float,
    account: str = '',
    min_price: float = 0.23,
    initial_wait_minutes: int = 2,
    price_reduction_per_minute: float = 0.01,
    port: int = None
) -> Optional[Trade]:
    """
    Place an option order and monitor it with dynamic pricing strategy.
    
    Pricing strategy:
    - Start with initial_price for initial_wait_minutes
    - Then reduce price by price_reduction_per_minute every minute
    - Stop if price would go below min_price
    
    Args:
        contract: Qualified option contract
        quantity: Number of contracts
        initial_price: Initial limit price
        account: IB account ID (empty string for default account)
        min_price: Minimum price to go down to (default: 0.23)
        initial_wait_minutes: Minutes to wait at initial price (default: 2)
        price_reduction_per_minute: Price reduction per minute after initial wait (default: 0.01)
        port: IB Gateway/TWS port number (optional)
    
    Returns:
        Trade object if order is filled, None otherwise
    """
    ib = get_ib_connection(port=port)
    
    # Create limit order (BUY order - starting high and reducing to get filled)
    order = Order()
    order.action = 'BUY'  # Buying options (starting at mid+15% and reducing)
    order.orderType = 'LMT'
    order.totalQuantity = quantity
    order.lmtPrice = round(initial_price, 2)
    order.tif = 'DAY'  # Day order
    
    if account:
        order.account = account
        logger.info(f"Using account: {account}")
    
    logger.info(
        f"Placing order: {order.action} {quantity}x {contract.symbol} {contract.lastTradeDateOrContractMonth} "
        f"{contract.right} {contract.strike} @ ${order.lmtPrice:.2f}"
    )
    
    # Place order
    trade = ib.placeOrder(contract, order)
    
    if not trade:
        logger.error("Failed to place order")
        return None
    
    logger.info(f"Order placed. Order ID: {trade.order.orderId}, Status: {trade.orderStatus.status}")
    
    # Monitor order
    start_time = datetime.now()
    initial_wait_end = start_time + timedelta(minutes=initial_wait_minutes)
    current_price = initial_price
    last_price_reduction_time = initial_wait_end
    
    logger.info(f"Monitoring order. Initial price: ${current_price:.2f} for {initial_wait_minutes} minutes")
    logger.info(f"After {initial_wait_minutes} minutes, will reduce by ${price_reduction_per_minute:.2f} per minute (min: ${min_price:.2f})")
    
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
                f"Filled: {trade.orderStatus.filled}/{trade.orderStatus.totalQuantity}"
            )
        
        # Check if order is filled
        if status == 'Filled':
            logger.info(
                f"Order FILLED! Price: ${fill_price:.2f}, "
                f"Quantity: {trade.orderStatus.filled}/{trade.orderStatus.totalQuantity}"
            )
            return trade
        
        # Check if order is cancelled or rejected
        if status in ['Cancelled', 'ApiCancelled', 'Rejected']:
            logger.warning(f"Order {status.lower()}. Reason: {trade.orderStatus.whyHeld or 'N/A'}")
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
                    ib.placeOrder(contract, trade.order)
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


def get_option_delta(row: Dict[str, str]) -> float:
    """
    Extract delta value from CSV row.
    
    Args:
        row: Dictionary containing option data from CSV
    
    Returns:
        Delta value as float, or 0.0 if not available
    """
    try:
        delta_str = row.get('delta', '')
        if delta_str:
            return abs(float(delta_str))
    except (ValueError, TypeError):
        pass
    return 0.0


def choose_option_by_risk_profile(
    rows_sorted: List[Dict[str, str]],
    profile: str
) -> Dict[str, str]:
    """
    Choose an option from sorted list based on risk profile.
    
    Args:
        rows_sorted: List of option rows sorted by |delta| (highest first = riskiest)
        profile: Risk profile ('risky', 'balanced', 'conservative')
    
    Returns:
        Selected option row dictionary
    """
    n = len(rows_sorted)
    if n == 0:
        raise ValueError("No options to choose from")
    
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
    
    selected = rows_sorted[idx]
    delta = get_option_delta(selected)
    
    logger.info(
        f"Risk profile '{profile}' selected option #{idx+1} of {n} ({description}): "
        f"{selected.get('symbol')} {selected.get('expiry')} {selected.get('right')} "
        f"{selected.get('strike')} (|delta|={delta:.4f})"
    )
    
    return selected


def place_order_from_csv_row(
    row: Dict[str, str],
    quantity: int,
    account: str = '',
    min_price: float = 0.23,
    initial_wait_minutes: int = 2,
    price_reduction_per_minute: float = 0.01,
    port: int = None
) -> Optional[Trade]:
    """
    Place an order for an option based on a CSV row.
    
    Args:
        row: Dictionary containing option data from CSV (symbol, expiry, right, strike, bid, ask, etc.)
        quantity: Number of contracts
        account: IB account ID (empty string for default account)
        min_price: Minimum price to go down to (default: 0.23)
        initial_wait_minutes: Minutes to wait at initial price (default: 2)
        price_reduction_per_minute: Price reduction per minute after initial wait (default: 0.01)
        port: IB Gateway/TWS port number (optional)
    
    Returns:
        Trade object if order is filled, None otherwise
    """
    ib = get_ib_connection(port=port)
    
    # Parse CSV row data
    symbol = row['symbol']
    expiry = row['expiry']
    right = row['right']
    strike = float(row['strike'])
    
    # Get bid/ask for mid price calculation
    try:
        bid = float(row['bid']) if row['bid'] else None
        ask = float(row['ask']) if row['ask'] else None
    except (ValueError, KeyError):
        bid = None
        ask = None
    
    # Calculate initial price (mid + 15%)
    mid_price = calculate_mid_price(bid, ask)
    if mid_price is None:
        logger.error(f"Cannot calculate mid price for {symbol} {expiry} {right} {strike} (bid={bid}, ask={ask})")
        return None
    
    initial_price = mid_price * 1.15  # Mid + 15%
    logger.info(f"Mid price: ${mid_price:.2f}, Initial order price (mid + 15%): ${initial_price:.2f}")
    
    # Determine exchange - check if symbol is a weekly class that maps to an index
    WEEKLY_OPTION_CLASSES = {'SPXW', 'SPXQ'}
    INDEX_SYMBOLS = {'SPX', 'RUT', 'NDX', 'VIX', 'DJX'}
    
    # Map weekly classes to underlying for exchange determination
    underlying_symbol_for_exchange = symbol.upper()
    if underlying_symbol_for_exchange in WEEKLY_OPTION_CLASSES:
        underlying_symbol_for_exchange = 'SPX'  # SPXW/SPXQ map to SPX
    
    exchange = 'CBOE' if underlying_symbol_for_exchange in INDEX_SYMBOLS else 'SMART'
    
    # Get trading class from chain (this will handle SPXW -> SPX mapping)
    underlying, chain = get_underlying_and_chain(symbol, exchange, 'USD', port=port)
    trading_class = getattr(chain, 'tradingClass', None)
    
    # Use underlying symbol from qualified contract (not the CSV symbol which might be SPXW)
    # The underlying contract will have the correct underlying symbol (e.g., SPX)
    option_symbol = underlying.symbol
    
    # Create option contract
    option = Option(
        symbol=option_symbol,  # Use underlying symbol (SPX), not input symbol (SPXW)
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
        exchange=exchange,
        tradingClass=trading_class  # This will be SPXW if input was SPXW
    )
    
    # Qualify contract
    logger.info(f"Qualifying contract: {symbol} {expiry} {right} {strike}")
    qualified = ib.qualifyContracts(option)
    if not qualified:
        logger.error(f"Could not qualify contract: {symbol} {expiry} {right} {strike}")
        return None
    
    contract = qualified[0]
    logger.info(f"Contract qualified: conId={contract.conId}")
    
    # Place and monitor order
    return place_and_monitor_option_order(
        contract=contract,
        quantity=quantity,
        initial_price=initial_price,
        account=account,
        min_price=min_price,
        initial_wait_minutes=initial_wait_minutes,
        price_reduction_per_minute=price_reduction_per_minute,
        port=port
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fetch option chain from IB and write to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available expirations with DTE
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --list-expirations
  
  # List expirations filtered by DTE range (0-30 days)
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --list-expirations --dte-min 0 --dte-max 30
  
  # Fetch QQQ puts, 20 strikes, 3 expirations
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --max-strikes 20 --max-expirations 3
  
  # Use specific port (e.g., TWS paper trading on port 7497)
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --port 7497 --max-strikes 20
  
  # Fetch options with DTE filtering (0-45 days)
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --dte-min 0 --dte-max 45 --max-strikes 20
  
  # Fetch expiration closest to 7 DTE
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --dte 7 --max-strikes 20
  
  # Fetch specific expirations
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --expirations 20241115,20241122,20241129 --max-strikes 20
  
  # Fetch options filtered by 2 standard deviations (like IB chain viewer)
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --std-dev 2.0 --max-expirations 3
  
  # Fetch both puts and calls for QQQ in one run
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right BOTH --max-strikes 20 --max-expirations 3
  
  # Place order from CSV row (interactive selection)
  python scripts/ib_option_chain_to_csv.py --input-csv reports/QQQ_P_options_20241113.csv --place-order --quantity 1 --account DU123456
  
  # Place order for specific row from CSV
  python scripts/ib_option_chain_to_csv.py --input-csv reports/QQQ_P_options_20241113.csv --place-order --order-row 5 --quantity 1 --account DU123456
  
  # Auto-select riskiest option and place order
  python scripts/ib_option_chain_to_csv.py --input-csv reports/QQQ_P_options_20241113.csv --place-order --risk-profile risky --quantity 1 --account DU123456
  
  # Auto-select balanced option and place order
  python scripts/ib_option_chain_to_csv.py --input-csv reports/QQQ_P_options_20241113.csv --place-order --risk-profile balanced --quantity 1 --account DU123456
  
  # Auto-select conservative option and place order
  python scripts/ib_option_chain_to_csv.py --input-csv reports/QQQ_P_options_20241113.csv --place-order --risk-profile conservative --quantity 1 --account DU123456
        """
    )
    
    parser.add_argument(
        '--symbol',
        required=False,
        help='Underlying symbol (e.g., QQQ, SPX, RUT, AAPL). Required unless using --place-order with --input-csv'
    )
    
    parser.add_argument(
        '--list-expirations',
        action='store_true',
        help='List available expirations with DTE and exit (do not fetch options)'
    )
    
    parser.add_argument(
        '--right',
        choices=['P', 'C', 'BOTH'],
        default='P',
        help='Option type: P for puts, C for calls, BOTH for both (default: P)'
    )
    
    parser.add_argument(
        '--max-strikes',
        type=int,
        default=20,
        help='Maximum number of strikes to fetch (default: 20)'
    )
    
    parser.add_argument(
        '--max-expirations',
        type=int,
        default=3,
        help='Maximum number of expirations to fetch (default: 3, use 0 for all)'
    )
    
    parser.add_argument(
        '--dte-min',
        type=int,
        default=None,
        help='Minimum Days To Expiration to include'
    )
    
    parser.add_argument(
        '--dte-max',
        type=int,
        default=None,
        help='Maximum Days To Expiration to include'
    )
    
    parser.add_argument(
        '--dte',
        type=int,
        default=None,
        help='Target DTE - finds expiration closest to this value (e.g., 7 for 7 DTE, overrides --dte-min/--dte-max)'
    )
    
    parser.add_argument(
        '--expirations',
        type=str,
        default=None,
        help='Comma-separated list of specific expirations to fetch (e.g., 20241115,20241122,20241129)'
    )
    
    parser.add_argument(
        '--std-dev',
        type=float,
        default=None,
        help='Filter strikes by standard deviation (e.g., 2.0 for "2 SD" like IB chain viewer, default: no filtering)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: auto-generated in reports/ folder)'
    )
    
    parser.add_argument(
        '--exchange',
        type=str,
        default='SMART',
        help='Exchange for contracts (default: SMART)'
    )
    
    parser.add_argument(
        '--currency',
        type=str,
        default='USD',
        help='Currency (default: USD)'
    )
    
    parser.add_argument(
        '--place-order',
        action='store_true',
        help='Place order for an option from CSV (requires --input-csv)'
    )
    
    parser.add_argument(
        '--input-csv',
        type=str,
        default=None,
        help='Input CSV file to read option data from (for order placement)'
    )
    
    parser.add_argument(
        '--order-row',
        type=int,
        default=None,
        help='Row number from CSV to place order for (0-indexed). If not specified, will show interactive selection'
    )
    
    parser.add_argument(
        '--quantity',
        type=int,
        default=1,
        help='Number of contracts to order (default: 1)'
    )
    
    parser.add_argument(
        '--account',
        type=str,
        default='',
        help='IB account ID for order placement (e.g., DU123456 for paper trading, U123456 for live). Empty string uses default account. For paper trading, ensure TWS/Gateway is configured for paper account.'
    )
    
    parser.add_argument(
        '--min-price',
        type=float,
        default=0.23,
        help='Minimum price to reduce order to (default: 0.23)'
    )
    
    parser.add_argument(
        '--initial-wait-minutes',
        type=int,
        default=2,
        help='Minutes to wait at initial price before reducing (default: 2)'
    )
    
    parser.add_argument(
        '--price-reduction-per-minute',
        type=float,
        default=0.01,
        help='Price reduction per minute after initial wait (default: 0.01)'
    )
    
    parser.add_argument(
        '--risk-profile',
        choices=['interactive', 'conservative', 'balanced', 'risky'],
        default='interactive',
        help=(
            "How to choose the option when --order-row is not specified. "
            "'interactive' = ask user; "
            "'conservative' = lowest |delta| (safest); "
            "'balanced' = middle |delta|; "
            "'risky' = highest |delta| (riskiest)."
        )
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='IB Gateway/TWS port number (overrides auto-detect & IB_PORT env).',
    )
    
    args = parser.parse_args()
    
    # Log command line parameters
    logger.info("=== Command Line Parameters ===")
    for key, value in vars(args).items():
        logger.info("  %s: %s", key, value)
    logger.info("=== End Command Line Parameters ===")
    
    selected_port = args.port
    if selected_port:
        logger.info("Using IB port override: %s", selected_port)
    else:
        selected_port = detect_ib_port()
        if selected_port is None:
            logger.error(
                "Unable to auto-detect an IB port. Please set IB_PORT env or pass --port."
            )
            sys.exit(1)
        logger.info("Auto-detected IB port: %s", selected_port)
    os.environ["IB_PORT"] = str(selected_port)
    
    # Handle order placement mode
    if args.place_order:
        if not args.input_csv:
            logger.error("--place-order requires --input-csv")
            sys.exit(1)
        
        if not os.path.exists(args.input_csv):
            logger.error(f"CSV file not found: {args.input_csv}")
            sys.exit(1)
        
        # Read CSV
        rows = []
        with open(args.input_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            logger.error("CSV file is empty or has no data rows")
            sys.exit(1)
        
        logger.info(f"Loaded {len(rows)} option rows from {args.input_csv}")
        
        # Select row
        if args.order_row is not None:
            # Explicit row number specified
            if args.order_row < 0 or args.order_row >= len(rows):
                logger.error(f"Row number {args.order_row} is out of range (0-{len(rows)-1})")
                sys.exit(1)
            selected_row = rows[args.order_row]
            logger.info(f"Using explicitly specified row {args.order_row}")
        elif args.risk_profile != "interactive":
            # Auto-select based on risk profile
            # Sort by |delta| (highest first = riskiest)
            rows_sorted = sorted(
                rows,
                key=lambda r: get_option_delta(r),
                reverse=True  # Highest |delta| first (riskiest)
            )
            
            logger.info(f"Sorting options by |delta| (riskiest to most conservative)")
            selected_row = choose_option_by_risk_profile(rows_sorted, args.risk_profile)
        else:
            # Interactive selection - show options sorted by risk
            rows_sorted = sorted(
                rows,
                key=lambda r: get_option_delta(r),
                reverse=True  # Highest |delta| first (riskiest)
            )
            
            print("\nAvailable options (sorted by risk, highest |delta| first):")
            print(f"{'Row':<6} {'Symbol':<8} {'Expiry':<12} {'Right':<6} {'Strike':<10} "
                  f"{'Bid':<10} {'Ask':<10} {'Mid':<10} {'|Delta|':<10}")
            print("-" * 100)
            for i, row in enumerate(rows_sorted[:50]):  # Show first 50
                try:
                    bid = float(row.get('bid', 0) or 0)
                    ask = float(row.get('ask', 0) or 0)
                    mid = calculate_mid_price(bid, ask) or 0.0
                    delta = get_option_delta(row)
                    print(f"{i:<6} {row.get('symbol', 'N/A'):<8} {row.get('expiry', 'N/A'):<12} "
                          f"{row.get('right', 'N/A'):<6} {row.get('strike', 'N/A'):<10} "
                          f"{bid:<10.2f} {ask:<10.2f} {mid:<10.2f} {delta:<10.4f}")
                except (ValueError, KeyError):
                    delta = get_option_delta(row)
                    print(f"{i:<6} {row.get('symbol', 'N/A'):<8} {row.get('expiry', 'N/A'):<12} "
                          f"{row.get('right', 'N/A'):<6} {row.get('strike', 'N/A'):<10} "
                          f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {delta:<10.4f}")
            
            if len(rows_sorted) > 50:
                print(f"... and {len(rows_sorted) - 50} more rows")
            
            print(f"\nNote: Row 0 = riskiest (highest |delta|), Row {len(rows_sorted)-1} = most conservative (lowest |delta|)")
            
            try:
                row_num = int(input(f"\nEnter row number (0-{len(rows_sorted)-1}): "))
                if row_num < 0 or row_num >= len(rows_sorted):
                    logger.error(f"Invalid row number: {row_num}")
                    sys.exit(1)
                selected_row = rows_sorted[row_num]
            except (ValueError, KeyboardInterrupt):
                logger.error("Invalid input or cancelled")
                sys.exit(1)
        
        # Display selected option
        logger.info(f"Selected option: {selected_row.get('symbol')} {selected_row.get('expiry')} "
                   f"{selected_row.get('right')} {selected_row.get('strike')}")
        
        # Place order
        logger.info(f"Placing order: {args.quantity} contract(s), Account: {args.account or 'default'}")
        logger.info(f"Pricing strategy: Start at mid+15% for {args.initial_wait_minutes} min, "
                   f"then reduce by ${args.price_reduction_per_minute:.2f}/min (min: ${args.min_price:.2f})")
        
        trade = place_order_from_csv_row(
            row=selected_row,
            quantity=args.quantity,
            account=args.account,
            min_price=args.min_price,
            initial_wait_minutes=args.initial_wait_minutes,
            price_reduction_per_minute=args.price_reduction_per_minute,
            port=selected_port
        )
        
        if trade and trade.orderStatus.status == 'Filled':
            logger.info("Order successfully filled!")
            sys.exit(0)
        else:
            logger.warning("Order not filled or cancelled")
            sys.exit(1)
    
    # Handle list-expirations mode
    if args.list_expirations:
        logger.info(f"Listing expirations for {args.symbol}...")
        expirations_with_dte = list_expirations(
            underlying_symbol=args.symbol.upper(),
            exchange=args.exchange,
            currency=args.currency,
            dte_min=args.dte_min,
            dte_max=args.dte_max,
            port=selected_port
        )
        
        if not expirations_with_dte:
            logger.warning("No expirations found matching criteria")
            sys.exit(1)
        
        print(f"\nAvailable expirations for {args.symbol}:")
        print(f"{'Expiration':<15} {'DTE':<10}")
        print("-" * 25)
        for expiry, dte in expirations_with_dte:
            print(f"{expiry:<15} {dte:<10}")
        print(f"\nTotal: {len(expirations_with_dte)} expirations")
        
        # Show how to use specific expirations
        if len(expirations_with_dte) > 0:
            expiry_list = ','.join([exp for exp, _ in expirations_with_dte[:5]])
            print(f"\nExample: Use --expirations {expiry_list} to fetch these expirations")
        
        sys.exit(0)
    
    # Parse specific expirations if provided
    specific_expirations = None
    if args.expirations:
        specific_expirations = [exp.strip() for exp in args.expirations.split(',')]
        logger.info(f"Using specific expirations: {specific_expirations}")
    
    # Validate symbol is provided for fetch mode
    if not args.symbol:
        logger.error("--symbol is required for fetching option chains")
        sys.exit(1)
    
    # Handle max_expirations: 0 means all
    max_expirations = None if args.max_expirations == 0 else args.max_expirations
    
    right_display = args.right if args.right != 'BOTH' else 'puts and calls'
    logger.info(f"Starting option chain fetch for {args.symbol} {right_display}")
    logger.info(f"Max strikes: {args.max_strikes}")
    if max_expirations:
        logger.info(f"Max expirations: {max_expirations}")
    else:
        logger.info("Max expirations: all (unlimited)")
    if args.dte is not None:
        logger.info(f"Target DTE: {args.dte}")
    elif args.dte_min is not None or args.dte_max is not None:
        logger.info(f"DTE range: {args.dte_min or 'any'} to {args.dte_max or 'any'}")
    if args.std_dev is not None:
        logger.info(f"Strike filter: {args.std_dev} standard deviations")
    
    success = fetch_options_to_csv(
        underlying_symbol=args.symbol.upper(),
        exchange=args.exchange,
        currency=args.currency,
        right=args.right,
        max_strikes=args.max_strikes,
        max_expirations=max_expirations,
        output_csv=args.output,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        target_dte=args.dte,
        specific_expirations=specific_expirations,
        std_dev=args.std_dev,
        port=selected_port
    )
    
    if success:
        logger.info("Option chain fetch completed successfully")
    else:
        logger.error("Option chain fetch failed")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup connection
        cleanup_ib_connection()

