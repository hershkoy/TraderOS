#!/usr/bin/env python3
"""
IB Option Chain to CSV Fetcher

Fetches option chain data from Interactive Brokers and writes to CSV.
Uses the project's shared IB API connection manager.

Usage:
    python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --max-strikes 20 --max-expirations 3
"""

import sys
import os
import csv
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import Contract, Option, Stock, Index
except ImportError:
    print("Error: ib_insync not found. Please install with: pip install ib_insync")
    sys.exit(1)

from utils.fetch_data import get_ib_connection, cleanup_ib_connection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out unwanted IBKR messages
ib_logger = logging.getLogger('ib_insync.wrapper')
ib_logger.setLevel(logging.WARNING)


def fetch_options_to_csv(
    underlying_symbol: str,
    exchange: str = 'SMART',
    currency: str = 'USD',
    right: str = 'P',           # 'P' for puts, 'C' for calls
    max_strikes: int = 20,      # limit how many strikes
    max_expirations: int = 3,   # limit how many expirations
    output_csv: Optional[str] = None
):
    """
    Pulls option chain for the given underlying and writes bid/ask, volume, IV, delta to CSV.
    
    Args:
        underlying_symbol: The underlying symbol (e.g., 'QQQ', 'SPX', 'RUT')
        exchange: Exchange for the contract (default: 'SMART')
        currency: Currency (default: 'USD')
        right: Option type - 'P' for puts, 'C' for calls (default: 'P')
        max_strikes: Maximum number of strikes to fetch (default: 20)
        max_expirations: Maximum number of expirations to fetch (default: 3)
        output_csv: Output CSV file path (default: auto-generated)
    """
    ib = get_ib_connection()
    
    try:
        if output_csv is None:
            dt = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = f'{underlying_symbol}_{right}_options_{dt}.csv'
        
        # Ensure output directory exists
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1) Define the underlying contract
        # For ETFs/Stocks: SecType='STK'
        # For index (SPX, RUT): SecType='IND'
        sec_type = 'STK'
        if underlying_symbol in ['SPX', 'RUT', 'NDX', 'VIX']:
            sec_type = 'IND'
        
        underlying = Contract()
        underlying.symbol = underlying_symbol
        underlying.secType = sec_type
        underlying.currency = currency
        underlying.exchange = exchange
        
        # 2) Qualify contract (IB adds conId etc.)
        logger.info(f"Qualifying underlying contract for {underlying_symbol}...")
        qualified_underlying = ib.qualifyContracts(underlying)
        if not qualified_underlying:
            logger.error(f"No qualified contracts found for {underlying_symbol}")
            return False
        
        underlying = qualified_underlying[0]
        logger.info(f"Underlying contract qualified: {underlying_symbol} -> conId: {underlying.conId}")
        
        # 3) Get option chain parameters
        logger.info(f"Requesting option chain parameters for {underlying_symbol}...")
        chains = ib.reqSecDefOptParams(
            underlying_symbol,
            '',   # futFopExchange - usually empty for stocks/ETFs
            underlying.secType,
            underlying.conId
        )
        
        if not chains:
            logger.error(f'No option chains found for {underlying_symbol}')
            return False
        
        chain = chains[0]  # usually one main chain
        logger.info(f"Found option chain with {len(chain.expirations)} expirations and {len(chain.strikes)} strikes")
        
        # Limit expirations and strikes so we don't explode requests
        expirations = sorted(chain.expirations)[:max_expirations]
        strikes = sorted(chain.strikes)
        
        # Optional: center strikes around ATM (for huge lists like SPX)
        # Get current underlying price
        logger.info(f"Requesting current market price for {underlying_symbol}...")
        ticker = ib.reqMktData(underlying, '', False, False)
        ib.sleep(2)  # give time to receive price
        
        underlying_price = ticker.last if ticker.last is not None else ticker.close
        if underlying_price is None:
            # fallback: just take first N strikes if we couldn't get a price
            logger.warning(f"Could not get current price for {underlying_symbol}, using first {max_strikes} strikes")
            selected_strikes = strikes[:max_strikes]
        else:
            # choose strikes closest to ATM
            logger.info(f"Underlying {underlying_symbol} current price: {underlying_price}")
            strikes_sorted_by_distance = sorted(
                strikes,
                key=lambda s: abs(s - underlying_price)
            )
            selected_strikes = sorted(strikes_sorted_by_distance[:max_strikes])
        
        logger.info(f'Using expirations: {expirations}')
        logger.info(f'Using {len(selected_strikes)} strikes (showing first 10): {selected_strikes[:10]}')
        
        # Cancel market data for underlying
        ib.cancelMktData(underlying)
        
        # 4) Build option contracts
        logger.info(f"Building {len(expirations) * len(selected_strikes)} option contracts...")
        option_contracts = []
        for expiry in expirations:
            for strike in selected_strikes:
                opt = Option(
                    symbol=underlying_symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right=right,
                    exchange=exchange
                )
                option_contracts.append(opt)
        
        # 5) Qualify option contracts
        logger.info(f"Qualifying {len(option_contracts)} option contracts...")
        qualified_contracts = ib.qualifyContracts(*option_contracts)
        logger.info(f"Successfully qualified {len(qualified_contracts)} option contracts")
        
        if not qualified_contracts:
            logger.error("No qualified option contracts found")
            return False
        
        # 6) Request market data for all contracts
        logger.info(f"Requesting market data for {len(qualified_contracts)} contracts...")
        tickers = [ib.reqMktData(c, '', False, False) for c in qualified_contracts]
        
        # Wait a bit for data to come in
        logger.info("Waiting for market data to arrive...")
        ib.sleep(3)
        
        # 7) Collect data and write to CSV
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
            'theta'
        ]
        
        rows_written = 0
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for contract, t in zip(qualified_contracts, tickers):
                # OptionGreeks lives in tickers[i].modelGreeks
                greeks = t.modelGreeks
                
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
                    'theta': greeks.theta if greeks and greeks.theta is not None else ''
                }
                
                writer.writerow(row)
                rows_written += 1
        
        logger.info(f"Wrote {rows_written} options to {output_csv}")
        
        # 8) Cancel all market data subscriptions
        logger.info("Cancelling market data subscriptions...")
        for t in tickers:
            ib.cancelMktData(t.contract)
        
        return True
        
    except Exception as e:
        logger.error(f"Error fetching options: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fetch option chain from IB and write to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch QQQ puts, 20 strikes, 3 expirations
  python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --max-strikes 20 --max-expirations 3
  
  # Fetch SPX calls, 15 strikes, 2 expirations
  python scripts/ib_option_chain_to_csv.py --symbol SPX --right C --max-strikes 15 --max-expirations 2
  
  # Fetch RUT puts with custom output file
  python scripts/ib_option_chain_to_csv.py --symbol RUT --right P --output results/rut_puts.csv
        """
    )
    
    parser.add_argument(
        '--symbol',
        required=True,
        help='Underlying symbol (e.g., QQQ, SPX, RUT, AAPL)'
    )
    
    parser.add_argument(
        '--right',
        choices=['P', 'C'],
        default='P',
        help='Option type: P for puts, C for calls (default: P)'
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
        help='Maximum number of expirations to fetch (default: 3)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: auto-generated)'
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
    
    args = parser.parse_args()
    
    logger.info(f"Starting option chain fetch for {args.symbol} {args.right}")
    logger.info(f"Max strikes: {args.max_strikes}, Max expirations: {args.max_expirations}")
    
    success = fetch_options_to_csv(
        underlying_symbol=args.symbol.upper(),
        exchange=args.exchange,
        currency=args.currency,
        right=args.right,
        max_strikes=args.max_strikes,
        max_expirations=args.max_expirations,
        output_csv=args.output
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

