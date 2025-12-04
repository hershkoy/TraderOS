"""
PEG (Price/Earnings to Growth) Screener

Fetches:
- Price from IB/Alpaca
- P/E or EPS from IB fundamentals
- EPS growth estimates from Finnhub

Calculates PEG ratio and filters for PEG < 1
"""

import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data.fetch_data import fetch_from_ib, get_ib_connection, create_ib_contract_with_primary_exchange
from utils.api.finnhub.finnhub_client import get_finnhub_client
from utils.api.ib.ib_port_detector import detect_ib_port
from utils.data.ticker_universe import TickerUniverseManager

LOGGER = logging.getLogger("peg_screener")
EASTERN_TZ = ZoneInfo("US/Eastern")

DEFAULT_MAX_PEG = 1.0
DEFAULT_MIN_GROWTH = 5.0  # Minimum 5% growth rate
DEFAULT_MIN_PE = 5.0  # Minimum P/E ratio (to filter out negative/zero P/E)


@dataclass
class PEGResult:
    symbol: str
    price: float
    pe_ratio: Optional[float]
    eps: Optional[float]
    eps_growth_rate: Optional[float]
    peg_ratio: Optional[float]
    timestamp: str


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PEG screener that finds stocks with PEG < 1 using IB fundamentals and Finnhub growth estimates."
    )
    parser.add_argument(
        "--max-peg",
        type=float,
        default=DEFAULT_MAX_PEG,
        help=f"Maximum PEG ratio to include (default: {DEFAULT_MAX_PEG})",
    )
    parser.add_argument(
        "--min-growth",
        type=float,
        default=DEFAULT_MIN_GROWTH,
        help=f"Minimum EPS growth rate percentage (default: {DEFAULT_MIN_GROWTH})",
    )
    parser.add_argument(
        "--min-pe",
        type=float,
        default=DEFAULT_MIN_PE,
        help=f"Minimum P/E ratio (default: {DEFAULT_MIN_PE})",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Explicit list of symbols to scan (overrides universe).",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        help="Limit the number of universe symbols (useful for testing).",
    )
    parser.add_argument(
        "--force-refresh-universe",
        action="store_true",
        help="Force refresh the ticker universe cache.",
    )
    parser.add_argument(
        "--output-report",
        action="store_true",
        help="Save a CSV report under the reports/ directory.",
    )
    parser.add_argument(
        "--report-prefix",
        default="peg_screener",
        help="Filename prefix for generated reports.",
    )
    parser.add_argument(
        "--ib-port",
        type=int,
        help="IB Gateway/TWS port (overrides auto-detection & IB_PORT env).",
    )
    parser.add_argument(
        "--provider",
        choices=["ib", "alpaca"],
        default="ib",
        help="Data provider for price and fundamentals (default: ib)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def load_universe(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        LOGGER.info("Using explicit symbol list (%d symbols)", len(symbols))
        return symbols

    manager = TickerUniverseManager()
    symbols = manager.get_combined_universe(force_refresh=args.force_refresh_universe)

    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]
        LOGGER.info(
            "Using first %d symbols from universe for testing", len(symbols)
        )
    else:
        LOGGER.info("Loaded %d symbols from ticker universe", len(symbols))

    return symbols


def fetch_latest_price_ib(symbol: str) -> Optional[float]:
    """Fetch latest price from IB"""
    try:
        # Fetch just 1 day of data to get latest price
        df = fetch_from_ib(symbol, 1, "1d")
        if df is None or df.empty:
            return None
        
        # Convert timestamp and get latest close
        df["timestamp"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
        df = df.sort_values("timestamp")
        latest_close = df.iloc[-1]["close"]
        
        LOGGER.debug(f"Latest price for {symbol}: ${latest_close:.2f}")
        return float(latest_close)
    except Exception as exc:
        LOGGER.warning(f"Failed to fetch price for {symbol}: {exc}")
        return None


def fetch_fundamentals_ib(symbol: str) -> Optional[dict]:
    """
    Fetch fundamental data from IB (P/E ratio and EPS)
    
    Uses IB's reqFundamentalData API with ReportSnapshot report type.
    Falls back to calculating from price if only one metric is available.
    
    Returns:
        Dictionary with 'pe_ratio' and 'eps' keys, or None if unavailable
    """
    try:
        from ib_insync import Stock
        import xml.etree.ElementTree as ET
        
        ib = get_ib_connection()
        
        # Create and qualify contract
        contract = create_ib_contract_with_primary_exchange(symbol)
        qualified_contracts = ib.qualifyContracts(contract)
        
        if not qualified_contracts:
            LOGGER.debug(f"Could not qualify contract for {symbol}")
            return None
        
        contract = qualified_contracts[0]
        
        # Get price first (we'll need it for calculations)
        price = fetch_latest_price_ib(symbol)
        if price is None or price <= 0:
            LOGGER.debug(f"No valid price for {symbol}")
            return None
        
        # Request fundamental data using ReportSnapshot
        # This report type contains P/E ratio and EPS
        report_type = "ReportSnapshot"
        data = {}
        
        try:
            fundamental_data = ib.reqFundamentalData(contract, report_type)
            
            if not fundamental_data:
                LOGGER.debug(f"No fundamental data returned for {symbol}")
                return None
            
            # IB returns XML string, parse it
            try:
                root = ET.fromstring(fundamental_data)
                
                # Extract P/E ratio - try multiple possible XML paths
                pe_value = None
                for path in [".//peRatio", ".//PE", ".//P/E", ".//pe_ratio"]:
                    elem = root.find(path)
                    if elem is not None and elem.text:
                        try:
                            pe_value = float(elem.text)
                            break
                        except (ValueError, TypeError):
                            continue
                
                # Extract EPS - try multiple possible XML paths
                eps_value = None
                for path in [".//earningsPerShare", ".//EPS", ".//earnings_per_share", ".//eps"]:
                    elem = root.find(path)
                    if elem is not None and elem.text:
                        try:
                            eps_value = float(elem.text)
                            break
                        except (ValueError, TypeError):
                            continue
                
                # If we got P/E but not EPS, calculate EPS from P/E and price
                if pe_value and pe_value > 0:
                    data['pe_ratio'] = pe_value
                    if not eps_value:
                        data['eps'] = price / pe_value
                    else:
                        data['eps'] = eps_value
                elif eps_value and eps_value > 0:
                    data['eps'] = eps_value
                    data['pe_ratio'] = price / eps_value
                else:
                    # No fundamental data found in XML
                    LOGGER.debug(f"Could not extract P/E or EPS from fundamental data for {symbol}")
                    return None
                
            except ET.ParseError as e:
                LOGGER.debug(f"Could not parse XML fundamental data for {symbol}: {e}")
                return None
            
        except Exception as e:
            LOGGER.debug(f"Error fetching IB fundamentals for {symbol}: {e}")
            # If fundamental data fetch fails, we can't calculate PEG
            return None
        
        # Validate the data
        if 'pe_ratio' not in data or data['pe_ratio'] <= 0:
            LOGGER.debug(f"Invalid P/E ratio for {symbol}: {data.get('pe_ratio')}")
            return None
        
        LOGGER.debug(f"Fundamentals for {symbol}: P/E={data['pe_ratio']:.2f}, EPS=${data.get('eps', 0):.2f}")
        return data
            
    except Exception as exc:
        LOGGER.debug(f"Failed to fetch fundamentals for {symbol}: {exc}")
        return None


def calculate_peg(pe_ratio: Optional[float], growth_rate: Optional[float]) -> Optional[float]:
    """
    Calculate PEG ratio
    
    PEG = P/E / Growth Rate
    
    Args:
        pe_ratio: Price-to-Earnings ratio
        growth_rate: EPS growth rate as percentage (e.g., 15.5 for 15.5%)
    
    Returns:
        PEG ratio, or None if calculation not possible
    """
    if pe_ratio is None or growth_rate is None:
        return None
    
    if growth_rate <= 0:
        return None
    
    if pe_ratio <= 0:
        return None
    
    peg = pe_ratio / growth_rate
    return peg


def evaluate_symbol(
    symbol: str,
    finnhub_client,
    args: argparse.Namespace,
) -> Optional[PEGResult]:
    """Evaluate a single symbol for PEG criteria"""
    
    # Fetch price
    price = fetch_latest_price_ib(symbol)
    if price is None or price <= 0:
        LOGGER.debug(f"Skipping {symbol}: no valid price")
        return None
    
    # Fetch fundamentals from IB
    fundamentals = fetch_fundamentals_ib(symbol)
    if not fundamentals:
        LOGGER.debug(f"Skipping {symbol}: no fundamentals available")
        return None
    
    pe_ratio = fundamentals.get('pe_ratio')
    eps = fundamentals.get('eps')
    
    # Filter by minimum P/E
    if pe_ratio is None or pe_ratio < args.min_pe:
        LOGGER.debug(f"Skipping {symbol}: P/E ratio {pe_ratio} below minimum {args.min_pe}")
        return None
    
    # Fetch EPS growth rate from Finnhub
    growth_rate = finnhub_client.get_eps_growth_rate(symbol)
    if growth_rate is None or growth_rate < args.min_growth:
        LOGGER.debug(f"Skipping {symbol}: growth rate {growth_rate}% below minimum {args.min_growth}%")
        return None
    
    # Calculate PEG
    peg_ratio = calculate_peg(pe_ratio, growth_rate)
    if peg_ratio is None:
        LOGGER.debug(f"Skipping {symbol}: could not calculate PEG")
        return None
    
    # Filter by maximum PEG
    if peg_ratio > args.max_peg:
        LOGGER.debug(f"Skipping {symbol}: PEG {peg_ratio:.2f} above maximum {args.max_peg}")
        return None
    
    return PEGResult(
        symbol=symbol,
        price=price,
        pe_ratio=pe_ratio,
        eps=eps,
        eps_growth_rate=growth_rate,
        peg_ratio=peg_ratio,
        timestamp=datetime.now(tz=EASTERN_TZ).isoformat(),
    )


def run_scan(symbols: List[str], args: argparse.Namespace) -> List[PEGResult]:
    """Run PEG scan on all symbols"""
    results: List[PEGResult] = []
    total = len(symbols)
    
    # Initialize Finnhub client
    try:
        finnhub_client = get_finnhub_client()
    except Exception as e:
        LOGGER.error(f"Failed to initialize Finnhub client: {e}")
        LOGGER.error("Make sure FINNHUB_API_KEY is set in your environment")
        return []
    
    for idx, symbol in enumerate(symbols, start=1):
        LOGGER.info("Scanning %s (%d/%d)", symbol, idx, total)
        
        try:
            result = evaluate_symbol(symbol, finnhub_client, args)
            if result:
                results.append(result)
                LOGGER.info(
                    "  Found: %s - Price=$%.2f, P/E=%.2f, Growth=%.2f%%, PEG=%.2f",
                    symbol,
                    result.price,
                    result.pe_ratio or 0,
                    result.eps_growth_rate or 0,
                    result.peg_ratio or 0,
                )
        except Exception as exc:
            LOGGER.warning(f"Error evaluating {symbol}: {exc}")
            continue
    
    # Close Finnhub client
    finnhub_client.close()
    
    return results


def write_report(results: List[PEGResult], prefix: str) -> Path:
    """Write results to CSV report"""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(tz=EASTERN_TZ).strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"{prefix}_{timestamp}.csv"

    df = pd.DataFrame([asdict(r) for r in results])
    df = df.sort_values("peg_ratio", ascending=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved report to %s", output_path)
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    
    # Log command line parameters
    LOGGER.info("=== Command Line Parameters ===")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)
    LOGGER.info("=== End Command Line Parameters ===")

    # Setup IB connection
    selected_port: Optional[int] = args.ib_port
    if selected_port:
        LOGGER.info("Using IB port override: %s", selected_port)
    else:
        selected_port = detect_ib_port()
        if selected_port is None:
            LOGGER.error(
                "Failed to auto-detect an IB port. Please specify one via --ib-port."
            )
            return
        LOGGER.info("Auto-detected IB port: %s", selected_port)

    if selected_port is not None:
        os.environ["IB_PORT"] = str(selected_port)

    # Check for Finnhub API key
    if not os.getenv("FINNHUB_API_KEY"):
        LOGGER.error("FINNHUB_API_KEY environment variable is required")
        LOGGER.error("Get your free API key from https://finnhub.io/register")
        return

    symbols = load_universe(args)
    if not symbols:
        LOGGER.error("No symbols available to scan.")
        return

    results = run_scan(symbols, args)
    if not results:
        LOGGER.warning("No symbols matched the PEG criteria.")
        return

    df = pd.DataFrame([asdict(r) for r in results])
    df = df.sort_values("peg_ratio", ascending=True).reset_index(drop=True)

    LOGGER.info("\n" + "="*80)
    LOGGER.info("PEG Screener Results (PEG < %.2f, Growth >= %.1f%%, P/E >= %.1f)", 
                args.max_peg, args.min_growth, args.min_pe)
    LOGGER.info("="*80)
    LOGGER.info(
        "\n%s",
        df.to_string(
            index=False,
            formatters={
                "price": "${:.2f}".format,
                "pe_ratio": "{:.2f}".format,
                "eps": "${:.2f}".format if df['eps'].notna().any() else "{:.2f}".format,
                "eps_growth_rate": "{:.2f}%".format,
                "peg_ratio": "{:.2f}".format,
            },
        ),
    )
    LOGGER.info("="*80)
    LOGGER.info(f"Found {len(results)} stocks matching criteria")

    if args.output_report:
        write_report(results, args.report_prefix)


if __name__ == "__main__":
    main()

