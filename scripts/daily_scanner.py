import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fetch_data import fetch_from_ib
from utils.ib_port_detector import detect_ib_port
from utils.ticker_universe import TickerUniverseManager


LOGGER = logging.getLogger("daily_scanner")
EASTERN_TZ = ZoneInfo("US/Eastern")

PCT_CHANGE_LOOKBACK = 14
BLUE_SKY_LOOKBACK = 10
VOLUME_LOOKBACK = 14
PCT_CHANGE_THRESHOLD = 30.0
DEFAULT_MIN_VOLUME = 5_000_000
DEFAULT_BARS = 80
DEFAULT_TOP = 5


@dataclass
class ScannerResult:
    symbol: str
    session_date: str
    pct_change_open14: float
    consecutive_days: int
    avg_volume_14: float
    volume: float
    open_price: float
    close_price: float
    high_price: float


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily scanner that replicates the Thinkorswim strategy using IB data."
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=DEFAULT_BARS,
        help=f"Number of daily bars to fetch per symbol (default: {DEFAULT_BARS})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Number of top results to display (default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=DEFAULT_MIN_VOLUME,
        help=f"Minimum volume threshold (default: {DEFAULT_MIN_VOLUME:,})",
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
        default="daily_scanner",
        help="Filename prefix for generated reports.",
    )
    parser.add_argument(
        "--ib-port",
        type=int,
        help="IB Gateway/Gateway port (overrides auto-detection & IB_PORT env).",
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


def fetch_daily_history(symbol: str, bars: int) -> Optional[pd.DataFrame]:
    try:
        df = fetch_from_ib(symbol, bars, "1d")
        if df is None or df.empty:
            return None

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
        df = df.sort_values("timestamp")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as exc:
        LOGGER.warning("Failed to fetch data for %s: %s", symbol, exc)
        return None


def filter_completed_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["session_date"] = df["timestamp"].dt.tz_convert(EASTERN_TZ).dt.date
    today_eastern = datetime.now(tz=EASTERN_TZ).date()
    completed = df[df["session_date"] < today_eastern]
    return completed


def compute_consecutive_true(series: pd.Series) -> int:
    count = 0
    for value in reversed(series.dropna().tolist()):
        if bool(value):
            count += 1
        else:
            break
    return count


def evaluate_symbol(
    symbol: str,
    df: pd.DataFrame,
    min_volume: int,
) -> Optional[ScannerResult]:
    completed = filter_completed_sessions(df)
    min_rows = max(
        PCT_CHANGE_LOOKBACK + 1,
        BLUE_SKY_LOOKBACK + 12,
        VOLUME_LOOKBACK + 1,
    )

    if len(completed) < min_rows:
        return None

    work_df = completed.copy()

    open_shift = work_df["open"].shift(PCT_CHANGE_LOOKBACK)
    work_df["pct_change_open14"] = (work_df["open"] - open_shift) / open_shift * 100.0

    work_df["prior10_high"] = (
        work_df["high"].shift(2).rolling(BLUE_SKY_LOOKBACK).max()
    )
    work_df["yesterday_high"] = work_df["high"].shift(1)
    work_df["blue_sky"] = work_df["yesterday_high"] > work_df["prior10_high"]

    work_df["avg_volume_14"] = work_df["volume"].rolling(VOLUME_LOOKBACK).mean()
    work_df["liquid"] = (work_df["volume"] > min_volume) | (
        work_df["avg_volume_14"] > min_volume
    )

    work_df["pct_condition"] = work_df["pct_change_open14"] > PCT_CHANGE_THRESHOLD

    current_row = work_df.iloc[-1]

    if not (
        bool(current_row["pct_condition"])
        and bool(current_row["blue_sky"])
        and bool(current_row["liquid"])
    ):
        return None

    consecutive_days = compute_consecutive_true(work_df["pct_condition"])

    return ScannerResult(
        symbol=symbol,
        session_date=current_row["session_date"].isoformat(),
        pct_change_open14=float(current_row["pct_change_open14"]),
        consecutive_days=consecutive_days,
        avg_volume_14=float(current_row["avg_volume_14"]),
        volume=float(current_row["volume"]),
        open_price=float(current_row["open"]),
        close_price=float(current_row["close"]),
        high_price=float(current_row["high"]),
    )


def run_scan(symbols: Iterable[str], bars: int, min_volume: int) -> List[ScannerResult]:
    results: List[ScannerResult] = []
    total = len(symbols)

    for idx, symbol in enumerate(symbols, start=1):
        LOGGER.info("Scanning %s (%d/%d)", symbol, idx, total)
        history = fetch_daily_history(symbol, bars)
        if history is None:
            continue

        result = evaluate_symbol(symbol, history, min_volume)
        if result:
            results.append(result)

    return results


def write_report(results: List[ScannerResult], prefix: str) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(tz=EASTERN_TZ).strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"{prefix}_{timestamp}.csv"

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved report to %s", output_path)
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

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

    symbols = load_universe(args)
    if not symbols:
        LOGGER.error("No symbols available to scan.")
        return

    results = run_scan(symbols, args.bars, args.min_volume)
    if not results:
        LOGGER.warning("No symbols matched the scanner criteria.")
        return

    df = pd.DataFrame([asdict(r) for r in results])
    df = df.sort_values(
        ["consecutive_days", "pct_change_open14"],
        ascending=[False, False],
    ).reset_index(drop=True)

    LOGGER.info("Top %d results:", min(args.top, len(df)))
    LOGGER.info(
        "\n%s",
        df.head(args.top).to_string(
            index=False,
            formatters={
                "pct_change_open14": "{:.2f}".format,
                "avg_volume_14": "{:,.0f}".format,
                "volume": "{:,.0f}".format,
            },
        ),
    )

    if args.output_report:
        write_report(results, args.report_prefix)


if __name__ == "__main__":
    main()

