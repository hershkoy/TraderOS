# Utilities

This directory contains utility scripts for the BackTrader Testing Framework.

## Utility Scripts

- **fetch_data.py** - Data fetching utility for Alpaca and IBKR
- **create_sample_report.py** - Sample report generator for testing
- **custom_tracking.py** - Custom tracking utilities for enhanced analytics
- **tradingview_report_generator.py** - TradingView-style HTML report generator
- **tradingview_style_report.html** - HTML template for TradingView-style reports

## Usage

### Data Fetching
```bash
# Fetch data from Alpaca
python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 10000

# Fetch data from IBKR
python utils/fetch_data.py --symbol NFLX --provider ib --timeframe 1d --bars 3000
```

### Sample Report Generation
```bash
# Create a sample report for testing
python utils/create_sample_report.py
```

### Custom Tracking
The `custom_tracking.py` module provides enhanced tracking capabilities that can be imported into strategies for better analytics.

### TradingView Report Generation
The `tradingview_report_generator.py` creates professional TradingView-style HTML reports with interactive charts and comprehensive analytics. It uses the `tradingview_style_report.html` template for consistent styling.

```bash
# Generate a TradingView-style report (called from main runner)
# The report generator is integrated into the main backtesting framework
```

## Note

These utilities are designed to support the main framework. The `fetch_data.py` script is particularly useful for setting up your data before running backtests, while the report generators provide professional output formatting.
