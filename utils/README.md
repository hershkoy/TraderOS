# Utilities

This directory contains utility scripts for the BackTrader Testing Framework.

## Utility Scripts

- **fetch_data.py** - Data fetching utility for Alpaca and IBKR
- **create_sample_report.py** - Sample report generator for testing
- **custom_tracking.py** - Custom tracking utilities for enhanced analytics

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

## Note

These utilities are designed to support the main framework. The `fetch_data.py` script is particularly useful for setting up your data before running backtests.
