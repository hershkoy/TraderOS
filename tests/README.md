# Tests

This directory contains all test files for the BackTrader Testing Framework.

## Test Files

- **test_bt.py** - Basic BackTrader functionality tests
- **test_custom_tracking.py** - Tests for custom tracking and analytics functionality
- **test_report_debug.py** - Debug tests for report generation and formatting
- **test_strategies.py** - Tests for strategy implementations
- **test_tradingview_report.py** - Tests for TradingView-style report generation
- **test_yaml_config.py** - Tests for YAML configuration system

## Running Tests

To run all tests:
```bash
python -m pytest tests/
```

To run a specific test:
```bash
python -m pytest tests/test_yaml_config.py
```

## Test Dependencies

Most tests require:
- Sample data files in the `data/` directory
- Proper environment setup (API keys, IBKR connection, etc.)
- Virtual environment activation

## Note

These tests are designed to validate the framework's functionality and may require specific data files or external services to run successfully.
