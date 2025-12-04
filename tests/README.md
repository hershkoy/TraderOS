# Tests

This directory contains all test files for the BackTrader Testing Framework, organized by test type.

## Test Organization

Tests are organized into the following categories:

### 📦 Unit Tests (`unit/`)
Isolated tests for individual functions, classes, and modules. These tests use `pytest` or `unittest` and typically use mocks to isolate the code under test.

**Files:**
- `test_option_strategies.py` - Option strategy classes (pytest)
- `test_spread_price_calculation.py` - Spread price calculations (unittest)
- `test_etl.py` - ETL operations (unittest with mocks)
- `test_pmcc_provider.py` - PMCC provider (unittest with mocks)
- `test_options_data_checks.py` - Options data validation (unittest with mocks)
- `test_assignment.py` - Assignment operations (unittest)
- `test_options_repo.py` - Options repository (unittest with mocks)
- `test_greeks.py` - Greeks calculations (unittest)
- `test_update_universe_data.py` - Universe data updates (unittest with mocks)
- `test_ticker_universe.py` - Ticker universe management (unittest with mocks)
- `test_env_loading.py` - Environment variable loading
- `test_yaml_config.py` - YAML configuration parsing
- `test_custom_tracking.py` - Custom tracking functionality
- `test_symbol_mapping_system.py` - Symbol mapping system
- `test_symbol_mapping_quick.py` - Quick symbol mapping tests
- `test_copy_fix.py` - Copy fix operations
- `test_ib_execution_converter.py` - IB execution data converter (unittest with mocks) - Converts IB API Fill objects to DataFrame format with correct Buy/Sell mapping and NetCash calculations

### 🔗 Integration Tests (`integration/`)
Tests that verify integration with external systems (databases, APIs, services).

**Files:**
- `test_timescaledb_integration.py` - TimescaleDB connection and operations
- `test_polygon_api.py` - Polygon.io API integration
- `test_fetch_data.py` - Data fetching from providers (Alpaca/IB)
- `test_historical_snapshots.py` - Historical data snapshots
- `test_eod_pricing.py` - End-of-day pricing
- `test_charting_server.py` - Charting server functionality
- `test_tradingview_report.py` - TradingView report generation

### 🔄 Functional Tests (`functional/`)
End-to-end tests that verify complete workflows and user scenarios.

**Files:**
- `test_auto_update.py` - Auto-update functionality
- `test_scanner_small.py` - Scanner on small symbol set
- `test_hl_after_ll_scanner.py` - HL after LL scanner workflow
- `test_collar_screener.py` - Collar screener workflow
- `test_bt.py` - BackTrader integration tests
- `test_strategies.py` - Strategy execution tests
- `test_leaps_strategy.py` - LEAPS strategy tests
- `test_processing_range.py` - Processing range functionality
- `test_progress_updates.py` - Progress update functionality
- `test_logging.py` - Logging system tests
- `test_report_debug.py` - Report debugging tests

### 🛠️ Utility Scripts (`utils/`)
Helper scripts and utilities (not actual tests, but useful for testing/debugging).

**Files:**
- `check_ib_orders.py` - Check IB orders via API
- `create_order_ib.py` - Create IB orders for testing
- `ib_conn.py` - IB connection utilities
- `test_failed_symbols.py` - Identify failed symbols

## Running Tests

### Run All Tests
```bash
python -m pytest tests/
```

### Run by Category
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/

# Functional tests only
python -m pytest tests/functional/
```

### Run Specific Test
```bash
python -m pytest tests/unit/test_option_strategies.py
python -m pytest tests/integration/test_timescaledb_integration.py
python -m pytest tests/functional/test_scanner_small.py
```

### Run with Verbose Output
```bash
python -m pytest tests/ -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Dependencies

Most tests require:
- Sample data files in the `data/` directory
- Proper environment setup (API keys, IBKR connection, etc.)
- Virtual environment activation
- TimescaleDB running (for integration tests)
- External API access (for integration tests)

## Test Structure

```
tests/
├── README.md                    # This file
├── unit/                        # Unit tests
│   ├── __init__.py
│   └── test_*.py
├── integration/                # Integration tests
│   ├── __init__.py
│   └── test_*.py
├── functional/                  # Functional/E2E tests
│   ├── __init__.py
│   └── test_*.py
└── utils/                        # Utility scripts
    ├── __init__.py
    └── *.py
```

## Writing New Tests

When adding new tests:
- **Unit tests** → `tests/unit/` - Test isolated functions/classes
- **Integration tests** → `tests/integration/` - Test external system integration
- **Functional tests** → `tests/functional/` - Test complete workflows
- **Utility scripts** → `tests/utils/` - Helper scripts for testing

## Notes

- Unit tests should be fast and isolated (use mocks for external dependencies)
- Integration tests may require external services to be running
- Functional tests verify end-to-end workflows
- Utility scripts are not run as part of the test suite but are useful for debugging
