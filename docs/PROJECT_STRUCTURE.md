# Project Structure

## Overview

This is a comprehensive backtesting framework with TradingView-style reporting and TimescaleDB integration. The project supports both equity and options trading strategies with professional-grade reporting and data management.

## Directory Structure

```
backTraderTest/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Docker configuration
├── Taskfile.yml                       # Task runner configuration
├── postgresql.conf                    # PostgreSQL configuration
│
├── Main Entry Points                  # Core application scripts
│   ├── backtrader_runner_yaml.py      # Main backtest runner
│   ├── scanner_runner.py              # Unified scanner runner
│   ├── hl_after_ll_scanner_runner.py  # HL scanner runner
│   ├── charting_server.py             # Charting web server
│   └── update_and_scan.py             # Update data and scan
│
├── Configuration Files                # YAML configuration files
│   ├── defaults.yaml                  # Default backtest config
│   ├── scanner_config.yaml            # Scanner configuration
│   ├── hl_after_ll_scanner_config.yaml # HL scanner config
│   └── collar_screener_config.yaml    # Collar screener config
│
├── strategies/                        # Trading strategy implementations
│   ├── mean_reversion_strategy.py     # Mean reversion (Bollinger Bands)
│   ├── mean_reversion_rsi_bt.py       # RSI-based mean reversion
│   ├── pnf_strategy.py                # Point & Figure strategy
│   ├── vcp_avwap_breakout.py          # VCP AVWAP breakout
│   ├── liquidity_sweep.py             # Liquidity sweep strategy
│   ├── weekly_bigvol_ttm_squeeze.py   # Weekly big volume TTM squeeze
│   ├── weekly_bigvol_components.py    # Weekly big volume components
│   ├── option_strategies.py           # Options strategy classes
│   └── pmcc_provider.py               # PMCC strategy provider
│
├── indicators/                        # Technical indicators
│   ├── moving_averages.py             # SMA, EMA, WMA
│   ├── momentum.py                    # RSI, MACD, Stochastic
│   ├── volume.py                      # Volume, OBV, VWAP
│   ├── trend.py                       # Bollinger Bands, ATR
│   └── supertrend.py                  # SuperTrend indicator
│
├── utils/                             # Utility modules
│   ├── timescaledb_client.py          # TimescaleDB connection
│   ├── timescaledb_loader.py          # Data loading from DB
│   ├── fetch_data.py                  # Data fetching (Alpaca/IB)
│   ├── ticker_universe.py             # Ticker universe management
│   ├── update_universe_data.py        # Universe data updater
│   ├── hl_after_ll_scanner.py         # HL after LL scanner logic
│   ├── squeeze_scanner.py             # Squeeze scanner logic
│   ├── screener_zero_cost_collar_enhanced.py # Collar screener
│   ├── tradingview_report_generator.py # Report generation
│   ├── ib_order_utils.py               # IB order utilities
│   ├── option_csv_utils.py             # Option CSV parsing
│   ├── config_processor.py            # YAML config processing
│   └── [other utility modules]
│
├── scripts/                           # Utility and data scripts
│   ├── api/                           # API integration scripts
│   │   ├── ib/                        # Interactive Brokers API
│   │   │   ├── ib_flex_multi_leg_report.py    # IB Flex query report generator
│   │   │   └── ib_option_chain_to_csv.py      # IB option chain export
│   │   └── polygon/                   # Polygon.io API
│   │       ├── polygon_backfill_contracts.py # Options data backfilling
│   │       ├── polygon_discover_contracts.py  # Contract discovery
│   │       ├── polygon_ingest_eod_quotes.py   # EOD quotes ingestion
│   │       └── run_polygon_pipeline_2y.py    # Polygon pipeline runner
│   ├── db/                            # Database operations
│   │   ├── check_database_data.py     # Database data checker
│   │   ├── check_duplicates.py        # Duplicate checker
│   │   ├── greeks_fill.py             # Greeks backfill
│   │   ├── optimize_database.py      # Database optimization
│   │   ├── quick_db_check.py         # Quick database check
│   │   ├── run_eod_prices_migration.py # EOD prices migration
│   │   └── run_snapshot_migration.py  # Snapshot migration
│   ├── data/                          # Data management and validation
│   │   ├── check_data_freshness.py    # Data freshness checker
│   │   ├── check_realtime.py          # Realtime data checker
│   │   ├── check_ticks.py             # Tick data checker
│   │   ├── debug_universe_update.py   # Universe update debugger
│   │   ├── filter_otc_stocks.py       # OTC stock filter
│   │   ├── generate_mock_options_data.py # Mock data generator
│   │   ├── identify_failed_symbols.py # Failed symbols identifier
│   │   ├── import_tickers_to_universe.py # Ticker import script
│   │   └── options_data_checks.py     # Options data validation
│   ├── scanners/                      # Scanner scripts
│   │   ├── daily_scanner.py           # Daily scanning script
│   │   └── ha_reversal_scanner.py    # HA reversal scanner
│   ├── trading/                       # Trading scripts
│   │   └── options_strategy_trader.py # Options trading script
│   └── pipeline/                      # Pipeline and automation
│       ├── run_daily_pipeline.bat     # Daily pipeline (Windows)
│       ├── run_daily_pipeline.sh      # Daily pipeline (Linux/macOS)
│       └── scheduler_setup.py         # Scheduler setup script
│
├── tests/                             # Test files
│   ├── test_*.py                       # Various test files
│   └── [test modules]
│
├── examples/                          # Example scripts
│   ├── backtrader_runner.py           # Backtest runner example
│   ├── mean_reversion_simple.py        # Simple mean reversion
│   ├── mean_reversion_rsi_example.py   # RSI example
│   ├── vcp_avwap_example.py           # VCP example
│   ├── example_hl_scanner_usage.py     # HL scanner example
│   └── [other examples]
│
├── crons/                             # Scheduled task scripts
│   ├── daily_scanner.bat              # Daily scanner batch file
│   ├── run_scanner.bat                # Scanner runner batch
│   ├── run_credit_spreads.bat         # Credit spreads batch
│   ├── run_vertical_spread_hedged.bat # Vertical spread batch
│   ├── run_collar_screener.bat        # Collar screener batch
│   ├── import_tickers.bat             # Ticker import batch
│   ├── update_and_scan.bat            # Update and scan batch
│   └── daily_spreads.yaml             # Daily spreads config
│
├── docs/                              # Documentation
│   ├── PROJECT_STRUCTURE.md           # This file
│   ├── strategies/                    # Strategy documentation
│   ├── features/                      # Feature documentation
│   ├── setup/                         # Setup and migration docs
│   └── archive/                       # Archived documentation
│
├── init-scripts/                      # Database initialization
│   ├── 01-init-database.sql           # Database init
│   ├── 02-ticker-universe-tables.sql  # Universe tables
│   ├── 03-options-schema.sql          # Options schema
│   ├── 04-option-snapshots-schema.sql # Snapshots schema
│   └── [other init scripts]
│
├── setup/                             # Setup scripts
│   └── setup_timescaledb.py           # TimescaleDB setup
│
├── data/                              # Data storage
│   ├── timescaledb/                   # TimescaleDB data
│   └── pgadmin/                       # pgAdmin data
│
├── reports/                           # Generated reports
├── logs/                              # Log files
├── exports/                           # Exported data
├── templates/                         # HTML templates
├── archive/                           # Archived files
└── venv/                              # Virtual environment
```

## Core Components

### 1. Backtesting Engine
- **Main Runner**: `backtrader_runner_yaml.py` - Runs backtests with YAML configuration
- **Strategies**: Located in `strategies/` directory
- **Indicators**: Located in `indicators/` directory
- **Reporting**: TradingView-style HTML reports with interactive charts

### 2. Data Management
- **Database**: TimescaleDB (PostgreSQL extension) for time-series data
- **Data Providers**: Alpaca Markets and Interactive Brokers (IBKR)
- **Data Fetching**: `utils/fetch_data.py` - Fetches historical data
- **Data Storage**: TimescaleDB hypertables for efficient queries

### 3. Scanning System
- **Unified Scanner**: `scanner_runner.py` - Supports multiple scanner types
- **HL After LL Scanner**: `hl_after_ll_scanner_runner.py` - Pattern detection
- **Squeeze Scanner**: Integrated in unified scanner
- **Auto-Update**: Automatically checks and updates data freshness

### 4. Options Trading
- **Options Data**: Polygon.io integration for options data
- **Screening**: Zero-cost collar screener
- **Trading**: Options strategy trader with IB integration
- **Backfilling**: Historical options contract data collection

### 5. Utilities
- **Database Utilities**: Connection, loading, migration scripts (in `scripts/db/`)
- **Data Utilities**: Freshness checking, duplicate detection, validation (in `scripts/data/`)
- **API Integration**: IB and Polygon API scripts (in `scripts/api/`)
- **Ticker Management**: Universe management and updates (in `utils/` and `scripts/data/`)
- **Reporting**: HTML report generation with charts (in `utils/`)
- **Scanners**: Scanner scripts for pattern detection (in `scripts/scanners/`)
- **Trading**: Options trading scripts (in `scripts/trading/`)
- **Pipeline**: Automation and scheduling scripts (in `scripts/pipeline/`)

## Key Features

### Data Management
- TimescaleDB for efficient time-series data storage
- Support for hourly and daily timeframes
- Automatic data resampling (1h → daily, weekly)
- Data fetching from Alpaca and IBKR APIs
- Data freshness checking and auto-update

### Strategy System
- Dynamic strategy discovery from `strategies/` directory
- Multiple built-in strategies (mean reversion, PnF, VCP, etc.)
- Custom tracking mixin for reliable performance metrics
- YAML-based configuration system

### Reporting
- TradingView-style HTML reports with interactive charts
- CSV export of individual trades
- JSON statistics export
- Plotly-based interactive visualizations

### Options Trading
- Polygon.io integration for options data
- Historical options contract backfilling
- LEAPS (Long-term Equity Anticipation Securities) support
- Options screening and analysis tools

### Database Infrastructure
- Docker Compose setup with TimescaleDB and pgAdmin
- Comprehensive database schema for market data and options
- Migration scripts and data management utilities

## Entry Points

### Running Backtests
```bash
python backtrader_runner_yaml.py --symbol NFLX --provider ALPACA --timeframe 1h --strategy mean_reversion
```

### Running Scanners
```bash
python scanner_runner.py --scanner hl_after_ll squeeze
```

### Running Charting Server
```bash
python charting_server.py
```

### Updating Data
```bash
python update_and_scan.py --scanner hl_after_ll --provider ALPACA
```

## Configuration

All configuration is done via YAML files:
- `defaults.yaml` - Default backtest configuration
- `scanner_config.yaml` - Scanner configuration
- `hl_after_ll_scanner_config.yaml` - HL scanner configuration
- `collar_screener_config.yaml` - Collar screener configuration

## Documentation Organization

- **`docs/PROJECT_STRUCTURE.md`** - This file (project structure overview)
- **`docs/strategies/`** - Strategy-specific documentation
- **`docs/features/`** - Feature-specific documentation (scanners, screeners)
- **`docs/setup/`** - Setup and migration documentation
- **`docs/archive/`** - Archived documentation

## Development Workflow

1. **Setup**: Run `python setup/setup_timescaledb.py` to initialize database
2. **Fetch Data**: Use `utils/fetch_data.py` to fetch historical data
3. **Develop Strategy**: Create strategy in `strategies/` directory
4. **Test**: Run backtest with `backtrader_runner_yaml.py`
5. **Scan**: Use scanners to find trading opportunities
6. **Report**: Review generated reports in `reports/` directory

## Notes

- All main entry point scripts are in the root directory for easy access
- Utility scripts are organized in `scripts/` directory with subfolders:
  - `scripts/api/` - API integration (IB and Polygon)
  - `scripts/db/` - Database operations
  - `scripts/data/` - Data management and validation
  - `scripts/scanners/` - Scanner scripts
  - `scripts/trading/` - Trading scripts
  - `scripts/pipeline/` - Pipeline and automation
- Tests are in `tests/` directory
- Examples are in `examples/` directory
- Configuration files are in the root directory
- Documentation is organized in `docs/` with subfolders

