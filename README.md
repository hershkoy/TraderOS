
# BackTrader Testing Framework

A comprehensive backtesting framework with TradingView-style reporting and TimescaleDB integration.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Setup TimescaleDB (Recommended)
The framework now uses TimescaleDB for efficient time-series data storage instead of Parquet files.

```bash
# Run the setup script
python setup_timescaledb.py
```

This will:
- Check for Docker and Docker Compose
- Start TimescaleDB with persistent data storage
- Test the database connection
- Show usage examples

**Alternative Manual Setup:**
```bash
# Start TimescaleDB manually
docker-compose up -d

# Test connection
python -c "from utils.timescaledb_client import test_connection; test_connection()"
```

## Data Fetching

The framework supports fetching historical data from multiple providers. Data is automatically saved in NautilusTrader-compatible format for seamless integration.

### Supported Data Providers

#### Alpaca Markets
- **API Setup**: Requires Alpaca API credentials
- **Environment Variables**: Set `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET` in `.env` file
- **Supported Timeframes**: 1h, 1d
- **Data Cap**: Maximum 10,000 bars per request
- **Data Feed**: IEX (free tier)

#### Interactive Brokers (IBKR)
- **Setup**: Requires IBKR TWS or IB Gateway running on localhost:4001
- **Supported Timeframes**: 1h, 1d
- **Data Cap**: Maximum 3,000 bars per request
- **Connection**: Local connection to IBKR platform

### Fetch Data Commands

#### Alpaca Examples
```bash
# Fetch maximum available historical data (loops through requests)
python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars max

# Fetch 5 years of NFLX 1-hour bars (capped to 10,000)
python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 10000

# Fetch 1 year of AAPL daily bars
python utils/fetch_data.py --symbol AAPL --provider alpaca --timeframe 1d --bars 365

# Fetch 6 months of TSLA 1-hour bars
python utils/fetch_data.py --symbol TSLA --provider alpaca --timeframe 1h --bars 4320
```

#### IBKR Examples
```bash
# Fetch maximum available historical data (loops through requests)
python utils/fetch_data.py --symbol NFLX --provider ib --timeframe 1d --bars max

# Fetch 1 year of NFLX daily bars (capped to 3,000)
python utils/fetch_data.py --symbol NFLX --provider ib --timeframe 1d --bars 3000

# Fetch 3 months of AAPL 1-hour bars
python utils/fetch_data.py --symbol AAPL --provider ib --timeframe 1h --bars 2160
```

### Data Storage

Data is now stored in TimescaleDB, a PostgreSQL extension optimized for time-series data. This provides:

- **Better Performance**: Optimized for time-series queries
- **Data Integrity**: ACID compliance and data consistency
- **Scalability**: Handles large datasets efficiently
- **Query Flexibility**: SQL queries with time-series functions
- **Persistence**: Data survives container restarts

**Database Schema:**
- `market_data` table with hypertable for time-series optimization
- Indexes on symbol, provider, and timeframe for fast queries
- Automatic data deduplication and conflict resolution

**Data Organization:**
- Symbol-based queries (e.g., NFLX, AAPL)
- Provider filtering (ALPACA, IB)
- Timeframe filtering (1h, 1d)
- Time-range queries with efficient chunking

### Data Format

All fetched data is automatically converted to NautilusTrader-compatible format with the following columns:
- `ts_event`: Timestamp in nanoseconds (UTC)
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `instrument_id`: Symbol name
- `venue_id`: Data provider (ALPACA/IB)
- `timeframe`: Time interval (1h/1d)

### Environment Setup

Create a `.env` file in the project root with your API credentials:
```env
ALPACA_API_KEY_ID=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret_key
```

### Advanced Features

#### Maximum Data Fetch (`--bars max`)
The `--bars max` option fetches the maximum available historical data by automatically looping through multiple requests:

- **Automatic Pagination**: Loops through requests until no more data is available
- **Rate Limiting**: Respects API limits with configurable delays between requests
- **Error Handling**: Retries failed requests with exponential backoff
- **Deduplication**: Automatically removes duplicate data points
- **Safety Limits**: Prevents infinite loops with 1M bar limit

#### Rate Limiting & Throttling
The script includes built-in protection against API rate limiting:

- **Alpaca**: 100ms delay between requests, 3 retries with 2-second delays
- **IBKR**: 500ms delay between requests, 3 retries with 2-second delays
- **Automatic Detection**: Detects rate limit responses (429 errors) and waits accordingly
- **Logging**: Detailed logging of rate limiting events and retry attempts

### Troubleshooting

- **Alpaca API Errors**: Verify your API credentials and ensure you have sufficient API quota
- **IBKR Connection Issues**: Ensure TWS/IB Gateway is running and configured for API connections on port 4001
- **Data Limits**: Respect provider-specific data caps to avoid API rate limiting
- **Rate Limiting**: If you encounter rate limits, the script will automatically retry with delays
- **Large Data Sets**: For maximum data fetches, expect longer execution times due to rate limiting delays

## Run Backtests

### New TimescaleDB Commands (Recommended)
```bash
# Run backtest with TimescaleDB
python backtrader_runner_yaml.py ^
  --config default.yaml ^
  --symbol NFLX ^
  --provider ALPACA ^
  --timeframe 1h ^
  --strategy mean_reversion ^
  --log-level DEBUG
```

## Run Scanners

### Unified Scanner Runner
The framework includes a unified scanner runner that supports multiple scanner types:

```bash
# HL After LL Scanner (LL → HH → HL patterns)
python scanner_runner.py ^
  --config scanner_config.yaml ^
  --scanner hl_after_ll ^
  --provider ALPACA ^
  --log-level DEBUG

# Run both HL and Squeeze scanners on each stock
python scanner_runner.py --scanner hl_after_ll squeeze

# Test with specific symbols
python scanner_runner.py --scanner hl_after_ll squeeze --symbols AAPL MSFT GOOGL

# Skip data updates for faster testing
python scanner_runner.py --scanner hl_after_ll squeeze --skip-update

# Run single scanner (legacy mode)
python scanner_runner.py --scanner hl_after_ll

### Scanner Types
- **hl_after_ll**: Detects LL → HH → HL reversal patterns
- **squeeze**


**Auto-update behavior:**
- ✅ Checks data freshness for all symbols
- ✅ Updates stale data (older than 1 day by default)
- ✅ Fetches missing data for new symbols
- ✅ Uses configured provider (ALPACA/IB)
- ✅ Continues scanning even if some updates fail



### Legacy Parquet Commands (Still Supported)
```bash
# Run backtest with parquet file (legacy)
python backtrader_runner_yaml.py ^
  --config default.yaml ^
  --parquet "data\ALPACA\NFLX\1h\nflx_1h.parquet" ^
  --strategy mean_reversion ^
  --log-level DEBUG
```

### Alternative Command Formats
```bash
# TimescaleDB format
python backtrader_runner_yaml.py --symbol NFLX --provider ALPACA --timeframe 1h --strategy mean_reversion

# Legacy parquet format
python backtrader_runner_yaml.py --parquet "data\ALPACA\NFLX\1h\nflx_1h.parquet" --strategy mean_reversion
```

### Simple Strategy Test
```bash
# TimescaleDB format
python examples/mean_reversion_simple.py --symbol NFLX --provider ALPACA --timeframe 1h --lookback 30 --std 1.5 --size 2

# Legacy parquet format
python examples/mean_reversion_simple.py --parquet "data\ALPACA\NFLX\1h\nflx_1h.parquet" --lookback 30 --std 1.5 --size 2
```

## Reports

After running a backtest, the system automatically generates:

1. **TradingView-style Report** (`tradingview_report.html`) - Professional interface with chart area and 5 tabs:
   - Overview: Key metrics and equity curve
   - Performance: Detailed performance metrics
   - Trades analysis: Trade statistics breakdown
   - Risk/performance ratios: Risk metrics and ratios
   - List of trades: Individual trade details

2. **Standard HTML Report** (`backtest_report.html`) - Detailed backtesting analysis

3. **CSV Export** - Trade data in spreadsheet format

4. **JSON Statistics** - Machine-readable performance data

All reports are saved in the `reports/` folder with timestamped directories.

## Database Management

### Accessing TimescaleDB

**Via pgAdmin (Web Interface):**
- URL: http://localhost:8080
- Email: admin@backtrader.com
- Password: admin

**Via Command Line:**
```bash
# Connect to TimescaleDB
docker-compose exec timescaledb psql -U backtrader_user -d backtrader

# List available symbols
SELECT DISTINCT symbol FROM market_data ORDER BY symbol;

# Get data summary
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT symbol) as unique_symbols,
    COUNT(DISTINCT provider) as unique_providers,
    MIN(to_timestamp(ts_event / 1000000000)) as earliest_date,
    MAX(to_timestamp(ts_event / 1000000000)) as latest_date
FROM market_data;
```

### Data Management Commands

**List Available Data:**
```bash
python -c "from utils.timescaledb_loader import get_available_data; print(get_available_data())"
```

**List Available Symbols:**
```bash
python -c "from utils.timescaledb_loader import list_available_symbols; print(list_available_symbols())"
```

**Test Database Connection:**
```bash
python -c "from utils.timescaledb_client import test_connection; test_connection()"
```

### Docker Commands

**Start TimescaleDB:**
```bash
docker-compose up -d
```

**Stop TimescaleDB:**
```bash
docker-compose down
```

**View Logs:**
```bash
docker-compose logs timescaledb
```

**Reset Database (WARNING: This will delete all data):**
```bash
docker-compose down -v
docker-compose up -d
```



useful commands:

python scripts/polygon_backfill_contracts.py --underlying QQQ --days-back 700 --log-level DEBUG -log-file "logs/debug_session.log"
python scripts/polygon_backfill_contracts.py --underlying QQQ --days-back 730 --continuous --dates-per-batch 15 --delay-between-batches 120 --log-level DEBUG --log-file "logs/debug_session.log"
python scripts/polygon_backfill_contracts.py --underlying QQQ --days-back 730 --max-dates-per-run 50 --log-level DEBUG --log-file "logs/debug_session.log"

slow:
python scripts/polygon_backfill_contracts.py --underlying QQQ --days-back 730 --continuous --dates-per-batch 10 --delay-between-batches 180 --log-level DEBUG --log-file "logs/debug_session.log"

detect revesal in (SPX) 1m chart, using heikin ashi
python scripts\ha_reversal_scanner.py --debug


# Fetch expiration closest to 7 DTE
python scripts/ib_option_chain_to_csv.py --symbol QQQ --right P --dte 7 --std-dev 2.0



# Auto-fetch and analyze
python scripts/spreads_trader.py --symbol QQQ --dte 7 --target-delta 0.10
python scripts/spreads_trader.py --input-csv reports\QQQ_P_options_20251113_214807.csv --symbol QQQ --expiry 20251120
python scripts/spreads_trader.py --symbol QQQ --dte 7 --create-orders-en --quantity 2

# Place order:
risky/balanced/conservative
python scripts/ib_option_chain_to_csv.py --input-csv reports/QQQ_P_options.csv --place-order --risk-profile risky --quantity 1 --account DU123456




#daily scanner:
python scripts\daily_scanner.py --output-report