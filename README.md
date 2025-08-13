
# BackTrader Testing Framework

A comprehensive backtesting framework with TradingView-style reporting.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)

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
# Fetch 5 years of NFLX 1-hour bars (capped to 10,000)
python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 10000

# Fetch 1 year of AAPL daily bars
python utils/fetch_data.py --symbol AAPL --provider alpaca --timeframe 1d --bars 365

# Fetch 6 months of TSLA 1-hour bars
python utils/fetch_data.py --symbol TSLA --provider alpaca --timeframe 1h --bars 4320
```

#### IBKR Examples
```bash
# Fetch 1 year of NFLX daily bars (capped to 3,000)
python utils/fetch_data.py --symbol NFLX --provider ib --timeframe 1d --bars 9999

# Fetch 3 months of AAPL 1-hour bars
python utils/fetch_data.py --symbol AAPL --provider ib --timeframe 1h --bars 2160
```

### Data Storage Structure

Data is automatically organized in the following structure:
```
data/
├── ALPACA/
│   ├── NFLX/
│   │   ├── 1h/
│   │   │   └── nflx_1h.parquet
│   │   └── 1d/
│   │       └── nflx_1d.parquet
│   └── AAPL/
│       └── 1d/
│           └── aapl_1d.parquet
└── IB/
    └── NFLX/
        └── 1d/
            └── nflx_1d.parquet
```

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

### Troubleshooting

- **Alpaca API Errors**: Verify your API credentials and ensure you have sufficient API quota
- **IBKR Connection Issues**: Ensure TWS/IB Gateway is running and configured for API connections on port 4001
- **Data Limits**: Respect provider-specific data caps to avoid API rate limiting

## Run Backtests

### Basic Strategy Run
```bash
python backtrader_runner_yaml.py ^
  --config default.yaml ^
  --parquet "data\ALPACA\NFLX\1h\nflx_1h.parquet" ^
  --strategy mean_reversion ^
  --log-level DEBUG
```

### Alternative Command Format
```bash
python backtrader_runner_yaml.py --config default.yaml --parquet "data\ALPACA\NFLX\1h\nflx_1h.parquet" --strategy mean_reversion
```

### Simple Strategy Test
```bash
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