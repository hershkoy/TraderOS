# PEG Screener

A screener that finds stocks with PEG (Price/Earnings to Growth) ratio < 1 using:
- **Price** from IB/Alpaca
- **P/E ratio and EPS** from IB fundamentals
- **EPS growth estimates** from Finnhub

## Overview

The PEG ratio is calculated as:

```
PEG = P/E Ratio / EPS Growth Rate
```

A PEG ratio < 1 typically indicates that a stock may be undervalued relative to its growth prospects.

## Setup

### 1. Get Finnhub API Key

1. Visit [Finnhub.io](https://finnhub.io/register) and create a free account
2. Get your API key from the dashboard
3. Add it to your `.env` file:

```bash
FINNHUB_API_KEY=your_api_key_here
```

**Free Plan Limits:**
- ~60 calls per minute
- For 500 tickers, you'll make ~500 calls per week (1 call per ticker for EPS estimates)
- Well within free tier limits

### 2. IB Connection

Make sure IB Gateway or TWS is running and accessible. The screener will:
- Auto-detect the IB port, or
- Use `IB_PORT` environment variable, or
- Accept `--ib-port` command line argument

### 3. Ticker Universe

The screener uses the ticker universe from `utils/ticker_universe.py`. By default, it scans:
- Custom universe (if available), or
- S&P 500 + NASDAQ-100

## Usage

### Basic Usage

```bash
python scripts/scanners/peg_screener.py
```

This will:
1. Load tickers from the universe
2. Fetch price and P/E from IB
3. Fetch EPS growth estimates from Finnhub
4. Calculate PEG ratios
5. Display stocks with PEG < 1.0

### Command Line Options

```bash
python scripts/scanners/peg_screener.py \
    --max-peg 1.0 \           # Maximum PEG ratio (default: 1.0)
    --min-growth 5.0 \         # Minimum growth rate % (default: 5.0)
    --min-pe 5.0 \             # Minimum P/E ratio (default: 5.0)
    --limit-symbols 50 \       # Limit to first N symbols (for testing)
    --output-report \           # Save CSV report
    --verbose                  # Enable verbose logging
```

### Example: Find Undervalued Growth Stocks

```bash
# Find stocks with PEG < 0.8 and growth > 10%
python scripts/scanners/peg_screener.py \
    --max-peg 0.8 \
    --min-growth 10.0 \
    --output-report
```

### Example: Scan Specific Symbols

```bash
python scripts/scanners/peg_screener.py \
    --symbols AAPL MSFT GOOGL AMZN \
    --verbose
```

## Output

### Console Output

The screener displays results sorted by PEG ratio (lowest first):

```
================================================================================
PEG Screener Results (PEG < 1.00, Growth >= 5.0%, P/E >= 5.0)
================================================================================
symbol  price    pe_ratio  eps      eps_growth_rate  peg_ratio
AAPL    $150.00  25.50     $5.88    18.50%           1.38
MSFT    $380.00  32.00     $11.88   28.00%           1.14
...
```

### CSV Report

With `--output-report`, results are saved to `reports/peg_screener_YYYYMMDD_HHMMSS.csv`:

```csv
symbol,price,pe_ratio,eps,eps_growth_rate,peg_ratio,timestamp
AAPL,150.00,25.50,5.88,18.50,1.38,2024-12-04T10:30:00-05:00
```

## How It Works

### Data Sources

1. **Price**: Fetched from IB using `fetch_from_ib()` - gets latest daily close
2. **P/E & EPS**: Fetched from IB using `reqFundamentalData()` with `ReportSnapshot` report type
3. **Growth Rate**: Calculated from Finnhub EPS estimates using CAGR formula

### Growth Rate Calculation

The growth rate is calculated from forward EPS estimates:

1. Fetch EPS estimates for next 4 quarters from Finnhub
2. Calculate compound annual growth rate (CAGR):
   ```
   CAGR = ((last_eps / first_eps)^(1/years) - 1) * 100
   ```
3. For quarterly data, each period = 0.25 years

### PEG Calculation

```
PEG = P/E Ratio / Growth Rate (as percentage)
```

Example:
- P/E = 25.0
- Growth Rate = 20.0%
- PEG = 25.0 / 20.0 = 1.25

## Rate Limiting

The screener automatically handles rate limiting:

- **Finnhub**: 50 requests/minute (configurable via `FINNHUB_REQUESTS_PER_MINUTE`)
- **IB**: Uses existing connection pooling and retry logic

For 500 tickers:
- ~500 Finnhub calls (1 per ticker)
- ~500 IB price calls
- ~500 IB fundamental calls

Total: ~1500 API calls per run (well within free tier limits)

## Troubleshooting

### "FINNHUB_API_KEY not found"

Make sure you've:
1. Created a `.env` file in the project root
2. Added `FINNHUB_API_KEY=your_key_here`
3. Or set it as an environment variable

### "Failed to auto-detect an IB port"

Make sure:
1. IB Gateway or TWS is running
2. API connections are enabled in IB settings
3. Or specify port manually: `--ib-port 4001`

### "No fundamental data returned"

Some symbols may not have fundamental data available in IB. The screener will skip these symbols and continue.

### "Insufficient EPS estimates"

Some stocks may not have enough analyst estimates on Finnhub. The screener requires at least 2 future periods with estimates to calculate growth rate.

## Integration with Other Tools

### Batch File

A batch file is provided for easy execution:

```batch
crons\run_peg_screener.bat
```

This batch file:
- Activates the virtual environment
- Creates timestamped log files in `logs/peg_screener_YYYYMMDD_HHMMSS.log`
- Runs the screener with `--output-report` enabled
- Checks for FINNHUB_API_KEY environment variable

### Scheduled Runs

You can schedule the batch file to run automatically using Windows Task Scheduler, or add it to your existing cron setup.

The batch file handles:
- Logging to `logs/peg_screener_*.log`
- Error handling and reporting
- Environment validation

### Combine with Other Screeners

The PEG screener can be combined with other screeners:

```bash
# Run multiple screeners and combine results
python scripts/scanners/peg_screener.py --output-report --report-prefix weekly_peg
python scripts/scanners/daily_scanner.py --output-report --report-prefix weekly_daily
```

## Future Enhancements

Potential improvements:
- Support for Alpaca fundamentals (if available)
- Historical PEG tracking
- Sector/industry filtering
- Additional growth metrics (revenue growth, etc.)
- Database storage of results for trend analysis

