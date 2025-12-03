# Universe Data Updater

A comprehensive script to fetch maximum available bars for all tickers in the ticker universe using the existing `fetch_data.py` functionality. This tool integrates the ticker universe management system with your data fetching pipeline.

## Features

- **Automatic Universe Integration**: Uses the ticker universe system to get all available tickers
- **Batch Processing**: Process tickers in configurable batches with delays
- **Resume Capability**: Resume interrupted updates from any point
- **Multiple Providers**: Support for both Alpaca and IBKR data providers
- **Flexible Timeframes**: 1m, 15m, 1h, and 1d data support
- **Rate Limiting**: Built-in delays to respect API rate limits
- **Comprehensive Logging**: Detailed logs with progress tracking
- **Dry Run Mode**: Preview what would be processed without fetching data
- **Progress Tracking**: Monitor success/failure rates and processing statistics
- **Symbol Mapping (IB)**: Automatic symbol conversion and caching for IBKR compatibility
- **Retry Logic**: Exponential backoff retry mechanism for timeout errors
- **Non-Blocking Database Saves**: Background worker thread for efficient data persistence
- **Custom Universe Files**: Load tickers from custom files instead of default universe
- **Date-Based Fetching**: Fetch data from a specific start date
- **Debugging Tools**: Test individual symbols and view symbol mappings

## Quick Start

### 1. Basic Usage



```bash

# normal usage
python utils/update_universe_data.py --provider ib --timeframe 1h --max-bars --skip-existing

# Fetch intraday 15m data (IB provider shown, Alpaca also supported)
python utils/update_universe_data.py --provider ib --timeframe 15m --max-bars

# Update all tickers with Alpaca daily data
python utils/update_universe_data.py --provider alpaca --timeframe 1d

# Update with hourly data
python utils/update_universe_data.py --provider alpaca --timeframe 1h
```

### 2. Batch Processing

```bash
# Process in batches of 20 tickers
python utils/update_universe_data.py --provider alpaca --timeframe 1d --batch-size 20

# Custom delays between batches and tickers
python utils/update_universe_data.py \
  --provider alpaca \
  --timeframe 1d \
  --batch-size 15 \
  --delay-batches 10.0 \
  --delay-tickers 2.0
```

### 3. Limited Processing

```bash
# Process only first 100 tickers
python utils/update_universe_data.py --provider alpaca --timeframe 1d --max-tickers 100

# Dry run to see what would be processed
python utils/update_universe_data.py --provider alpaca --timeframe 1d --dry-run
```

### 4. Resume Interrupted Updates

```bash
# Resume from index 150
python utils/update_universe_data.py --provider alpaca --timeframe 1d --resume-from 150

# Process specific range (index 100 to 199)
python utils/update_universe_data.py --provider alpaca --timeframe 1d --start-index 100 --max-tickers 100
```

### 5. Custom Universe Files

```bash
# Load tickers from a custom file
python utils/update_universe_data.py --provider ib --timeframe 1m --max-bars --universe-file spx.txt
```

### 6. Date-Based Fetching

```bash
# Fetch data since a specific date
python utils/update_universe_data.py --provider ib --timeframe 1h --since 2020-01-01

# Fetch data since a specific date and time
python utils/update_universe_data.py --provider ib --timeframe 1m --since 2020-01-01T09:30:00
```

### 7. Retry Configuration (IB Provider)

```bash
# Use custom retry settings for IB API timeouts
python utils/update_universe_data.py --provider ib --timeframe 1d --max-retries 5 --retry-base-delay 3.0

# Aggressive retry settings for unstable connections
python utils/update_universe_data.py --provider ib --timeframe 1h --max-retries 10 --retry-base-delay 1.0
```

### 8. Testing and Debugging

```bash
# Test a single symbol to debug issues
python utils/update_universe_data.py --provider ib --timeframe 1d --test-symbol STE

# Enable enhanced IB debugging to see permission errors
python utils/update_universe_data.py --provider ib --timeframe 1h --debug-ib --test-symbol STE

# View symbol mappings (IB provider)
python utils/update_universe_data.py --provider ib --view-mappings

# Show symbol mapping statistics
python utils/update_universe_data.py --provider ib --mapping-stats

# Clear invalid symbol mappings
python utils/update_universe_data.py --provider ib --clear-invalid-mappings
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | Data provider: `alpaca` or `ib` | `alpaca` |
| `--timeframe` | Timeframe: `1m`, `15m`, `1h`, or `1d` | `1d` |
| `--batch-size` | Tickers per batch | `10` |
| `--delay-batches` | Seconds between batches | `5.0` |
| `--delay-tickers` | Seconds between tickers | `1.0` |
| `--max-tickers` | Maximum tickers to process | All |
| `--start-index` | Index to start processing from (0-based) | `0` |
| `--resume-from` | Resume processing from this index (0-based) | None |
| `--force-refresh` | Force refresh ticker universe | False |
| `--max-bars` | Fetch maximum available bars instead of fixed amounts | False |
| `--skip-existing` | Skip tickers that already have data for the timeframe | False |
| `--dry-run` | Show what would be processed without fetching data | False |
| `--universe-file` | Load tickers from custom universe file | None |
| `--max-retries` | Maximum retries for timeout errors | `3` |
| `--retry-base-delay` | Base delay (seconds) for exponential backoff retries | `2.0` |
| `--since` | Start date for fetching data (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) | None |
| `--test-symbol` | Test a single symbol for debugging | None |
| `--debug-ib` | Enable enhanced IB logging for debugging | False |
| `--view-mappings` | View symbol mappings and exit (IB provider) | False |
| `--mapping-stats` | Show symbol mapping statistics and exit (IB provider) | False |
| `--clear-invalid-mappings` | Clear all invalid symbol mappings and exit (IB provider) | False |

## Python API Usage

### Basic Usage

```python
from utils.update_universe_data import UniverseDataUpdater

# Initialize updater
updater = UniverseDataUpdater(provider="alpaca", timeframe="1d")

# Update all tickers
results = updater.update_universe_data(
    batch_size=20,
    delay_between_batches=10.0,
    delay_between_tickers=2.0
)

print(f"Processed {results['total_tickers']} tickers")
print(f"Success rate: {results['success_rate']:.1f}%")
```

### Advanced Usage

```python
# Resume from a specific point
results = updater.resume_update(
    last_processed_index=150,
    batch_size=15,
    max_tickers=50
)

# Get universe tickers first
tickers = updater.get_universe_tickers(force_refresh=True)
print(f"Universe contains {len(tickers)} tickers")

# Process specific ticker
success = updater.fetch_ticker_data("AAPL", use_max_bars=True)

# Update with custom universe file
results = updater.update_universe_data(
    universe_file="custom_tickers.txt",
    use_max_bars=True,
    skip_existing=True
)

# Update with start date
results = updater.update_universe_data(
    start_date="2020-01-01",
    use_max_bars=True
)

# View symbol mappings (IB provider only)
if updater.provider == "ib":
    mappings = updater.view_symbol_mappings(limit=100)
    for mapping in mappings:
        print(f"{mapping['original_symbol']} -> {mapping['ib_symbol']}")
    
    # Get mapping statistics
    stats = updater.get_symbol_mapping_stats()
    print(f"Total mappings: {stats['total_mappings']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
```

## Configuration

### Rate Limiting

Adjust delays based on your API provider's rate limits:

- **Alpaca**: Default delays work well for most use cases
- **IBKR**: May need longer delays depending on your connection

### Batch Sizes

- **Small batches (5-10)**: Good for testing and avoiding rate limits
- **Medium batches (15-25)**: Balanced approach for production use
- **Large batches (50+)**: Faster processing but higher risk of rate limiting

### Delays

- **Between tickers**: 0.5-2.0 seconds (prevents overwhelming individual requests)
- **Between batches**: 5-15 seconds (allows API to recover)

### Retry Configuration (IB Provider)

The script includes automatic retry logic with exponential backoff for IB API timeout errors:

- **Default retries**: 3 attempts with 2.0s base delay
- **Exponential backoff**: Delay = base_delay * (2^attempt) + (attempt * 0.5)
- **Customizable**: Use `--max-retries` and `--retry-base-delay` to adjust

For unstable connections or frequent timeouts, increase retries:
```bash
python utils/update_universe_data.py --provider ib --timeframe 1h --max-retries 5 --retry-base-delay 3.0
```

## Output and Logging

### Log Files

- **Console output**: Real-time progress and statistics
- **Log file**: `logs/universe_update.log` with detailed information
- **Complete log**: `logs/universe_update_complete.log` with all logs
- **Summary file**: `logs/universe_update_summary_YYYYMMDD_HHMMSS.txt`

### Progress Tracking

The script provides real-time updates:

```
[PROCESSING] AAPL (1/500, overall index: 0)
[SUCCESS] SUCCESSFULLY fetched 252 bars for AAPL
[PROCESSING] MSFT (2/500, overall index: 1)
[SUCCESS] SUCCESSFULLY fetched 252 bars for MSFT
[BATCH] Completed batch 1. Taking 5.0s break...
[PROGRESS] Progress: 10/500 tickers processed
[STATS] Success: 10, Failed: 0
```

### Results Summary

After completion, a summary file is created with:

- Total tickers processed
- Success/failure counts and rates
- Processing duration
- Failed symbols list
- Skipped symbols list
- Resume information
- Database operation statistics
- Time breakdown (data fetching vs database operations)

## Error Handling

### Network Failures

- Individual ticker failures are logged and tracked
- Processing continues with remaining tickers
- Failed symbols are listed in the summary
- Automatic retry with exponential backoff for timeout errors (IB provider)

### Rate Limiting

- Built-in delays prevent most rate limit issues
- Automatic retry logic with exponential backoff
- Graceful handling of provider-specific errors
- Additional delays after timeout errors to help with rate limiting

### Interruptions

- Use `Ctrl+C` to safely interrupt processing
- Resume from the last processed index using `--resume-from` or `--start-index`
- Progress is saved and can be resumed
- Database operations complete before script exits

### IB-Specific Issues

- **Symbol not found**: Automatically tries symbol variations (e.g., BF.B, BF B, BF-B)
- **No market data permissions**: Check US Value Bundle subscription in IB Client Portal
- **Contract ambiguity**: Automatically resolves using primary exchange
- **Timeout errors**: Automatic retry with exponential backoff
- Use `--debug-ib` flag to see detailed IB error messages

## Best Practices

### 1. Start Small

```bash
# Test with a small batch first
python utils/update_universe_data.py --provider alpaca --timeframe 1d --max-tickers 10
```

### 2. Monitor Progress

```bash
# Watch the log file
tail -f universe_update.log

# Check progress in real-time
python utils/update_universe_data.py --provider alpaca --timeframe 1d --dry-run
```

### 3. Resume Strategy

```bash
# If interrupted, resume from the last processed index
python utils/update_universe_data.py --provider alpaca --timeframe 1d --resume-from 150
```

### 4. Batch Optimization

```bash
# For production use, optimize batch size and delays
python utils/update_universe_data.py \
  --provider alpaca \
  --timeframe 1d \
  --batch-size 25 \
  --delay-batches 8.0 \
  --delay-tickers 1.5
```

## Integration with Existing Systems

### Ticker Universe

- Automatically uses the ticker universe management system
- Supports force refresh of ticker lists
- Handles both S&P 500 and NASDAQ-100 indices
- Supports custom universe files via `--universe-file`

### Fetch Data

- Integrates with existing `fetch_data.py` functionality
- Uses the same data providers and timeframes
- Maintains consistent data format and storage
- Supports date-based fetching with `--since` parameter

### Database

- Data is automatically saved to TimescaleDB
- Uses existing database schema and connections
- Maintains data consistency and integrity
- Non-blocking database saves via background worker thread
- Automatic connection management and reconnection on failures

### Symbol Mapping (IB Provider)

- Automatic symbol conversion for IBKR compatibility
- Caches symbol mappings in database (`symbol_mappings` table)
- Tries multiple symbol variations (e.g., BF.B, BF B, BF-B)
- Stores contract IDs and primary exchanges for faster future lookups
- View and manage mappings via `--view-mappings`, `--mapping-stats`, `--clear-invalid-mappings`

## Troubleshooting

### Common Issues

1. **No tickers available**: Run ticker universe refresh first
2. **Rate limiting**: Increase delays between requests
3. **Database errors**: Check TimescaleDB connection
4. **Memory issues**: Reduce batch size

### Debug Mode

```bash
# Enable verbose logging
python -u utils/update_universe_data.py --provider alpaca --timeframe 1d --max-tickers 5

# Test a single symbol to debug issues (IB provider)
python utils/update_universe_data.py --provider ib --timeframe 1d --test-symbol STE

# Enable enhanced IB debugging to see permission and contract errors
python utils/update_universe_data.py --provider ib --timeframe 1h --debug-ib --test-symbol STE

# View symbol mappings to see how symbols are converted
python utils/update_universe_data.py --provider ib --view-mappings

# Check symbol mapping statistics
python utils/update_universe_data.py --provider ib --mapping-stats
```

### Support

- Check the log files for detailed error information
- Verify your API credentials and rate limits
- Ensure the ticker universe is properly populated
- Test with small batches before large updates

## Examples

See `examples/update_universe_example.py` for complete usage examples and demonstrations of all features.

### Complete Example Workflows

```bash
# 1. Fetch SPX 1-minute data since 2020-01-01 (using fetch_data.py directly)
python utils/fetch_data.py --symbol SPX --provider ib --timeframe 1m --bars max --since 2020-01-01

# 2. Update all tickers with IB hourly data, skip existing, fetch max bars
python utils/update_universe_data.py --provider ib --timeframe 1h --max-bars --skip-existing

# 3. Process specific range with custom retry settings
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1h \
  --start-index 200 \
  --max-tickers 50 \
  --max-bars \
  --skip-existing \
  --max-retries 5 \
  --retry-base-delay 3.0

# 4. Fetch data from custom universe file since a specific date
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1m \
  --max-bars \
  --universe-file spx.txt \
  --since 2020-01-01

# 5. Test and debug a problematic symbol
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1d \
  --debug-ib \
  --test-symbol STE

# 6. View and manage symbol mappings
python utils/update_universe_data.py --provider ib --view-mappings
python utils/update_universe_data.py --provider ib --mapping-stats
python utils/update_universe_data.py --provider ib --clear-invalid-mappings
```

