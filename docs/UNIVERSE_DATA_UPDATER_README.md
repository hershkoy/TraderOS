# Universe Data Updater

A comprehensive script to fetch maximum available bars for all tickers in the ticker universe using the existing `fetch_data.py` functionality. This tool integrates the ticker universe management system with your data fetching pipeline.

## Features

- **Automatic Universe Integration**: Uses the ticker universe system to get all available tickers
- **Batch Processing**: Process tickers in configurable batches with delays
- **Resume Capability**: Resume interrupted updates from any point
- **Multiple Providers**: Support for both Alpaca and IBKR data providers
- **Flexible Timeframes**: Hourly (1h) and daily (1d) data support
- **Rate Limiting**: Built-in delays to respect API rate limits
- **Comprehensive Logging**: Detailed logs with progress tracking
- **Dry Run Mode**: Preview what would be processed without fetching data
- **Progress Tracking**: Monitor success/failure rates and processing statistics

## Quick Start

### 1. Basic Usage



```bash

# normal usage
python utils/update_universe_data.py --provider ib -timeframe 1h --max-bars --skip-existing

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
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | Data provider: `alpaca` or `ib` | `alpaca` |
| `--timeframe` | Timeframe: `1h` or `1d` | `1d` |
| `--batch-size` | Tickers per batch | `10` |
| `--delay-batches` | Seconds between batches | `5.0` |
| `--delay-tickers` | Seconds between tickers | `1.0` |
| `--max-tickers` | Maximum tickers to process | All |
| `--resume-from` | Resume from this index | `0` |
| `--force-refresh` | Force refresh ticker universe | False |
| `--dry-run` | Show what would be processed | False |

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
success = updater.fetch_ticker_data("AAPL")
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

## Output and Logging

### Log Files

- **Console output**: Real-time progress and statistics
- **Log file**: `universe_update.log` with detailed information
- **Summary file**: `universe_update_summary_YYYYMMDD_HHMMSS.txt`

### Progress Tracking

The script provides real-time updates:

```
Processing AAPL (1/500, overall index: 0)
✅ Successfully fetched 252 bars for AAPL
Processing MSFT (2/500, overall index: 1)
✅ Successfully fetched 252 bars for MSFT
Completed batch 1. Taking 5.0s break...
Progress: 10/500 tickers processed
Success: 10, Failed: 0
```

### Results Summary

After completion, a summary file is created with:

- Total tickers processed
- Success/failure counts and rates
- Processing duration
- Failed symbols list
- Resume information

## Error Handling

### Network Failures

- Individual ticker failures are logged and tracked
- Processing continues with remaining tickers
- Failed symbols are listed in the summary

### Rate Limiting

- Built-in delays prevent most rate limit issues
- Automatic retry logic from `fetch_data.py`
- Graceful handling of provider-specific errors

### Interruptions

- Use `Ctrl+C` to safely interrupt processing
- Resume from the last processed index
- Progress is saved and can be resumed

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

### Fetch Data

- Integrates with existing `fetch_data.py` functionality
- Uses the same data providers and timeframes
- Maintains consistent data format and storage

### Database

- Data is automatically saved to TimescaleDB
- Uses existing database schema and connections
- Maintains data consistency and integrity

## Troubleshooting

### Common Issues

1. **No tickers available**: Run ticker universe refresh first
2. **Rate limiting**: Increase delays between requests
3. **Database errors**: Check TimescaleDB connection
4. **Memory issues**: Reduce batch size

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH=.
python -u utils/update_universe_data.py --provider alpaca --timeframe 1d --max-tickers 5
```

### Support

- Check the log files for detailed error information
- Verify your API credentials and rate limits
- Ensure the ticker universe is properly populated
- Test with small batches before large updates

## Examples

See `examples/update_universe_example.py` for complete usage examples and demonstrations of all features.
