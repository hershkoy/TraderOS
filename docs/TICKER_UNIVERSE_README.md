# Ticker Universe Management System

A robust ticker universe management system for the BackTrader framework that fetches, caches, and manages ticker lists from various market indices using TimescaleDB.

## Features

- **Automatic Caching**: 24-hour cache expiry with fallback to stale data
- **Multiple Indices**: S&P 500 and NASDAQ-100 support
- **Rich Metadata**: Company names, sectors, and index membership
- **TimescaleDB Integration**: Built on the existing database infrastructure
- **Error Handling**: Graceful fallbacks when data fetching fails
- **Performance Optimized**: Proper indexing and efficient queries

## Quick Start

### 1. Database Setup

Run the migration script to create the required tables:

```sql
-- Run this after the main database initialization
\i init-scripts/02-ticker-universe-tables.sql
```

### 2. Basic Usage

```python
from utils.ticker_universe import TickerUniverseManager

# Initialize manager
manager = TickerUniverseManager()

# Get S&P 500 tickers (fetches from Wikipedia if not cached)
sp500_tickers = manager.get_sp500_tickers()

# Get NASDAQ-100 tickers
nasdaq100_tickers = manager.get_nasdaq100_tickers()

# Get combined universe (deduplicated)
combined_universe = manager.get_combined_universe()

# Force refresh all indices
refresh_results = manager.refresh_all_indices()
```

### 3. Convenience Functions

```python
from utils.ticker_universe import get_sp500_tickers, get_combined_universe

# Simple function calls
sp500 = get_sp500_tickers()
universe = get_combined_universe()
```

## API Reference

### TickerUniverseManager Class

#### Methods

- `get_sp500_tickers(force_refresh=False)`: Get S&P 500 tickers
- `get_nasdaq100_tickers(force_refresh=False)`: Get NASDAQ-100 tickers  
- `get_combined_universe(force_refresh=False)`: Get combined universe
- `get_cached_combined_universe()`: Get universe from cache only
- `get_ticker_info(symbol)`: Get detailed ticker information
- `get_universe_stats()`: Get universe statistics
- `refresh_all_indices()`: Force refresh all indices

#### Properties

- `db_client`: TimescaleDB client instance
- `CACHE_EXPIRY_HOURS`: Cache expiration time (24 hours)

### Database Schema

#### ticker_universe Table
- `index_name`: Index identifier (sp500, nasdaq100)
- `symbol`: Stock symbol
- `company_name`: Company name
- `sector`: Industry sector
- `last_updated`: Last update timestamp
- `is_active`: Whether ticker is currently active

#### ticker_cache_metadata Table
- `index_name`: Index identifier
- `last_fetched`: Last fetch timestamp
- `ticker_count`: Number of tickers in index
- `cache_expiry_hours`: Cache expiration time

## Example Usage

See `examples/ticker_universe_example.py` for a complete demonstration.

## Testing

Run the test suite:

```bash
python -m pytest tests/test_ticker_universe.py -v
```

## Configuration

- **Cache Expiry**: Modify `CACHE_EXPIRY_HOURS` in the module
- **Data Sources**: Update Wikipedia URLs if needed
- **Database**: Configure via TimescaleDB client settings

## Error Handling

The system includes robust error handling:
- Network failures fall back to cached data
- Database errors are logged and handled gracefully
- Invalid data is filtered out during processing

## Performance Considerations

- Uses database indexes for fast queries
- Implements soft deletes (is_active flag)
- Batches database operations
- Caches frequently accessed data

## Dependencies

- `pandas`: Data processing
- `psycopg2`: PostgreSQL/TimescaleDB connection
- `timescaledb_client`: Database client wrapper
