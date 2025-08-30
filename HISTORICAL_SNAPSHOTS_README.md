# Historical Bid/Ask Snapshots via Polygon API

This document describes the implementation of historical bid/ask data retrieval using Polygon's upgraded API snapshot endpoint.

## Overview

With Polygon's API upgrade, you can now pull **historical bid/ask** data via the snapshot endpoint and wire it into your backfill process. This provides end-of-day pricing data for options contracts as they existed on specific historical dates.

## What Was Implemented

### 1. Enhanced Polygon Client (`utils/polygon_client.py`)

- **Configurable Rate Limits**: Rate limits are now configurable via `POLYGON_REQUESTS_PER_MINUTE` environment variable (default: 60 req/min)
- **Historical Snapshots**: Added `as_of` parameter to `get_options_snapshot()` method to fetch historical data

```python
# Fetch historical snapshot for a specific date
snap = client.get_options_snapshot(
    option_ticker="O:QQQ251219C00550000",
    as_of="2024-07-09"  # YYYY-MM-DD format
)
```

### 2. New Database Table (`init-scripts/04-option-snapshots-schema.sql`)

Created `option_eod_snapshots` table to store historical bid/ask data:

```sql
CREATE TABLE option_eod_snapshots (
  option_id       TEXT NOT NULL,                 -- References option_contracts.option_id
  as_of           DATE NOT NULL,                 -- Date of the snapshot
  bid             NUMERIC(18,6),                 -- Bid price
  ask             NUMERIC(18,6),                 -- Ask price
  mid             NUMERIC(18,6),                 -- Calculated mid price
  last            NUMERIC(18,6),                 -- Last trade price
  volume          BIGINT,                        -- Volume for the day
  open_interest   BIGINT,                        -- Open interest
  created_at      TIMESTAMPTZ DEFAULT NOW(),     -- When record was created
  PRIMARY KEY (option_id, as_of)
);
```

### 3. Enhanced Backfill Script (`scripts/polygon_backfill_contracts.py`)

The backfill process now automatically fetches historical snapshots for each contract discovery date:

- After discovering contracts for a specific date, fetches historical bid/ask data for that same date
- Stores the data in the `option_eod_snapshots` table
- Avoids duplicate API calls for the same ticker
- Handles errors gracefully and continues processing

### 4. New Utility Methods

- `upsert_option_snapshots()`: Stores historical snapshot data with conflict resolution
- Enhanced error handling and logging for snapshot operations

## How It Works

1. **Contract Discovery**: The backfill discovers contracts that existed on a specific historical date
2. **Historical Snapshots**: For each discovered contract, fetches the snapshot as of that same date
3. **Data Storage**: Stores bid/ask, mid price, volume, and open interest in the database
4. **Rate Limiting**: Respects configurable API rate limits to avoid hitting Polygon's limits

## Usage

### 1. Run Database Migration

```bash
python run_snapshot_migration.py
```

### 2. Test Historical Snapshots

```bash
python test_historical_snapshots.py
```

### 3. Run Enhanced Backfill

```bash
python scripts/polygon_backfill_contracts.py --days-back 30 --sample-rate 1
```

### 4. Configure Rate Limits (Optional)

Set environment variable to control API rate limits:

```bash
# Windows CMD
set POLYGON_REQUESTS_PER_MINUTE=120

# Or in your .env file
POLYGON_REQUESTS_PER_MINUTE=120
```

## Data Structure

The historical snapshots provide:

- **Bid/Ask**: Historical bid and ask prices as they existed on the specified date
- **Mid Price**: Calculated as `(bid + ask) / 2`
- **Volume**: Trading volume for that day
- **Open Interest**: Open interest as of that date
- **Last Trade**: Last trade price and size (if available)

## Benefits

1. **Historical Accuracy**: Get actual bid/ask spreads as they existed on specific dates
2. **Backtesting**: Use real historical pricing data for strategy backtesting
3. **Risk Analysis**: Analyze historical bid/ask spreads for risk assessment
4. **Performance**: Efficient storage and retrieval of historical pricing data

## Rate Limiting Considerations

- **Default**: 60 requests per minute (configurable)
- **Historical Snapshots**: Each contract requires one API call per date
- **Sampling**: Use `--sample-rate` parameter to control how many dates to process
- **Monitoring**: Check logs for rate limit warnings and adjust accordingly

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: Increase `POLYGON_REQUESTS_PER_MINUTE` or reduce `--sample-rate`
2. **Missing Data**: Some historical dates may not have snapshot data available
3. **API Errors**: Check Polygon API status and your API key permissions

### Debug Mode

Run with debug logging to see detailed API interactions:

```bash
python scripts/polygon_backfill_contracts.py --log-level DEBUG
```

## Next Steps

1. **Data Validation**: Verify the quality of historical snapshot data
2. **Performance Optimization**: Consider batch processing for large datasets
3. **Analytics**: Build queries to analyze historical bid/ask patterns
4. **Integration**: Wire the historical data into your backtesting strategies

## Files Modified/Created

- `utils/polygon_client.py` - Enhanced with historical snapshot support
- `scripts/polygon_backfill_contracts.py` - Added snapshot fetching and storage
- `init-scripts/04-option-snapshots-schema.sql` - New database table
- `test_historical_snapshots.py` - Test script for verification
- `run_snapshot_migration.py` - Database migration runner
- `HISTORICAL_SNAPSHOTS_README.md` - This documentation

## Support

For issues or questions about the historical snapshot functionality, check the logs and ensure your Polygon API key has the necessary permissions for the snapshot endpoint.
