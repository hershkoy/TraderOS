# TimescaleDB Migration Guide

This document outlines the migration from Parquet file storage to TimescaleDB for the BackTrader framework.

## Overview

The framework has been updated to use TimescaleDB (a PostgreSQL extension optimized for time-series data) instead of Parquet files for data storage. This provides better performance, data integrity, and scalability.

## What Changed

### 1. New Infrastructure

**Docker Setup:**
- `docker-compose.yml` - TimescaleDB and pgAdmin containers
- `init-scripts/01-init-database.sql` - Database schema and functions
- `setup_timescaledb.py` - Automated setup script

**Database Schema:**
- `market_data` table with hypertable for time-series optimization
- Indexes on symbol, provider, and timeframe
- Automatic data deduplication and conflict resolution
- Helper functions for data insertion and retrieval

### 2. New Python Modules

**TimescaleDB Client (`utils/timescaledb_client.py`):**
- Database connection management
- Data insertion and retrieval functions
- Summary statistics and data management
- Connection pooling and error handling

**TimescaleDB Loader (`utils/timescaledb_loader.py`):**
- Replaces Parquet file loading functions
- Backward compatibility with existing parquet paths
- Data loading for BackTrader integration
- Utility functions for data discovery

### 3. Updated Scripts

**Fetch Data Script (`utils/fetch_data.py`):**
- Now saves data to TimescaleDB instead of Parquet files
- Maintains same API and functionality
- Better error handling and logging

**Backtrader Runner (`backtrader_runner_yaml.py`):**
- New command-line parameters: `--symbol`, `--provider`, `--timeframe`
- Backward compatibility with `--parquet` parameter
- Automatic data loading from TimescaleDB

### 4. New Dependencies

**Added to `requirements.txt`:**
- `psycopg2-binary==2.9.9` - PostgreSQL driver for Python

## Migration Steps

### 1. Setup TimescaleDB

```bash
# Run automated setup
python setup/setup_timescaledb.py

# Or manually
docker-compose up -d
```

### 2. Install New Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test Integration

```bash
# Run integration tests
python test_timescaledb_integration.py
```

### 4. Migrate Existing Data (Optional)

If you have existing Parquet files, you can migrate them:

```bash
# For each parquet file, fetch the data again
python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 1000
```

## New Usage Patterns

### Fetching Data

**Old (Parquet):**
```bash
python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 1000
# Data saved to: data/ALPACA/NFLX/1h/nflx_1h.parquet
```

**New (TimescaleDB):**
```bash
python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 1000
# Data saved to TimescaleDB database
```

### Running Backtests

**Old (Parquet):**
```bash
python backtrader_runner_yaml.py --parquet "data/ALPACA/NFLX/1h/nflx_1h.parquet" --strategy mean_reversion
```

**New (TimescaleDB):**
```bash
python backtrader_runner_yaml.py --symbol NFLX --provider ALPACA --timeframe 1h --strategy mean_reversion
```

**Legacy Support (Still Works):**
```bash
python backtrader_runner_yaml.py --parquet "data/ALPACA/NFLX/1h/nflx_1h.parquet" --strategy mean_reversion
```

## Database Management

### Accessing the Database

**Web Interface (pgAdmin):**
- URL: http://localhost:8080
- Email: admin@backtrader.com
- Password: admin

**Command Line:**
```bash
docker-compose exec timescaledb psql -U backtrader_user -d backtrader
```

### Useful Queries

```sql
-- List all symbols
SELECT DISTINCT symbol FROM market_data ORDER BY symbol;

-- Get data summary
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT symbol) as unique_symbols,
    COUNT(DISTINCT provider) as unique_providers,
    MIN(to_timestamp(ts_event / 1000000000)) as earliest_date,
    MAX(to_timestamp(ts_event / 1000000000)) as latest_date
FROM market_data;

-- Get data for specific symbol
SELECT * FROM get_market_data('NFLX', '1h', 'ALPACA');
```

### Python Utilities

```python
# List available data
from utils.timescaledb_loader import get_available_data
print(get_available_data())

# List symbols
from utils.timescaledb_loader import list_available_symbols
print(list_available_symbols())

# Test connection
from utils.timescaledb_client import test_connection
test_connection()
```

## Benefits of TimescaleDB

### Performance
- **Hypertables**: Automatic partitioning by time for fast queries
- **Indexes**: Optimized for time-series queries
- **Compression**: Automatic data compression for storage efficiency

### Data Integrity
- **ACID Compliance**: Full transaction support
- **Constraints**: Data validation and consistency
- **Backup/Recovery**: Standard PostgreSQL backup tools

### Scalability
- **Horizontal Scaling**: Can be distributed across multiple nodes
- **Large Datasets**: Handles millions of records efficiently
- **Concurrent Access**: Multiple users can access data simultaneously

### Query Flexibility
- **SQL**: Standard SQL with time-series extensions
- **Time Functions**: Built-in time-series analysis functions
- **Aggregations**: Efficient time-based aggregations

## Troubleshooting

### Common Issues

**1. Connection Failed**
```bash
# Check if TimescaleDB is running
docker-compose ps

# Check logs
docker-compose logs timescaledb

# Restart services
docker-compose restart
```

**2. Import Error for psycopg2**
```bash
# Install dependencies
pip install -r requirements.txt

# On Windows, you might need:
pip install psycopg2-binary
```

**3. Permission Denied**
```bash
# Check Docker permissions
docker-compose down
docker-compose up -d
```

### Data Recovery

**Backup Database:**
```bash
docker-compose exec timescaledb pg_dump -U backtrader_user backtrader > backup.sql
```

**Restore Database:**
```bash
docker-compose exec -T timescaledb psql -U backtrader_user backtrader < backup.sql
```

## Backward Compatibility

The migration maintains full backward compatibility:

1. **Parquet Paths**: Old `--parquet` parameter still works
2. **File Structure**: Existing parquet files can still be used
3. **API Compatibility**: All existing scripts continue to work
4. **Data Format**: Same NautilusTrader-compatible format

## Future Enhancements

Potential improvements for future versions:

1. **Real-time Data**: Live data streaming capabilities
2. **Advanced Analytics**: Built-in technical indicators
3. **Data Validation**: Enhanced data quality checks
4. **Performance Monitoring**: Query performance metrics
5. **Multi-tenancy**: Support for multiple users/organizations

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the test script: `test_timescaledb_integration.py`
3. Check Docker logs: `docker-compose logs timescaledb`
4. Verify setup: `python setup_timescaledb.py`
