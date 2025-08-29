# Options Pipeline Documentation

This document provides comprehensive documentation for the options data pipeline used in the LEAPS strategy implementation.

## Table Schemas & Indexes

### Core Tables

#### `option_contracts`
Static metadata table for option contracts.

```sql
CREATE TABLE option_contracts (
  option_id       TEXT PRIMARY KEY,              -- e.g. QQQ_2025-06-20_035000C
  underlying      TEXT NOT NULL,                 -- 'QQQ'
  expiration      DATE NOT NULL,
  strike_cents    INT  NOT NULL,                 -- store as int to avoid fp rounding
  option_right    CHAR(1) NOT NULL,              -- 'C' or 'P'
  multiplier      INT  NOT NULL DEFAULT 100,
  first_seen      TIMESTAMPTZ,
  last_seen       TIMESTAMPTZ
);
```

#### `option_quotes`
Time-series data for option quotes and Greeks.

```sql
CREATE TABLE option_quotes (
  ts              TIMESTAMPTZ NOT NULL,
  option_id       TEXT NOT NULL,
  bid             NUMERIC(18,6),
  ask             NUMERIC(18,6),
  last            NUMERIC(18,6),
  volume          BIGINT,
  open_interest   BIGINT,
  iv              NUMERIC(12,6),                 -- Implied Volatility
  delta           NUMERIC(12,6),
  gamma           NUMERIC(12,6),
  theta           NUMERIC(12,6),
  vega            NUMERIC(12,6),
  snapshot_type   TEXT DEFAULT 'eod',            -- 'eod','1545','intraday_agg'...
  PRIMARY KEY (option_id, ts)
);
```

### Indexes

```sql
-- Performance indexes
CREATE INDEX ON option_contracts (underlying, expiration);
CREATE INDEX ON option_quotes (option_id, ts DESC);

-- Hypertable for time-series optimization
SELECT create_hypertable('option_quotes','ts', chunk_time_interval => INTERVAL '7 days');
```

### Views

#### `option_chain_eod`
Latest EOD quotes for each contract with calculated fields.

#### `option_chain_with_underlying`
Joins options data with underlying market data, including moneyness and days to expiration.

#### `pmcc_candidates`
Filters options for PMCC strategy criteria (LEAPS and short calls).

## Command Snippets to Run Discover/Ingest

### 1. Discover Contracts

```bash
# Discover QQQ contracts for next 365 days
python scripts/polygon_discover_contracts.py --underlying QQQ --days-ahead 365

# Backfill historical contracts (last 2 years)
python scripts/polygon_backfill_contracts.py --underlying QQQ --start-date 2022-01-01
```

### 2. Ingest EOD Quotes

```bash
# Ingest quotes for a specific date
python scripts/polygon_ingest_eod_quotes.py --date 2024-01-15

# Ingest quotes for a date range
python scripts/polygon_ingest_eod_quotes.py --start-date 2024-01-01 --end-date 2024-01-31

# Ingest quotes for all active contracts
python scripts/polygon_ingest_eod_quotes.py --all-active
```

### 3. Data Integrity Checks

```bash
# Run daily data integrity checks
python scripts/options_data_checks.py

# Check specific date
python scripts/options_data_checks.py --date 2024-01-15
```

### 4. Database Operations

```bash
# Apply all SQL migrations
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader < init-scripts/03-options-schema.sql
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader < init-scripts/04-options-upsert-functions.sql
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader < init-scripts/05-options-views.sql
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader < init-scripts/06-options-continuous-aggregates.sql
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader < init-scripts/07-options-retention-policies.sql
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader < init-scripts/08-options-join-helpers.sql

# Check retention policies
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader -c "SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention';"
```

## Rate Limits: Polygon Free vs $29 Plan

### Free Plan Limitations
- **Rate Limit**: ~5 requests per minute
- **Historical Data**: Limited to 2 years back
- **Greeks**: Not available
- **Intraday Data**: Not available
- **Real-time Data**: Not available

### $29 Plan Benefits
- **Rate Limit**: ~5 requests per second
- **Historical Data**: Full history available
- **Greeks**: Available (IV, Delta, Gamma, Theta, Vega)
- **Intraday Data**: Available
- **Real-time Data**: Available

### Rate Limiting Implementation

The pipeline implements automatic rate limiting:

```python
# In utils/polygon_client.py
class PolygonClient:
    def __init__(self, api_key: str):
        self.rate_limit_delay = 12  # seconds for free plan (5 req/min)
        # For $29 plan: self.rate_limit_delay = 0.2  # seconds (5 req/sec)
```

### Best Practices for Free Plan

1. **Batch Operations**: Process multiple contracts in single requests
2. **Caching**: Cache contract discovery results
3. **Scheduling**: Run during off-peak hours
4. **Error Handling**: Implement exponential backoff for rate limit errors

## Backtest Selector Examples

### Delta vs Moneyness Fallback

The PMCC strategy uses delta as the primary selection criteria, with moneyness as a fallback when Greeks are not available.

#### LEAPS Selection

```sql
-- Primary: Delta-based selection
SELECT * FROM option_chain_with_underlying 
WHERE expiration - date(ts) >= 365 
  AND option_right = 'C' 
  AND delta BETWEEN 0.6 AND 0.85;

-- Fallback: Moneyness-based selection (when delta is NULL)
SELECT * FROM option_chain_with_underlying 
WHERE expiration - date(ts) >= 365 
  AND option_right = 'C' 
  AND delta IS NULL
  AND moneyness BETWEEN 0.9 AND 1.1;
```

#### Short Call Selection

```sql
-- Primary: Delta-based selection
SELECT * FROM option_chain_with_underlying 
WHERE 25 <= (expiration - date(ts)) <= 45 
  AND option_right = 'C' 
  AND delta BETWEEN 0.15 AND 0.35;

-- Fallback: Moneyness-based selection
SELECT * FROM option_chain_with_underlying 
WHERE 25 <= (expiration - date(ts)) <= 45 
  AND option_right = 'C' 
  AND delta IS NULL
  AND moneyness BETWEEN 1.02 AND 1.08;
```

### Python Implementation

```python
# Using the PMCC provider
from strategies.pmcc_provider import PMCCProvider

provider = PMCCProvider(db_config, underlying='QQQ')

# Get candidates for a specific date
candidates = provider.get_pmcc_candidates(datetime(2024, 1, 15))

# Access the selected options
leaps = candidates['leaps']
short_call = candidates['short_call']
metrics = candidates['strategy_metrics']

print(f"LEAPS: {leaps['option_id']} - Delta: {leaps['delta']}")
print(f"Short Call: {short_call['option_id']} - Delta: {short_call['delta']}")
print(f"Max Profit: ${metrics['max_profit']:.2f}")
```

### Strategy Metrics Calculation

```python
# Example strategy metrics
{
    'net_debit': 1307.50,           # Cost to enter position
    'max_profit': 1692.50,          # Maximum potential profit
    'max_loss': 1307.50,            # Maximum potential loss
    'breakeven': 363.08,            # Underlying price to break even
    'probability_of_profit': 0.25,  # Rough probability estimate
    'leaps_cost': 1562.50,          # Cost of LEAPS position
    'short_call_credit': 255.00,    # Credit from short call
    'leaps_strike': 350.00,         # LEAPS strike price
    'short_strike': 380.00          # Short call strike price
}
```

## Data Quality & Monitoring

### Key Metrics to Monitor

1. **Contract Discovery**: Number of new contracts found daily
2. **Quote Coverage**: Percentage of active contracts with quotes
3. **Data Freshness**: Time lag between market close and data availability
4. **Greeks Availability**: Percentage of quotes with calculated Greeks

### Alert Thresholds

- Contract discovery drops >20% from previous day
- Quote coverage <90% for active contracts
- Data freshness >2 hours after market close
- Greeks availability <80% for ITM options

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: Implement exponential backoff
2. **Missing Greeks**: Use moneyness fallback or calculate manually
3. **Data Gaps**: Check for market holidays or technical issues
4. **Performance Issues**: Verify indexes and retention policies

### Debug Commands

```bash
# Check data freshness
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader -c "
SELECT DATE(ts) as date, COUNT(*) as quotes_count 
FROM option_quotes 
WHERE ts >= NOW() - INTERVAL '7 days' 
GROUP BY DATE(ts) 
ORDER BY date DESC;"

# Check contract coverage
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader -c "
SELECT underlying, COUNT(*) as contracts 
FROM option_contracts 
GROUP BY underlying;"

# Check Greeks availability
docker exec -i backtrader_timescaledb psql -U backtrader_user -d backtrader -c "
SELECT DATE(ts) as date, 
       COUNT(*) as total_quotes,
       COUNT(delta) as quotes_with_delta,
       ROUND(COUNT(delta) * 100.0 / COUNT(*), 2) as delta_coverage_pct
FROM option_quotes 
WHERE ts >= NOW() - INTERVAL '7 days' 
GROUP BY DATE(ts) 
ORDER BY date DESC;"
```

## Performance Optimization

### Indexing Strategy

- Primary indexes on `(option_id, ts)` for time-series queries
- Secondary indexes on `(underlying, expiration)` for contract discovery
- Partial indexes for strategy-specific queries

### Retention & Compression

- Raw quotes retained for 120 days
- Continuous aggregates for historical analysis
- Compression enabled after 90 days for storage efficiency

### Query Optimization

- Use time-bucketed queries for large date ranges
- Leverage continuous aggregates for daily/weekly/monthly views
- Implement proper connection pooling for high-frequency operations
