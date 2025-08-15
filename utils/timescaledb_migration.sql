-- TimescaleDB Migration Script for market_data table
-- This script converts the table to use proper TIMESTAMPTZ with hypertable setup

-- 1. First, let's check the current table structure
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'market_data' 
ORDER BY ordinal_position;

-- 2. Check if hypertable exists
SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_data';

-- 3. Create a new table with proper TIMESTAMPTZ structure
CREATE TABLE market_data_new (
    ts TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    provider VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume BIGINT NOT NULL,
    ts_event BIGINT -- Keep original for reference if needed
);

-- 4. Migrate data from old table to new table
INSERT INTO market_data_new (ts, symbol, provider, timeframe, open, high, low, close, volume, ts_event)
SELECT 
    to_timestamp(ts_event / 1e9) as ts,
    symbol,
    provider,
    timeframe,
    open,
    high,
    low,
    close,
    volume,
    ts_event
FROM market_data;

-- 5. Create hypertable on the new table
SELECT create_hypertable('market_data_new', 'ts', chunk_time_interval => INTERVAL '7 days');

-- 6. Create indexes for optimal query performance
-- Primary index for symbol + time range queries (most common)
CREATE INDEX ON market_data_new (symbol, ts DESC);

-- Index for provider + time range queries
CREATE INDEX ON market_data_new (provider, ts DESC);

-- Index for timeframe + time range queries
CREATE INDEX ON market_data_new (timeframe, ts DESC);

-- Composite index for symbol + provider + timeframe queries
CREATE INDEX ON market_data_new (symbol, provider, timeframe, ts DESC);

-- 7. Drop the old table and rename the new one
DROP TABLE market_data;
ALTER TABLE market_data_new RENAME TO market_data;

-- 8. Verify the migration
SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_data';

-- 9. Test query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM market_data 
WHERE symbol = 'NFLX' 
ORDER BY ts DESC 
LIMIT 100;

-- 10. Show table statistics
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename = 'market_data'
ORDER BY attname;


