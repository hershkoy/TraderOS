-- Reset TimescaleDB database with proper TIMESTAMPTZ structure
-- This script drops the existing table and recreates it with optimal structure

-- Drop existing table and related objects
DROP TABLE IF EXISTS market_data CASCADE;
DROP VIEW IF EXISTS market_data_view CASCADE;
DROP FUNCTION IF EXISTS get_market_data CASCADE;
DROP FUNCTION IF EXISTS insert_market_data CASCADE;

-- Recreate the table with proper TIMESTAMPTZ structure
CREATE TABLE market_data (
    ts TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    provider VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(15,6) NOT NULL,
    high DECIMAL(15,6) NOT NULL,
    low DECIMAL(15,6) NOT NULL,
    close DECIMAL(15,6) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for time-series data with proper time partitioning
SELECT create_hypertable('market_data', 'ts', chunk_time_interval => INTERVAL '7 days');

-- Create indexes for optimal query performance
-- Primary index for symbol + time range queries (most common)
CREATE INDEX idx_market_data_symbol_ts ON market_data (symbol, ts DESC);

-- Index for provider + time range queries
CREATE INDEX idx_market_data_provider_ts ON market_data (provider, ts DESC);

-- Index for timeframe + time range queries
CREATE INDEX idx_market_data_timeframe_ts ON market_data (timeframe, ts DESC);

-- Composite index for symbol + provider + timeframe queries
CREATE INDEX idx_market_data_symbol_provider_timeframe_ts ON market_data (symbol, provider, timeframe, ts DESC);

-- Create a view for easier data access
CREATE VIEW market_data_view AS
SELECT 
    ts as timestamp,
    symbol,
    provider,
    timeframe,
    open,
    high,
    low,
    close,
    volume,
    created_at
FROM market_data
ORDER BY ts;

-- Create a function to get data for a specific symbol and timeframe
CREATE OR REPLACE FUNCTION get_market_data(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_provider VARCHAR(20) DEFAULT NULL,
    p_start_time TIMESTAMPTZ DEFAULT NULL,
    p_end_time TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    symbol VARCHAR(20),
    provider VARCHAR(20),
    timeframe VARCHAR(10),
    open DECIMAL(15,6),
    high DECIMAL(15,6),
    low DECIMAL(15,6),
    close DECIMAL(15,6),
    volume BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        md.ts as timestamp,
        md.symbol,
        md.provider,
        md.timeframe,
        md.open,
        md.high,
        md.low,
        md.close,
        md.volume
    FROM market_data md
    WHERE md.symbol = p_symbol
      AND md.timeframe = p_timeframe
      AND (p_provider IS NULL OR md.provider = p_provider)
      AND (p_start_time IS NULL OR md.ts >= p_start_time)
      AND (p_end_time IS NULL OR md.ts <= p_end_time)
    ORDER BY md.ts;
END;
$$ LANGUAGE plpgsql;

-- Create a function to insert market data
CREATE OR REPLACE FUNCTION insert_market_data(
    p_ts TIMESTAMPTZ,
    p_symbol VARCHAR(20),
    p_provider VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_open DECIMAL(15,6),
    p_high DECIMAL(15,6),
    p_low DECIMAL(15,6),
    p_close DECIMAL(15,6),
    p_volume BIGINT
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO market_data (
        ts, symbol, provider, timeframe, 
        open, high, low, close, volume
    ) VALUES (
        p_ts, p_symbol, p_provider, p_timeframe,
        p_open, p_high, p_low, p_close, p_volume
    )
    ON CONFLICT (ts, symbol, provider, timeframe) 
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        created_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO backtrader_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO backtrader_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO backtrader_user;

-- Verify the setup
SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_data';
SELECT * FROM timescaledb_information.chunks WHERE hypertable_name = 'market_data';


