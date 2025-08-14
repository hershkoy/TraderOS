-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create market data table
CREATE TABLE IF NOT EXISTS market_data (
    ts_event BIGINT NOT NULL,
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

-- Create hypertable for time-series data
SELECT create_hypertable('market_data', 'ts_event', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_provider ON market_data (provider);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data (timeframe);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_market_data_provider_symbol ON market_data (provider, symbol);

-- Create a view for easier data access
CREATE OR REPLACE VIEW market_data_view AS
SELECT 
    to_timestamp(ts_event / 1000000000) as timestamp,
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
ORDER BY ts_event;

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
        to_timestamp(md.ts_event / 1000000000) as timestamp,
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
      AND (p_start_time IS NULL OR to_timestamp(md.ts_event / 1000000000) >= p_start_time)
      AND (p_end_time IS NULL OR to_timestamp(md.ts_event / 1000000000) <= p_end_time)
    ORDER BY md.ts_event;
END;
$$ LANGUAGE plpgsql;

-- Create a function to insert market data
CREATE OR REPLACE FUNCTION insert_market_data(
    p_ts_event BIGINT,
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
        ts_event, symbol, provider, timeframe, 
        open, high, low, close, volume
    ) VALUES (
        p_ts_event, p_symbol, p_provider, p_timeframe,
        p_open, p_high, p_low, p_close, p_volume
    )
    ON CONFLICT (ts_event, symbol, provider, timeframe) 
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
