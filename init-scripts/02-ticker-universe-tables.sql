-- Migration script to add ticker universe tables
-- This script should be run after the main database initialization

-- Create ticker_universe table for storing ticker information
CREATE TABLE IF NOT EXISTS ticker_universe (
    index_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    last_updated TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (index_name, symbol)
);

-- Create ticker_cache_metadata table for tracking cache status
CREATE TABLE IF NOT EXISTS ticker_cache_metadata (
    index_name VARCHAR(50) PRIMARY KEY,
    last_fetched TIMESTAMPTZ NOT NULL,
    ticker_count INTEGER NOT NULL,
    cache_expiry_hours INTEGER DEFAULT 24
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ticker_universe_symbol 
ON ticker_universe (symbol);

CREATE INDEX IF NOT EXISTS idx_ticker_universe_last_updated 
ON ticker_universe (last_updated);

CREATE INDEX IF NOT EXISTS idx_ticker_universe_active 
ON ticker_universe (is_active) WHERE is_active = TRUE;

-- Create a view for easier ticker universe access
CREATE OR REPLACE VIEW ticker_universe_view AS
SELECT 
    index_name,
    symbol,
    company_name,
    sector,
    last_updated,
    is_active
FROM ticker_universe
WHERE is_active = TRUE
ORDER BY index_name, symbol;

-- Create a function to get active tickers by index
CREATE OR REPLACE FUNCTION get_active_tickers(p_index_name VARCHAR(50))
RETURNS TABLE (
    symbol VARCHAR(20),
    company_name VARCHAR(255),
    sector VARCHAR(100),
    last_updated TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tu.symbol,
        tu.company_name,
        tu.sector,
        tu.last_updated
    FROM ticker_universe tu
    WHERE tu.index_name = p_index_name 
      AND tu.is_active = TRUE
    ORDER BY tu.symbol;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get ticker information
CREATE OR REPLACE FUNCTION get_ticker_info(p_symbol VARCHAR(20))
RETURNS TABLE (
    index_name VARCHAR(50),
    symbol VARCHAR(20),
    company_name VARCHAR(255),
    sector VARCHAR(100),
    last_updated TIMESTAMPTZ,
    is_active BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tu.index_name,
        tu.symbol,
        tu.company_name,
        tu.sector,
        tu.last_updated,
        tu.is_active
    FROM ticker_universe tu
    WHERE tu.symbol = p_symbol
    ORDER BY tu.last_updated DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get universe statistics
CREATE OR REPLACE FUNCTION get_universe_stats()
RETURNS TABLE (
    index_name VARCHAR(50),
    ticker_count BIGINT,
    last_fetched TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tcm.index_name,
        tcm.ticker_count,
        tcm.last_fetched
    FROM ticker_cache_metadata tcm
    ORDER BY tcm.index_name;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions to backtrader_user
GRANT ALL PRIVILEGES ON TABLE ticker_universe TO backtrader_user;
GRANT ALL PRIVILEGES ON TABLE ticker_cache_metadata TO backtrader_user;
GRANT ALL PRIVILEGES ON TABLE ticker_universe_view TO backtrader_user;
GRANT EXECUTE ON FUNCTION get_active_tickers(VARCHAR) TO backtrader_user;
GRANT EXECUTE ON FUNCTION get_ticker_info(VARCHAR) TO backtrader_user;
GRANT EXECUTE ON FUNCTION get_universe_stats() TO backtrader_user;

-- Insert initial cache metadata (empty)
INSERT INTO ticker_cache_metadata (index_name, last_fetched, ticker_count, cache_expiry_hours)
VALUES 
    ('sp500', NOW(), 0, 24),
    ('nasdaq100', NOW(), 0, 24)
ON CONFLICT (index_name) DO NOTHING;

-- Log the migration
DO $$
BEGIN
    RAISE NOTICE 'Ticker universe tables migration completed successfully';
    RAISE NOTICE 'Tables created: ticker_universe, ticker_cache_metadata';
    RAISE NOTICE 'Views created: ticker_universe_view';
    RAISE NOTICE 'Functions created: get_active_tickers, get_ticker_info, get_universe_stats';
END $$;
