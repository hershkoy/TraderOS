-- Options Summary Dashboard
-- This file creates a simple SQL view for monitoring and plotting options data
-- Provides per-day counts and statistics for data quality monitoring

-- Create a daily summary view with key metrics
CREATE OR REPLACE VIEW options_summary_daily AS
SELECT 
    DATE(ts) as trade_date,
    COUNT(DISTINCT option_id) as contracts_with_quotes,
    COUNT(*) as total_quotes,
    ROUND(AVG(bid)::NUMERIC, 2) as avg_bid,
    ROUND(AVG(ask)::NUMERIC, 2) as avg_ask,
    ROUND(AVG(ask - bid)::NUMERIC, 2) as avg_spread,
    ROUND(AVG((ask - bid) / NULLIF((bid + ask) / 2, 0) * 100)::NUMERIC, 2) as avg_spread_pct,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY open_interest)::NUMERIC, 0) as median_oi,
    ROUND(AVG(open_interest)::NUMERIC, 0) as avg_oi,
    ROUND(AVG(volume)::NUMERIC, 0) as avg_volume,
    COUNT(CASE WHEN delta IS NOT NULL THEN 1 END) as quotes_with_delta,
    ROUND(COUNT(CASE WHEN delta IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2) as delta_coverage_pct,
    COUNT(CASE WHEN iv IS NOT NULL THEN 1 END) as quotes_with_iv,
    ROUND(COUNT(CASE WHEN iv IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2) as iv_coverage_pct,
    COUNT(CASE WHEN option_right = 'C' THEN 1 END) as call_quotes,
    COUNT(CASE WHEN option_right = 'P' THEN 1 END) as put_quotes,
    ROUND(AVG(CASE WHEN option_right = 'C' THEN (ask - bid) / NULLIF((bid + ask) / 2, 0) * 100 END)::NUMERIC, 2) as call_avg_spread_pct,
    ROUND(AVG(CASE WHEN option_right = 'P' THEN (ask - bid) / NULLIF((bid + ask) / 2, 0) * 100 END)::NUMERIC, 2) as put_avg_spread_pct
FROM option_chain_eod
GROUP BY DATE(ts)
ORDER BY trade_date DESC;

-- Create a view for LEAPS-specific metrics
CREATE OR REPLACE VIEW leaps_summary_daily AS
SELECT 
    DATE(ts) as trade_date,
    COUNT(DISTINCT option_id) as leaps_contracts,
    COUNT(*) as leaps_quotes,
    ROUND(AVG(delta)::NUMERIC, 3) as avg_leaps_delta,
    ROUND(AVG(iv)::NUMERIC, 3) as avg_leaps_iv,
    ROUND(AVG((ask - bid) / NULLIF((bid + ask) / 2, 0) * 100)::NUMERIC, 2) as avg_leaps_spread_pct,
    ROUND(AVG(open_interest)::NUMERIC, 0) as avg_leaps_oi,
    ROUND(AVG(volume)::NUMERIC, 0) as avg_leaps_volume,
    MIN(expiration - DATE(ts)) as min_dte,
    MAX(expiration - DATE(ts)) as max_dte,
    ROUND(AVG(expiration - DATE(ts))::NUMERIC, 0) as avg_dte
FROM option_chain_eod
WHERE expiration - DATE(ts) >= 365 
  AND option_right = 'C'
GROUP BY DATE(ts)
ORDER BY trade_date DESC;

-- Create a view for short-term options metrics
CREATE OR REPLACE VIEW short_term_summary_daily AS
SELECT 
    DATE(ts) as trade_date,
    COUNT(DISTINCT option_id) as short_term_contracts,
    COUNT(*) as short_term_quotes,
    ROUND(AVG(delta)::NUMERIC, 3) as avg_short_delta,
    ROUND(AVG(iv)::NUMERIC, 3) as avg_short_iv,
    ROUND(AVG((ask - bid) / NULLIF((bid + ask) / 2, 0) * 100)::NUMERIC, 2) as avg_short_spread_pct,
    ROUND(AVG(open_interest)::NUMERIC, 0) as avg_short_oi,
    ROUND(AVG(volume)::NUMERIC, 0) as avg_short_volume,
    MIN(expiration - DATE(ts)) as min_dte,
    MAX(expiration - DATE(ts)) as max_dte,
    ROUND(AVG(expiration - DATE(ts))::NUMERIC, 0) as avg_dte
FROM option_chain_eod
WHERE (expiration - DATE(ts)) >= 25 AND (expiration - DATE(ts)) <= 45 
  AND option_right = 'C'
GROUP BY DATE(ts)
ORDER BY trade_date DESC;

-- Create a view for PMCC strategy candidates
CREATE OR REPLACE VIEW pmcc_candidates_summary_daily AS
SELECT 
    DATE(ts) as trade_date,
    COUNT(DISTINCT CASE WHEN expiration - DATE(ts) >= 365 THEN option_id END) as leaps_candidates,
    COUNT(DISTINCT CASE WHEN (expiration - DATE(ts)) >= 25 AND (expiration - DATE(ts)) <= 45 THEN option_id END) as short_call_candidates,
    ROUND(AVG(CASE WHEN expiration - DATE(ts) >= 365 THEN delta END)::NUMERIC, 3) as avg_leaps_delta,
    ROUND(AVG(CASE WHEN (expiration - DATE(ts)) >= 25 AND (expiration - DATE(ts)) <= 45 THEN delta END)::NUMERIC, 3) as avg_short_delta,
    ROUND(AVG(CASE WHEN expiration - DATE(ts) >= 365 THEN (ask - bid) / NULLIF((bid + ask) / 2, 0) * 100 END)::NUMERIC, 2) as avg_leaps_spread_pct,
    ROUND(AVG(CASE WHEN (expiration - DATE(ts)) >= 25 AND (expiration - DATE(ts)) <= 45 THEN (ask - bid) / NULLIF((bid + ask) / 2, 0) * 100 END)::NUMERIC, 2) as avg_short_spread_pct
FROM option_chain_eod
WHERE option_right = 'C'
  AND (
    (expiration - DATE(ts) >= 365 AND delta BETWEEN 0.6 AND 0.85) OR
    ((expiration - DATE(ts)) >= 25 AND (expiration - DATE(ts)) <= 45 AND delta BETWEEN 0.15 AND 0.35)
  )
GROUP BY DATE(ts)
ORDER BY trade_date DESC;

-- Example queries for monitoring:
-- SELECT * FROM options_summary_daily WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days';
-- SELECT * FROM leaps_summary_daily WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days';
-- SELECT * FROM short_term_summary_daily WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days';
-- SELECT * FROM pmcc_candidates_summary_daily WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days';
