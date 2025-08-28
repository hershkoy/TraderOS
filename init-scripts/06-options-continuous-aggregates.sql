-- Options continuous aggregates for TimescaleDB
-- Provides pre-aggregated views for better query performance

-- Continuous aggregate for daily option quotes
-- Aggregates data to 1-day buckets for efficient querying
CREATE MATERIALIZED VIEW cq_option_eod
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', ts) AS day,
    option_id,
    -- Price aggregates
    last(bid, ts) AS bid,
    last(ask, ts) AS ask,
    last(last, ts) AS last,
    -- Volume and open interest aggregates
    max(open_interest) AS open_interest,
    sum(volume) AS volume,
    -- Greeks (last value of the day)
    last(iv, ts) AS iv,
    last(delta, ts) AS delta,
    last(gamma, ts) AS gamma,
    last(theta, ts) AS theta,
    last(vega, ts) AS vega,
    -- Additional metrics
    count(*) AS quote_count,
    min(ts) AS first_quote_time,
    max(ts) AS last_quote_time
FROM option_quotes
GROUP BY day, option_id;

-- Add refresh policy to update the continuous aggregate daily
-- This will refresh the view every day at 22:00 UTC (after market close)
SELECT add_continuous_aggregate_policy('cq_option_eod',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 day');

-- Add comments for documentation
COMMENT ON MATERIALIZED VIEW cq_option_eod IS 'Daily continuous aggregate of option quotes for efficient querying';
