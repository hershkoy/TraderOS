-- Options Retention Policies
-- This file implements TimescaleDB retention policies for options data
-- Raw option_quotes are kept for 120 days, daily EOD data is kept indefinitely

-- Add retention policy to keep raw option_quotes for 120 days
-- This is useful if you later add intraday data, but for now keeps EOD data manageable
SELECT add_retention_policy('option_quotes', INTERVAL '120 days');

-- Note: The continuous aggregates (cq_option_eod, cq_option_weekly, cq_option_monthly)
-- will keep their aggregated data indefinitely, providing historical analysis capabilities
-- even after raw data is dropped by the retention policy.

-- Verify the retention policy was created
-- You can check with: SELECT * FROM timescaledb_information.retention_policies;
