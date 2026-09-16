-- Keep historical OptionsDX / research EOD quotes.
-- The original 120-day retention on option_quotes would drop 2010-2023
-- chain history as soon as the job ran. Continuous aggregates are not a
-- substitute for strike-level bid/ask/greeks.

SELECT remove_retention_policy('option_quotes', if_exists => true);

-- Wider chunks: EOD chains span years; 7-day chunks create hundreds of
-- empty-ish pieces during a bulk load. Only affects newly created chunks.
SELECT set_chunk_time_interval('option_quotes', INTERVAL '1 year');
