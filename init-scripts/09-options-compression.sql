-- Options Compression Policy
-- This file enables TimescaleDB compression on option_quotes older than 90 days
-- Compression helps reduce storage requirements for historical data

-- Enable compression on option_quotes table
-- This sets up the table to be compressible with option_id as the segment by column
ALTER TABLE option_quotes SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'option_id'
);

-- Add compression policy to compress chunks older than 90 days
-- This will automatically compress data that is older than 90 days
SELECT add_compression_policy('option_quotes', INTERVAL '90 days');

-- Verify the compression policy was created
-- You can check with: SELECT * FROM timescaledb_information.compression_settings;
-- And: SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression';

-- Note: Compression will be applied automatically by the background job
-- You can manually compress existing chunks with:
-- SELECT compress_chunk(chunk_name) FROM timescaledb_information.chunks 
-- WHERE hypertable_name = 'option_quotes' AND NOT is_compressed;
