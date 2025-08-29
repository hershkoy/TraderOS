-- Options Performance Indexes
-- This file adds performance indexes to optimize PMCC/LEAPS selector queries
-- These indexes will improve query performance for strategy selection

-- Add partial index on option_contracts for calls only (PMCC strategy uses calls)
-- This index optimizes queries that filter by call options for QQQ
CREATE INDEX IF NOT EXISTS idx_option_contracts_calls_qqq 
ON option_contracts (expiration, strike_cents) 
WHERE option_right = 'C' AND underlying = 'QQQ';

-- Add index for LEAPS selection (expiration >= 365 days from current date)
-- This helps with queries that filter for long-term options
CREATE INDEX IF NOT EXISTS idx_option_contracts_leaps 
ON option_contracts (expiration, underlying, option_right) 
WHERE option_right = 'C';

-- Add index for short-term calls (25-45 DTE range)
-- This optimizes queries for short call selection in PMCC strategy
CREATE INDEX IF NOT EXISTS idx_option_contracts_short_calls 
ON option_contracts (expiration, underlying, option_right) 
WHERE option_right = 'C';

-- Add composite index for option_quotes to optimize joins with option_contracts
-- This helps with queries that join quotes and contracts
CREATE INDEX IF NOT EXISTS idx_option_quotes_ts_option_id 
ON option_quotes (ts DESC, option_id);

-- Add index for underlying and date range queries
-- This optimizes queries that filter by underlying and date ranges
CREATE INDEX IF NOT EXISTS idx_option_quotes_underlying_ts 
ON option_quotes (option_id, ts DESC);

-- Verify indexes were created
-- You can check with: \d+ option_contracts
-- And: \d+ option_quotes
