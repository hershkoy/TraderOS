-- Option EOD Prices schema migration
-- Creates option_eod_prices table for storing historical end-of-day pricing data

-- Historical end-of-day prices for option contracts
CREATE TABLE option_eod_prices (
  option_id       TEXT NOT NULL,                 -- References option_contracts.option_id
  as_of           DATE NOT NULL,                 -- Date of the EOD data
  open            NUMERIC(18,6),                 -- Opening price
  high            NUMERIC(18,6),                 -- High price for the day
  low             NUMERIC(18,6),                 -- Low price for the day
  close           NUMERIC(18,6),                 -- Closing price
  volume          BIGINT,                        -- Trading volume for the day
  vwap            NUMERIC(18,6),                 -- Volume-weighted average price
  transactions    BIGINT,                        -- Number of transactions
  created_at      TIMESTAMPTZ DEFAULT NOW(),     -- When this record was created
  PRIMARY KEY (option_id, as_of)
);

-- Create indexes for performance
CREATE INDEX ON option_eod_prices (option_id, as_of DESC);
CREATE INDEX ON option_eod_prices (as_of);
CREATE INDEX ON option_eod_prices (close);  -- For price-based queries

-- Add foreign key constraint to option_contracts
ALTER TABLE option_eod_prices 
ADD CONSTRAINT fk_option_eod_prices_option_id 
FOREIGN KEY (option_id) REFERENCES option_contracts(option_id) ON DELETE CASCADE;

-- Add comments for documentation
COMMENT ON TABLE option_eod_prices IS 'Historical end-of-day prices for option contracts';
COMMENT ON COLUMN option_eod_prices.as_of IS 'Date of the EOD data (YYYY-MM-DD)';
COMMENT ON COLUMN option_eod_prices.vwap IS 'Volume-weighted average price for the day';
COMMENT ON COLUMN option_eod_prices.transactions IS 'Number of transactions for the day';
COMMENT ON COLUMN option_eod_prices.created_at IS 'Timestamp when this EOD record was created in our system';

