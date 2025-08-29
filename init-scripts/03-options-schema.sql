-- Options schema migration for LEAPS strategy
-- Creates option_contracts and option_quotes tables with TimescaleDB features

-- One row per option contract (static metadata)
CREATE TABLE option_contracts (
  option_id       TEXT PRIMARY KEY,              -- e.g. OCC-style: QQQ_2025-06-20_000350C
  underlying      TEXT NOT NULL,                 -- 'QQQ'
  expiration      DATE NOT NULL,
  strike_cents    INT  NOT NULL,                 -- store as int to avoid fp rounding
  option_right    CHAR(1) NOT NULL,              -- 'C' or 'P'
  multiplier      INT  NOT NULL DEFAULT 100,
  first_seen      TIMESTAMPTZ,
  last_seen       TIMESTAMPTZ,
  polygon_ticker  TEXT                           -- Polygon's ticker for API calls (e.g., O:QQQ250920C00300000)
);

-- Quotes/"EOD"/mid/greeks across time (hypertable)
CREATE TABLE option_quotes (
  ts              TIMESTAMPTZ NOT NULL,
  option_id       TEXT NOT NULL,
  bid             NUMERIC(18,6),
  ask             NUMERIC(18,6),
  last            NUMERIC(18,6),
  volume          BIGINT,
  open_interest   BIGINT,
  -- optional Greeks if you have them (Polygon paid add-ons or your own calc)
  iv              NUMERIC(12,6),
  delta           NUMERIC(12,6),
  gamma           NUMERIC(12,6),
  theta           NUMERIC(12,6),
  vega            NUMERIC(12,6),
  -- snapshot type helps when mixing "EOD close" vs "15:45" etc.
  snapshot_type   TEXT DEFAULT 'eod',            -- 'eod','1545','intraday_agg'...
  PRIMARY KEY (option_id, ts)
);

-- Create hypertable for time-series data
SELECT create_hypertable('option_quotes','ts', chunk_time_interval => INTERVAL '7 days');

-- Create indexes for performance
CREATE INDEX ON option_contracts (underlying, expiration);
CREATE INDEX ON option_quotes (option_id, ts DESC);

-- Add comments for documentation
COMMENT ON TABLE option_contracts IS 'Static metadata for option contracts';
COMMENT ON TABLE option_quotes IS 'Time-series quotes and Greeks for option contracts';
COMMENT ON COLUMN option_contracts.option_id IS 'Deterministic option identifier: UNDERLYING_EXPIRATION_STRIKECENTS_OPTION_RIGHT';
COMMENT ON COLUMN option_contracts.strike_cents IS 'Strike price in cents to avoid floating point precision issues';
COMMENT ON COLUMN option_quotes.snapshot_type IS 'Type of market snapshot: eod, intraday, etc.';
