-- Options helper views for the LEAPS strategy
-- Provides convenient views for querying option data

-- View: Current option chain at close
-- Shows the latest quotes for each option contract
CREATE OR REPLACE VIEW option_chain_eod AS
SELECT 
    q.ts,
    c.underlying,
    c.expiration,
    c.strike_cents,
    c.option_right,
    q.bid,
    q.ask,
    q.last,
    q.volume,
    q.open_interest,
    q.iv,
    q.delta,
    q.gamma,
    q.theta,
    q.vega,
    c.option_id,
    c.multiplier,
    -- Calculate strike price in dollars
    (c.strike_cents / 100.0) AS strike_price,
    -- Calculate days to expiration
    (c.expiration - CURRENT_DATE) AS days_to_expiration
FROM option_quotes q
JOIN option_contracts c USING (option_id)
WHERE q.snapshot_type = 'eod'
  AND q.ts = (
      SELECT MAX(q2.ts) 
      FROM option_quotes q2 
      WHERE q2.option_id = q.option_id 
        AND q2.snapshot_type = 'eod'
  );

-- View: Option chain with underlying price
-- Joins option data with underlying market data
CREATE OR REPLACE VIEW option_chain_with_underlying AS
SELECT 
    o.*,
    m.close AS underlying_close,
    -- Calculate moneyness (underlying price / strike price)
    CASE 
        WHEN m.close IS NOT NULL AND o.strike_price > 0 
        THEN (m.close / o.strike_price)
        ELSE NULL 
    END AS moneyness,
    -- Calculate intrinsic value
    CASE 
        WHEN o.option_right = 'C' AND m.close > o.strike_price
        THEN (m.close - o.strike_price) * o.multiplier
        WHEN o.option_right = 'P' AND m.close < o.strike_price
        THEN (o.strike_price - m.close) * o.multiplier
        ELSE 0
    END AS intrinsic_value,
    -- Calculate time value (option price - intrinsic value)
    CASE 
        WHEN o.last IS NOT NULL AND (
            CASE 
                WHEN o.option_right = 'C' AND m.close > o.strike_price
                THEN (m.close - o.strike_price) * o.multiplier
                WHEN o.option_right = 'P' AND m.close < o.strike_price
                THEN (o.strike_price - m.close) * o.multiplier
                ELSE 0
            END
        ) IS NOT NULL
        THEN (o.last - (
            CASE 
                WHEN o.option_right = 'C' AND m.close > o.strike_price
                THEN (m.close - o.strike_price) * o.multiplier
                WHEN o.option_right = 'P' AND m.close < o.strike_price
                THEN (o.strike_price - m.close) * o.multiplier
                ELSE 0
            END
        ))
        ELSE NULL
    END AS time_value
FROM option_chain_eod o
LEFT JOIN market_data m ON (
    m.symbol = o.underlying 
    AND DATE(m.ts) = DATE(o.ts)
    AND m.timeframe = '1d'  -- Assuming daily data
);

-- View: LEAPS candidates
-- Filters for long-term call options suitable for LEAPS strategy
CREATE OR REPLACE VIEW leaps_candidates AS
SELECT 
    *,
    -- Additional LEAPS-specific calculations
    CASE 
        WHEN delta IS NOT NULL THEN
            CASE 
                WHEN delta BETWEEN 0.6 AND 0.85 THEN 'Optimal'
                WHEN delta BETWEEN 0.5 AND 0.9 THEN 'Acceptable'
                ELSE 'Outside Range'
            END
        WHEN moneyness IS NOT NULL THEN
            CASE 
                WHEN moneyness BETWEEN 0.9 AND 1.1 THEN 'Optimal'
                WHEN moneyness BETWEEN 0.8 AND 1.2 THEN 'Acceptable'
                ELSE 'Outside Range'
            END
        ELSE 'Unknown'
    END AS suitability_score
FROM option_chain_with_underlying
WHERE option_right = 'C'
  AND days_to_expiration >= 365
  AND (
      (delta IS NOT NULL AND delta BETWEEN 0.6 AND 0.85)
      OR 
      (delta IS NULL AND moneyness BETWEEN 0.9 AND 1.1)
  )
  AND underlying_close IS NOT NULL
  AND last IS NOT NULL;

-- View: Short call candidates
-- Filters for short-term call options suitable for covered calls
CREATE OR REPLACE VIEW short_call_candidates AS
SELECT 
    *,
    -- Additional short call-specific calculations
    CASE 
        WHEN delta IS NOT NULL THEN
            CASE 
                WHEN delta BETWEEN 0.15 AND 0.35 THEN 'Optimal'
                WHEN delta BETWEEN 0.1 AND 0.4 THEN 'Acceptable'
                ELSE 'Outside Range'
            END
        WHEN moneyness IS NOT NULL THEN
            CASE 
                WHEN moneyness BETWEEN 1.02 AND 1.08 THEN 'Optimal'
                WHEN moneyness BETWEEN 1.0 AND 1.1 THEN 'Acceptable'
                ELSE 'Outside Range'
            END
        ELSE 'Unknown'
    END AS suitability_score
FROM option_chain_with_underlying
WHERE option_right = 'C'
  AND days_to_expiration BETWEEN 25 AND 45
  AND (
      (delta IS NOT NULL AND delta BETWEEN 0.15 AND 0.35)
      OR 
      (delta IS NULL AND moneyness BETWEEN 1.02 AND 1.08)
  )
  AND underlying_close IS NOT NULL
  AND last IS NOT NULL;

-- View: PMCC strategy candidates
-- Combines LEAPS and short call candidates for PMCC strategy
CREATE OR REPLACE VIEW pmcc_candidates AS
SELECT 
    'LEAPS' AS leg_type,
    *,
    'Long-term call for core position' AS strategy_note
FROM leaps_candidates
WHERE suitability_score IN ('Optimal', 'Acceptable')

UNION ALL

SELECT 
    'SHORT_CALL' AS leg_type,
    *,
    'Short-term call for income generation' AS strategy_note
FROM short_call_candidates
WHERE suitability_score IN ('Optimal', 'Acceptable');

-- Add comments for documentation
COMMENT ON VIEW option_chain_eod IS 'Current option chain at end-of-day with latest quotes';
COMMENT ON VIEW option_chain_with_underlying IS 'Option chain joined with underlying market data and calculated metrics';
COMMENT ON VIEW leaps_candidates IS 'Long-term call options suitable for LEAPS strategy';
COMMENT ON VIEW short_call_candidates IS 'Short-term call options suitable for covered calls';
COMMENT ON VIEW pmcc_candidates IS 'Combined view of LEAPS and short call candidates for PMCC strategy';
