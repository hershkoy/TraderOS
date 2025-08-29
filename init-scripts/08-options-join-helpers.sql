-- Options Join Helpers
-- This file implements functions and views to join options data with underlying market data

-- Function to get underlying close price at daily resolution (nearest prior bar)
CREATE OR REPLACE FUNCTION get_underlying_close(symbol TEXT, d TIMESTAMPTZ) 
RETURNS NUMERIC AS '
DECLARE
    result NUMERIC;
BEGIN
    -- Get the close price from market_data for the given symbol and date
    -- Use daily timeframe and find the nearest prior bar
    SELECT close INTO result
    FROM market_data 
    WHERE market_data.symbol = $1 
      AND market_data.timeframe = ''1D''
      AND market_data.ts <= $2
    ORDER BY market_data.ts DESC 
    LIMIT 1;
    
    RETURN result;
END;
' LANGUAGE plpgsql;

-- View that joins option_chain_eod with market_data to include underlying close and calculated fields
CREATE OR REPLACE VIEW option_chain_with_underlying AS
SELECT 
    o.ts,
    o.underlying,
    o.expiration,
    o.strike_cents,
    o.option_right,
    o.bid,
    o.ask,
    o.last,
    o.volume,
    o.open_interest,
    o.iv,
    o.delta,
    o.gamma,
    o.theta,
    o.vega,
    o.option_id,
    o.multiplier,
    o.strike_price,
    -- Get underlying close price for the same date
    get_underlying_close(o.underlying, o.ts) as underlying_close,
    -- Calculate moneyness: underlying_close / strike_price
    CASE 
        WHEN o.strike_price > 0 THEN 
            get_underlying_close(o.underlying, o.ts) / o.strike_price
        ELSE NULL 
    END as moneyness,
    -- Calculate days to expiration
    o.expiration - DATE(o.ts) as days_to_expiration
FROM option_chain_eod o;
