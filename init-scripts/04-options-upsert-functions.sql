-- Options UPSERT functions for idempotent operations
-- Provides helper functions for upserting option contracts and quotes

-- Function to upsert option contracts
CREATE OR REPLACE FUNCTION upsert_option_contracts(
    p_option_id TEXT,
    p_underlying TEXT,
    p_expiration DATE,
    p_strike_cents INTEGER,
    p_option_right CHAR(1),
    p_multiplier INTEGER DEFAULT 100
) RETURNS VOID AS $$
BEGIN
    INSERT INTO option_contracts (
        option_id, underlying, expiration, strike_cents, 
        option_right, multiplier, first_seen, last_seen
    ) VALUES (
        p_option_id, p_underlying, p_expiration, p_strike_cents,
        p_option_right, p_multiplier, NOW(), NOW()
    )
    ON CONFLICT (option_id) DO UPDATE SET
        last_seen = NOW(),
        underlying = EXCLUDED.underlying,
        expiration = EXCLUDED.expiration,
        strike_cents = EXCLUDED.strike_cents,
        option_right = EXCLUDED.option_right,
        multiplier = EXCLUDED.multiplier;
END;
$$ LANGUAGE plpgsql;

-- Function to upsert option quotes
CREATE OR REPLACE FUNCTION upsert_option_quotes(
    p_ts TIMESTAMPTZ,
    p_option_id TEXT,
    p_bid NUMERIC,
    p_ask NUMERIC,
    p_last NUMERIC,
    p_volume BIGINT,
    p_open_interest BIGINT,
    p_iv NUMERIC,
    p_delta NUMERIC,
    p_gamma NUMERIC,
    p_theta NUMERIC,
    p_vega NUMERIC,
    p_snapshot_type TEXT DEFAULT 'eod'
) RETURNS VOID AS $$
BEGIN
    INSERT INTO option_quotes (
        ts, option_id, bid, ask, last, volume, open_interest,
        iv, delta, gamma, theta, vega, snapshot_type
    ) VALUES (
        p_ts, p_option_id, p_bid, p_ask, p_last, p_volume, p_open_interest,
        p_iv, p_delta, p_gamma, p_theta, p_vega, p_snapshot_type
    )
    ON CONFLICT (option_id, ts) DO UPDATE SET
        bid = EXCLUDED.bid,
        ask = EXCLUDED.ask,
        last = EXCLUDED.last,
        volume = EXCLUDED.volume,
        open_interest = EXCLUDED.open_interest,
        iv = EXCLUDED.iv,
        delta = EXCLUDED.delta,
        gamma = EXCLUDED.gamma,
        theta = EXCLUDED.theta,
        vega = EXCLUDED.vega,
        snapshot_type = EXCLUDED.snapshot_type;
END;
$$ LANGUAGE plpgsql;

-- Function to upsert multiple option contracts in batch
CREATE OR REPLACE FUNCTION upsert_option_contracts_batch(
    contracts_data option_contracts[]
) RETURNS INTEGER AS $$
DECLARE
    contract option_contracts;
    inserted_count INTEGER := 0;
BEGIN
    FOREACH contract IN ARRAY contracts_data
    LOOP
        INSERT INTO option_contracts (
            option_id, underlying, expiration, strike_cents, 
            option_right, multiplier, first_seen, last_seen
        ) VALUES (
            contract.option_id, contract.underlying, contract.expiration, contract.strike_cents,
            contract.option_right, contract.multiplier, NOW(), NOW()
        )
        ON CONFLICT (option_id) DO UPDATE SET
            last_seen = NOW(),
            underlying = EXCLUDED.underlying,
            expiration = EXCLUDED.expiration,
            strike_cents = EXCLUDED.strike_cents,
            option_right = EXCLUDED.option_right,
            multiplier = EXCLUDED.multiplier;
        
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to upsert multiple option quotes in batch
CREATE OR REPLACE FUNCTION upsert_option_quotes_batch(
    quotes_data option_quotes[]
) RETURNS INTEGER AS $$
DECLARE
    quote option_quotes;
    inserted_count INTEGER := 0;
BEGIN
    FOREACH quote IN ARRAY quotes_data
    LOOP
        INSERT INTO option_quotes (
            ts, option_id, bid, ask, last, volume, open_interest,
            iv, delta, gamma, theta, vega, snapshot_type
        ) VALUES (
            quote.ts, quote.option_id, quote.bid, quote.ask, quote.last, quote.volume, quote.open_interest,
            quote.iv, quote.delta, quote.gamma, quote.theta, quote.vega, quote.snapshot_type
        )
        ON CONFLICT (option_id, ts) DO UPDATE SET
            bid = EXCLUDED.bid,
            ask = EXCLUDED.ask,
            last = EXCLUDED.last,
            volume = EXCLUDED.volume,
            open_interest = EXCLUDED.open_interest,
            iv = EXCLUDED.iv,
            delta = EXCLUDED.delta,
            gamma = EXCLUDED.gamma,
            theta = EXCLUDED.theta,
            vega = EXCLUDED.vega,
            snapshot_type = EXCLUDED.snapshot_type;
        
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Add comments for documentation
COMMENT ON FUNCTION upsert_option_contracts(TEXT, TEXT, DATE, INTEGER, CHAR(1), INTEGER) IS 
'Upsert a single option contract, updating last_seen on conflict';

COMMENT ON FUNCTION upsert_option_quotes(TIMESTAMPTZ, TEXT, NUMERIC, NUMERIC, NUMERIC, BIGINT, BIGINT, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, TEXT) IS 
'Upsert a single option quote, updating all fields on conflict';

COMMENT ON FUNCTION upsert_option_contracts_batch(option_contracts[]) IS 
'Batch upsert multiple option contracts using array input';

COMMENT ON FUNCTION upsert_option_quotes_batch(option_quotes[]) IS 
'Batch upsert multiple option quotes using array input';
