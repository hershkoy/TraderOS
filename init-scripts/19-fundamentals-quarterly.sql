-- SEC companyfacts-derived quarterly fundamentals panel
-- Used for cash-flow four-filter research (OCF/debt, CapEx/OCF, buybacks, FCF ROA)

CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
    symbol TEXT NOT NULL,
    cik TEXT NOT NULL,
    period_end DATE NOT NULL,
    filed DATE NOT NULL,
    fy INT,
    fp TEXT NOT NULL,
    ocf DOUBLE PRECISION,
    capex DOUBLE PRECISION,
    fcf DOUBLE PRECISION,
    total_debt DOUBLE PRECISION,
    assets DOUBLE PRECISION,
    shares_out DOUBLE PRECISION,
    buybacks_ttm DOUBLE PRECISION,
    issuance_ttm DOUBLE PRECISION,
    net_buybacks_ttm DOUBLE PRECISION,
    mkt_cap DOUBLE PRECISION,
    ocf_to_debt DOUBLE PRECISION,
    capex_to_ocf DOUBLE PRECISION,
    net_buyback_pct DOUBLE PRECISION,
    fcf_roa DOUBLE PRECISION,
    revenue DOUBLE PRECISION,
    gross_profit DOUBLE PRECISION,
    operating_income DOUBLE PRECISION,
    net_income DOUBLE PRECISION,
    pretax_income DOUBLE PRECISION,
    interest_expense DOUBLE PRECISION,
    interest_income DOUBLE PRECISION,
    current_assets DOUBLE PRECISION,
    current_liabilities DOUBLE PRECISION,
    equity DOUBLE PRECISION,
    diluted_shares DOUBLE PRECISION,
    diluted_eps DOUBLE PRECISION,
    depreciation DOUBLE PRECISION,
    ebitda DOUBLE PRECISION,
    gross_margin DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    net_margin DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    current_ratio DOUBLE PRECISION,
    debt_to_ebitda DOUBLE PRECISION,
    net_interest DOUBLE PRECISION,
    interest_to_ocf DOUBLE PRECISION,
    sic TEXT,
    sic_desc TEXT,
    exclude_sector BOOLEAN,
    debt_zero BOOLEAN,
    tag_ocf TEXT,
    tag_capex TEXT,
    tag_debt TEXT,
    source TEXT DEFAULT 'sec_companyfacts',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, period_end, fp)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_quarterly_filed
    ON fundamentals_quarterly (filed);

CREATE INDEX IF NOT EXISTS idx_fundamentals_quarterly_symbol_filed
    ON fundamentals_quarterly (symbol, filed);

CREATE INDEX IF NOT EXISTS idx_fundamentals_quarterly_exclude
    ON fundamentals_quarterly (exclude_sector)
    WHERE exclude_sector IS NOT TRUE;

-- Statement columns for the long-book screen. CREATE TABLE above does not
-- alter a table that already exists.
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS revenue DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS gross_profit DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS operating_income DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS net_income DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS pretax_income DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS interest_expense DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS interest_income DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS current_assets DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS current_liabilities DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS equity DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS diluted_shares DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS diluted_eps DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS depreciation DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS ebitda DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS gross_margin DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS operating_margin DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS net_margin DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS roe DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS current_ratio DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS debt_to_ebitda DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS net_interest DOUBLE PRECISION;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS interest_to_ocf DOUBLE PRECISION;
