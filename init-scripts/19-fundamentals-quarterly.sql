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
