-- Manual /hot "Bought" positions and SELL NOW stop tracking.
-- Applied on first store use (CREATE / ALTER IF NOT EXISTS).

ALTER TABLE channel_touch_15m_settings
    ADD COLUMN IF NOT EXISTS telegram_on_sell BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS channel_touch_bought_trades (
    id SERIAL PRIMARY KEY,
    stock TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '15m',
    h2_time TEXT,
    entry_px DOUBLE PRECISION NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    hard_stop DOUBLE PRECISION NOT NULL,
    atr_at_entry DOUBLE PRECISION,
    stop_pct_used DOUBLE PRECISION,
    trail_pct DOUBLE PRECISION NOT NULL DEFAULT 0.10,
    peak_px DOUBLE PRECISION NOT NULL,
    current_stop DOUBLE PRECISION NOT NULL,
    last_price DOUBLE PRECISION,
    last_price_ts TIMESTAMPTZ,
    dist_to_stop_pct DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'open',
    exit_reason TEXT,
    sell_notified_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ct_bought_one_active
    ON channel_touch_bought_trades (stock, timeframe)
    WHERE status IN ('open', 'sell_now');

CREATE INDEX IF NOT EXISTS idx_ct_bought_status
    ON channel_touch_bought_trades (status);
