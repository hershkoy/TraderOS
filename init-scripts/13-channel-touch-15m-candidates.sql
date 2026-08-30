-- 15m channel-touch live candidates + dashboard settings
-- Applied on first store use as well (CREATE TABLE IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS channel_touch_15m_settings (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    telegram_on_fill BOOLEAN NOT NULL DEFAULT TRUE,
    telegram_on_hot BOOLEAN NOT NULL DEFAULT FALSE,
    proximity_below_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_abs_dist_pct DOUBLE PRECISION,
    sort_key TEXT NOT NULL DEFAULT 'abs_dist',
    sort_dir TEXT NOT NULL DEFAULT 'asc',
    status_filter TEXT NOT NULL DEFAULT 'all',
    search TEXT NOT NULL DEFAULT '',
    as_of TEXT,
    n_universe INTEGER NOT NULL DEFAULT 0,
    stale_warning TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO channel_touch_15m_settings (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS channel_touch_15m_candidates (
    stock TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    as_of TEXT,
    h2_time TEXT,
    channel_start TEXT,
    channel_end TEXT,
    channel_span_days DOUBLE PRECISION,
    wait_bars INTEGER,
    wait_ok BOOLEAN,
    support DOUBLE PRECISION,
    resist DOUBLE PRECISION,
    last_close DOUBLE PRECISION,
    dist_to_resist_pct DOUBLE PRECISION,
    volume_rel_20 DOUBLE PRECISION,
    overshoot_prior DOUBLE PRECISION,
    fill_px DOUBLE PRECISION,
    overshoot DOUBLE PRECISION,
    support_x0 INTEGER,
    support_y0 DOUBLE PRECISION,
    support_slope DOUBLE PRECISION,
    channel_width DOUBLE PRECISION,
    h2_idx INTEGER,
    as_of_i INTEGER,
    last_price DOUBLE PRECISION,
    last_price_ts TIMESTAMPTZ,
    dist_live_pct DOUBLE PRECISION,
    hot BOOLEAN NOT NULL DEFAULT FALSE,
    hot_notified_on DATE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ct15m_cand_hot
    ON channel_touch_15m_candidates (hot);

CREATE INDEX IF NOT EXISTS idx_ct15m_cand_status
    ON channel_touch_15m_candidates (status);

CREATE INDEX IF NOT EXISTS idx_ct15m_cand_dist
    ON channel_touch_15m_candidates (dist_live_pct);
