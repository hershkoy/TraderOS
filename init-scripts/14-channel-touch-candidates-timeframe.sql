-- Add 1d nightly candidates to the same hot-dashboard table (stock + timeframe).
-- Existing rows default to 15m. Applied on first store use.

ALTER TABLE channel_touch_15m_settings
    ADD COLUMN IF NOT EXISTS timeframe_filter TEXT NOT NULL DEFAULT 'all';
ALTER TABLE channel_touch_15m_settings
    ADD COLUMN IF NOT EXISTS as_of_1d TEXT;
ALTER TABLE channel_touch_15m_settings
    ADD COLUMN IF NOT EXISTS n_universe_1d INTEGER NOT NULL DEFAULT 0;

ALTER TABLE channel_touch_15m_candidates
    ADD COLUMN IF NOT EXISTS timeframe TEXT;
UPDATE channel_touch_15m_candidates
    SET timeframe = '15m'
    WHERE timeframe IS NULL OR timeframe = '';
ALTER TABLE channel_touch_15m_candidates
    ALTER COLUMN timeframe SET DEFAULT '15m';
ALTER TABLE channel_touch_15m_candidates
    ALTER COLUMN timeframe SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ct15m_cand_tf
    ON channel_touch_15m_candidates (timeframe);
