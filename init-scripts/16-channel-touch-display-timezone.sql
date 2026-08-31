-- Display timezone for /hot timestamps (naive stored times are UTC).
-- exchange = America/New_York (NYSE/Nasdaq). Applied on first store use.

ALTER TABLE channel_touch_15m_settings
    ADD COLUMN IF NOT EXISTS display_timezone TEXT NOT NULL DEFAULT 'exchange';
