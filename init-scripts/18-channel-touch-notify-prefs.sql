-- Per-detector notify matrix for /hot (JSON blob; legacy bool columns stay synced).

ALTER TABLE channel_touch_15m_settings
    ADD COLUMN IF NOT EXISTS notify_prefs JSONB;
