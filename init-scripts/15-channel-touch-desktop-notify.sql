-- Desktop-notify flag for /hot (Windows toast is separate; this is the dashboard checkbox).

ALTER TABLE channel_touch_15m_settings
    ADD COLUMN IF NOT EXISTS desktop_notify BOOLEAN NOT NULL DEFAULT TRUE;
