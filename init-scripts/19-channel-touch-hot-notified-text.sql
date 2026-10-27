-- Sleeve scanners persist hot_notified_on as YYYY-MM-DD:approach / :arm.
-- DATE kept only the calendar day, so the once-per-session Telegram fired
-- again on every minute tick (QQQ LEAPS gap premarket is the dense window).

ALTER TABLE channel_touch_15m_candidates
    ALTER COLUMN hot_notified_on TYPE TEXT
    USING hot_notified_on::text;
