# 15m hot-candidates dashboard (no TV alerts) — 2026-08-30

Replaced TradingView drawing alerts for the 15m H5 book with:

- TimescaleDB `channel_touch_15m_candidates` + `channel_touch_15m_settings`
- RTH minute Alpaca proximity cron (`channel_touch_15m_proximity`, seeded disabled)
- Flask dashboard `http://localhost:5000/hot` (sort/filter by `|dist_live_pct|` to resist, live last via 5s poll)
- Telegram on H5 fills and/or newly-hot as persisted UI checkboxes (once per symbol per day for hot)

Last price is still **not** a fill. 15m bar-close job still rebuilds rails and gates H5.
Jobs stay disabled until IB 15m catch-up. Playbook: `docs/features/channel_touch_15m_live.md`.
