# IB 5m hourly watchdog

**2026-09-05.** Friday `after_rth_ib_backfill` died on reboot; Saturday 00:00 ET safety restart had already skipped. Change the 5m safety job from Sat/Sun 00:00 ET only to an **hourly watchdog in the backfill window**.

## Schedule (America/New_York)

`backfill_ib_5m_universe`:

- Weekends: every hour (`0 * * * 0,6`)
- Mon–Fri 17:00–23:00 ET and 00:00–08:00 ET (not RTH 09:15–16:30)
- Still `--skip-if-job-running after_rth_ib_backfill`
- `--reset-failed` on restart so Gateway drops are not a permanent skip list
- Clears leftover `ib_5m_universe.stop`; ignores recycled PID locks

## Cold-start coverage (same day)

The 11:00 local watchdog **did fire** but died at 11:02: TimescaleDB was healthy (`docker ps` Up) while `load_ib_coverage("15m")` `COUNT(*) GROUP BY symbol` hit `statement_timeout` 120s on a cold cache.

Fix: `utils/db/market_data_coverage.py` — recent-chunk `DISTINCT symbol` + per-symbol `ORDER BY ts LIMIT 1`, cache `first_ts`. No full-table GROUP BY.

Playbook: `docs/features/ib_5m_backfill.md`.
