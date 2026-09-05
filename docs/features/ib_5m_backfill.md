# IB 5m universe backfill

Historical **5-minute** OHLCV from Interactive Brokers for the same names that already have IB **15m** in TimescaleDB. Used later for more realistic strategy fills. Not a live 15m substitute.

Fill order is **newest year first, all symbols**, then the next older year: default **2025-01-01 through now** (includes 2026 YTD) for the whole 15m universe, then calendar **2024 … 2020**. That way a whole-universe 2025+ simulation can start before 2018–2019 history exists. `--no-year-slice` restores per-symbol full history (first 15m bar → now). `--no-through-now` caps 2025 at 2026-01-01.

Client IDs (keep distinct; overlapping `ib.connect()` wedges Gateway):

| Role | Client | When |
|------|--------|------|
| Live 15m hot/armed | **8826** | Every RTH 15m close |
| 15m universe catch-up | **8822** | After 16:30 ET, before 5m |
| 5m universe | **8823** | After 15m catch-up until next 09:15 ET (weekends continuous) |
| 5m single symbol | **8824** | Manual |
| 15m single symbol | **8821** | Manual |

## Schedule

CronRunner jobs in `crons/crontab.yaml` (America/New_York unless noted):

1. **16:00 ET-ish** — existing `channel_touch_nightly` (local 23:00) refreshes **Alpaca 1d** and runs the daily H2 scan. Alpaca, not IB.
2. **16:30 ET Mon–Fri** — `after_rth_ib_backfill`: IB **15m** catch-up (`--reset-resume --fresh-hours 12`, client 8822), then IB **5m** historical (client 8823) until the next weekday **09:15 ET**. Friday’s run continues through the weekend.
3. **09:15 ET Mon–Fri** — `stop_ib_5m_backfill`: writes `logs/data/ib_5m_universe.stop`, waits, then kills the job tree so live 15m owns Gateway.
4. **Hourly in the backfill window** — `backfill_ib_5m_universe` watchdog: weekends all hours; Mon–Fri **17:00–08:00 ET** (not RTH 09:15–16:30). Restarts 5m if `after_rth_ib_backfill` died (`--skip-if-job-running after_rth_ib_backfill --reset-failed`). Skips if that job or a live 5m pid lock is held. Clears leftover stop files and ignores stale locks after reboot.
5. Overnight `backfill_ib_15m_universe` at 02:30 local is **disabled** (folded into step 2).

The 5m process also self-exits at next 09:15 ET (`--until`) and refuses to start Mon–Fri 09:15–16:30 ET unless `--allow-rth`.

## Resume

TimescaleDB `MIN/MAX(ts)` **inside the current year window** is the cursor (not global `MAX(ts)` — a 2026 last bar must not skip 2024). Each IB window (default **7 days**, official 5m max) is upserted before the next request. Ctrl+C, stop file, `--until`, or the RTH guard stop after the current window. Re-run continues the same year-first queue.

Failed qualify / insert: `logs/data/ib_5m_universe_failed.txt` (skip until `--reset-failed`). Gateway drops (`ConnectionRefused`, `Not connected`) are **not** written there; the hourly watchdog also passes `--reset-failed` so an outage cannot skip thousands of names. Empty 5m in an older year is not a global fail. Progress: `logs/data/ib_5m_universe_progress.json`. Coverage CSV: `reports/ascending_channels/ib_5m_coverage.csv`.

Do **not** copy the 15m skip-forever resume file pattern. `--fresh-hours` still skips names that are caught up on the newest (through-now) slice.

## Commands

```bat
venv\Scripts\activate
set PYTHONPATH=.

python scripts\data\backfill_ib_5m_universe.py --inventory
python scripts\data\backfill_ib_5m_universe.py --dry-run --limit 2
python scripts\data\backfill_ib_5m_universe.py --sleep 1 --ib-client-id 8823 --ib-port 4001
python scripts\data\backfill_ib_5m_universe.py --no-year-slice

crons\after_rth_ib_backfill.bat
crons\backfill_ib_5m_universe.bat
crons\stop_ib_5m_backfill.bat

python scripts\data\backfill_ib_5m_symbol.py --symbols SPY --ib-client-id 8824 --since 2018-01-01
python scripts\pipeline\cron_manager.py stop after_rth_ib_backfill
```

Probe Gateway first: `tests/utils/ib_conn.py --port 4001`. Do not run 5m during RTH next to client 8826.

Expect **days**, not hours: ~1,478 names × years of 5m at ~1s/window. Weekend wall-clock is the bulk of the first fill.
