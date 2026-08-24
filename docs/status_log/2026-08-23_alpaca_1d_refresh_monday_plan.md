# ALPACA 1d data refresh + Monday channel-touch test (2026-08-23)

## Freshness probe (before update)
| Metric | Value |
|--------|-------|
| Symbols in DB | 2217 |
| Global max `ts` | 2026-08-21 (14 symbols) |
| Majority stuck at | **2025-11-21** (~1842 symbols) |
| Also stale cluster | 2025-10-24 (~347) |

Gap is ~9 months for most names (not a full calendar year, but close to "last scan a year ago").

## Fix applied
- `utils/data/update_universe_data.py`: DB worker used relative `from ..db...` which failed when run as a script (`attempted relative import...`). Fetches succeeded but **nothing was saved**. Switched to absolute `utils.db.timescaledb_client` with relative fallback.

## Update job — completed
Initial full run (2217 symbols from `2025-11-01`):
```bat
python utils\data\update_universe_data.py --provider alpaca --timeframe 1d --since 2025-11-01 --universe-file reports\ascending_channels\alpaca_1d_symbols.txt --batch-size 25 --delay-tickers 0.2 --delay-batches 2
```

### Hang / resume
- Cursor terminal **aborted** around **CRGY (505/2217)** after Alpaca returned bars; Python stayed alive but **blocked on dead stdout** (0 CPU).
- Killed hung PIDs; mid-run freshness was already ~1429/2217 with `MAX(ts) >= 2026-08-20`.
- Resumed **788** stale symbols via `reports/ascending_channels/alpaca_1d_need_update.txt`, logging to `logs/alpaca_1d_gapfill_resume.log` (file redirect — survives terminal death).
- Resume finished **2026-08-23 22:21** in ~12 min: **781 saved / 7 failed / 99.1%**.
  - Hard fails: ABBNY, CAJPY, CCCS, ERJ, ETNB, TRML, WBA

### Post-refresh freshness
| Metric | Value |
|--------|-------|
| Fresh `MAX(ts) >= 2026-08-20` | **2109 / 2217** |
| Still stale | **108** (thin / delisted / no recent Alpaca bars; not only the 7 hard fails) |
| Missing entirely | 0 |

Summary: `logs/data/universe/universe_update_summary_20260823_222104.txt`

## Refreshed-universe channel-touch test (same evening)
See `docs/status_log/2026-08-23_channel_touch_refreshed_universe_scan.md`.

## Monday (2026-08-25) EOD test-run plan
Channel-touch is **daily + 15-bar pivot confirm** → scan **after RTH close**, not live IB streams.

Suggested sequence (ET):
1. **16:15** — Incremental update last 5 trading days:
   ```bat
   python utils\data\update_universe_data.py --provider alpaca --timeframe 1d --since 2026-08-18 --universe-file reports\ascending_channels\alpaca_1d_symbols.txt --batch-size 25 --delay-tickers 0.2
   ```
2. **16:30** — Run with `--end` = Monday's session date (cold cache OK if end date is new):
   ```bat
   python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --trail-pct-wide 0.18 --max-entries-per-day 1 --friction-pct 0.25 --workers 4 --load-workers 8 --end 2026-08-25
   ```
3. **16:45** — Generate interactive TV report; review new Touch≥3 / RS-top1 candidates for Tuesday open
4. IB: subscribe/execute **candidates + open positions only** (not full 2.2k)

### Ops notes
- Prefer `venv\Scripts\python.exe ... > logs\....log 2>&1` (or a `.bat`) so a dead Cursor terminal cannot hang the job.
- Default `--end` in the backtester is still `2025-11-26` — always pass a current `--end` after refreshes.
- Parquet cache is keyed by `(start,end)`; new `--end` creates new cache files (no need to wipe old ranges).
