# 15m channel-touch live monitor

Research-only loop for the **H5 + overshoot p80 + vol>=2** book. It does **not** replace the daily nightly job (`channel_touch_nightly.py`, H2 resist-break span365).

Interactive report: `reports/ascending_channels/current_best/15m_h5_overshoot_vol.html`.

---

## Why not scan all 1,478 names every 15 minutes on IB?

| Approach | Time / limit | Use? |
|----------|----------------|------|
| Re-run the full detector on 2018–now IB 15m | ~51 min (cold TimescaleDB load) | No — misses the 15m clock |
| IB historical last bar × 1,478 | ~25 min at 1s pacing; IB pacing violations | No |
| IB streaming all names | Typical ~100 market-data lines | No |
| Alpaca IEX snapshot (last trade + daily volume) × 1,478 | **~1.3s** (timed 2026-08-30, batch 200) | Yes — **proximity only** |
| Alpaca IEX 15m bars × 1,478 | Fast, but only ~1/3 of names returned; IEX volume ≠ IB | No for H5 volume |
| Armed H2 watchlist + Alpaca last + IB 15m on the hot list | Seconds for last price; IB only for tens–low hundreds | **Yes** |

Last price is a **proximity screen**, not a fill. The backtest fill is a **completed 15m bar**: close above the rising resistance after wait-12, fill clipped to the bar's range (often the rail; if the whole bar is already through, fill is the bar low and overshoot is large).

---

## What this book actually fills

Same detector v1 (`find_h2_l3_setups`), `--preset 15m` geometry:

| Gate | Live value | Notes |
|------|------------|--------|
| Entry | H2 resist-break | First close above resistance after H2 |
| Min wait | 12 bars | Closes above resist during wait are ignored (not fills) |
| Span | ≤ 10 days | Do not copy daily 365 |
| Unique symbol/day | on | Not RS top1 |
| H5 | prior-bar `volume_rel_20 >= 2` | IB 15m volume, not Alpaca IEX |
| Overshoot | fill `channel_pos - 1 >= 0.08` | Frozen live floor ≈ unique-book train p80 / top quintile (0.088+). Expanding-year p80 is **not** recomputed each bar. |
| Proximity | last **at or above** resist | Tight-break-from-below was the rejected unique H2 filter. Quality names are often already through the rail after wait-12. |
| Friction (research) | 0.10% | Not a scan gate |

~2.5 fills per RTH day on the tight stack in sample. Still **research** — drop-top-N holds; no full bootstrap / max-open pass. Do not wire into the daily nightly cron.

---

## Data gap (must backfill first)

Stored IB 15m inventory (TimescaleDB, 2026-08-30): **1,479 symbols**.

| | Timestamp (UTC) |
|--|-----------------|
| First bar (min) | 2018-01-02 14:30 |
| Last bar (median) | **2025-11-28 17:45** |
| Last bar (min) | 2025-11-21 20:45 |
| Last bar (max) | 2026-08-26 16:00 (1 name, almost certainly SPY) |

**1,478 / 1,479** names have no 2026 bars. A live 15m scan on that panel will see last year. CSV: `reports/ascending_channels/ib_15m_coverage.csv`.

Incremental backfill (one IB request per stale symbol from `MAX(ts)` to now, client id **8822**):

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\data\backfill_ib_15m_universe.py --inventory
python scripts\data\backfill_ib_15m_universe.py --dry-run --limit 5
python scripts\data\backfill_ib_15m_universe.py --sleep 1 --ib-client-id 8822
```

Or: `crons\backfill_ib_15m_universe.bat`

| Piece | Path |
|-------|------|
| Universe backfill | `scripts/data/backfill_ib_15m_universe.py` |
| Single-symbol (SPY, etc.) | `scripts/data/backfill_ib_15m_symbol.py` (client **8821**) |
| Coverage CSV | `reports/ascending_channels/ib_15m_coverage.csv` |
| Resume | `logs/data/ib_15m_universe_resume.txt` |

Expect **hours**, not minutes: ~1,478 symbols × ~1s pacing plus IB hist. Keep Gateway up; use a **distinct client id** if another IB hist session is running. Re-run is resume-safe. `--fresh-hours 36` skips names already current. `--stale-before 2026-01-01` only touches the 2025-ending names.

After backfill, re-check coverage with `--inventory` (last_max should be the latest RTH 15m bar).

Alpaca 15m is **not** a substitute. Research volume is IB. IEX daily volume on the snapshot is fine for “is this name trading,” not for `volume_rel_20`.

---

## Live loop

TradingView drawing alerts are **not** the live path. CronRunner ticks every minute;
proximity uses Alpaca last on the armed list; fills still wait for a completed 15m bar.

Dashboard: `http://localhost:5000/hot` (charting_server). Source of truth is TimescaleDB
(`channel_touch_15m_candidates` + `channel_touch_15m_settings`). Rows are keyed by
`(stock, timeframe)` so **15m** and **1d nightly** share the page (filter All / 15m / 1d).
JSON watchlist is a debug sidecar. Nightly writes 1d armed/waiting/filled H2 rows unless
`--skip-hot-dashboard`.

Telegram on H5 fills vs newly-hot is a persisted UI setting on that page (cron reads the same row).

```
overnight / weekend
  backfill_ib_15m_universe.py          # catch IB 15m up to now

each 15m bar close (RTH)
  load last ~40 sessions IB 15m from TimescaleDB
  find armed H2 setups (wait>=12, span<=10, not cancelled)
  replace TimescaleDB candidates (keep live last if same H2)
  if last completed bar closed above resist:
      unique-symbol/day + prior-bar vol>=2 + fill overshoot>=0.08
      Telegram if settings.telegram_on_fill

each RTH minute
  Alpaca last on the armed list (~1s)
  update last_price / dist_live_pct / hot in TimescaleDB
  Telegram newly-hot if settings.telegram_on_hot (once per symbol per day)

dashboard /hot (poll 5s)
  read candidates from TimescaleDB
  refresh Alpaca if last_price_ts older than ~5s (process throttle)
  sort/filter by |dist_live_pct| to resist
```

Do **not** stream 1,478 names. The detector runs on **stored** bars (short lookback, not 2018–now). Alpaca is last trade only. IB hist is for the backfill and any future “refresh last 2D for the hot list” — not a full-universe poll every bar.

### Commands

```bat
venv\Scripts\activate
set PYTHONPATH=.

REM Armed watchlist only (no Telegram); writes JSON + TimescaleDB
python scripts\scanners\channel_touch_15m.py --mode watchlist --dry-run

REM Full 15m loop, no Telegram
python scripts\scanners\channel_touch_15m.py --mode run --dry-run

REM Debug on 50 names
python scripts\scanners\channel_touch_15m.py --mode run --dry-run --max-symbols 50

REM Minute proximity (DB first, JSON fallback)
python scripts\scanners\channel_touch_15m.py --mode proximity --dry-run
python scripts\scanners\channel_touch_15m.py --mode fills --dry-run

REM Dashboard
python charting_server.py
REM then open http://localhost:5000/hot
```

Cron wrappers (do not replace nightly). Seeded **disabled** until IB 15m is current:

- `crons\channel_touch_15m.bat` — 15m bar-close `--mode run`
- `crons\channel_touch_15m_proximity.bat` — RTH minute `--mode proximity`

Enable after backfill: `python scripts\pipeline\cron_manager.py enable channel_touch_15m` and `enable channel_touch_15m_proximity`.

If IB 15m is still stale, the scanner **warns** and still builds a watchlist from old bars. It will not produce trustworthy live fills until backfill is done (`--stale-hours 36`).

---

## Files

| Piece | Path |
|-------|------|
| Live helpers | `utils/scanning/channel_touch_15m.py` |
| Candidates store | `utils/scanning/channel_touch_candidates_store.py` |
| Dashboard API | `utils/scanning/channel_touch_hot_api.py` |
| Scanner | `scripts/scanners/channel_touch_15m.py` |
| Dashboard | `charting_server.py` `/hot` + `templates/hot_candidates.html` |
| Schema | `init-scripts/13-channel-touch-15m-candidates.sql` |
| IB 15m universe backfill | `scripts/data/backfill_ib_15m_universe.py` |
| Watchlist JSON (sidecar) | `reports/ascending_channels/channel_touch_15m_watchlist.json` |
| Fills log | `reports/ascending_channels/channel_touch_15m_fills_log.csv` |
| Scanner logs | `logs/scanners/channel_touch_15m_*.log` |
| Daily nightly (unchanged) | `scripts/scanners/channel_touch_nightly.py` |

Telegram uses the same `.env` keys as nightly: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
Fills and newly-hot also raise a **Windows toast + system sound** (`utils/notify/desktop.py`) unless
**Desktop + sound** is unchecked on `/hot`. With `/hot` open, newly-hot names also beep in the browser.

---

## What not to do

- Do not use TradingView trendline alerts as the 15m live path (dashboard + minute cron instead).
- Do not copy daily span 365, RSI 50, in-channel, or beyond-width 0.25 onto 15m.
- Do not treat “price 2% below resist and climbing” as the hot list for this book (dashboard may *show* distance; hot/Telegram default remains at-or-above).
- Do not fire a fill on last price crossing the rail — wait for the 15m close.
- Do not use Alpaca IEX 15m `volume_rel` as the H5 gate.
- Do not promote 15m into the daily nightly cron until bootstrap / max-open stress is done.
- Do not stream the IB 15m universe.

---

## Status

See `docs/status_log/edge_hunt/channel_touch/2026-08-30_channel_touch_15m_live_monitor.md`.
