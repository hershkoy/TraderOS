# QQQ LEAPS gap live monitor (`/hot` Detector **LG**)

Live sleeve for the Ravish QQQ LEAPS gap claim-check (research: `utils/research/qqq_leaps_gap.py`, HTML under `reports/research/qqq_leaps_gap/current_best/`).

## Rules

| Mode | Trigger | Size |
|------|---------|------|
| Strat1 | Open gap ≤ **−1%** and prior close **> SMA100** | 1× ~60δ ~365 DTE call |
| Strat2 | Open gap ≤ **−2%** | 2× same |

Exit research: TP +50% premium, no stop. Live monitor alerts entry only (does not auto-trade or track LEAPS TP).

## Hot / approaching (getting close)

Cron dense **08:00–10:30 ET**, then every 5m until **11:00**:

1. Load QQQ 1d from TimescaleDB → prior close + SMA100.
2. Alpaca snapshot → last (+ session open when available).
3. Soft **HOT / approaching** when estimated gap reaches **−0.7%** (strat1 + SMA) or **−1.5%** (strat2) — premarket or early RTH.
4. **BUY NOW / armed** after 09:30 when official open (or open proxy) meets −1%/−2%.

`/hot` Last column: `current / gap-trigger open` (green when gap already meets the active rule). Support = SMA100; Resist = trigger open.

## Wiring

| Piece | Path |
|-------|------|
| Logic | `utils/scanning/qqq_leaps_gap_live.py` |
| Scanner | `scripts/scanners/qqq_leaps_gap_live.py` |
| Cron | `crons/qqq_leaps_gap_live.bat` + job `qqq_leaps_gap_live` in `crons/crontab.yaml` |
| Detector | `timeframe=lg` on `channel_touch_15m_candidates` |
| Notify | `/hot` matrix detector **LG** — hot / approaching / armed (defaults on for Telegram + browser) |

## Manual

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\scanners\qqq_leaps_gap_live.py --dry-run
python scripts\scanners\qqq_leaps_gap_live.py
```

Restart ChartingServer after deploy so `/hot` picks up the LG filter + notify prefs. CronRunner reloads `crontab.yaml` on its next tick (or restart the CronRunner task).
