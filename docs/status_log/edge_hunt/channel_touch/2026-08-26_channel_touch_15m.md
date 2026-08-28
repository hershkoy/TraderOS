# Channel-touch 15m — scaled hunt (not a daily drop-in)

Date: 2026-08-26

New edge hunt: same classical 3-touch detector, **scaled to 15m**. Not the live daily keepers.

## What is reused

- Detector v1 `find_channels` (Edwards/Magee). No rewrite.
- Backtest `scripts/research/backtest_channel_touch_trades.py` (`--preset 15m`)
- Loader `utils/data/ohlcv_loader.py` (IB 15m parquet cache)
- RS top1 / in-channel / span filters / squeeze-adaptive trail / ATR k=2

Added only:
- `utils/research/channel_touch_scale.py` — 15m defaults
- `find_channels_windowed` — overlapping slices so history is not truncated to the last 16 swings

## Scale

| Piece | Daily | 15m |
|-------|-------|-----|
| Universe | ALPACA 1d | **IB 15m** (~1,478; through ~2025-12-01) |
| Pivots | 15 bars (~3w) | **8 bars (~2h)** — shorter, not 15*26 |
| % rallies / error / trails / stop clamp | 4% / 1.2% / 10–18% / 1.5–6% | daily / sqrt(26) |
| RS 63/126 | 63/126 daily bars | **intended** 63/126 sessions (1638 / 3276 15m bars) |
| Channel window | last 16 swings | slide **15 sessions**, step 5 |
| Max channel span | 365 calendar days | **10 calendar days** |
| Friction | 0.25% | 0.10% default |

## Data gap

IB 15m previously had **no SPY**. Backfilled 2026-08-26 via `scripts/data/backfill_ib_15m_symbol.py --ib-client-id 8821`: **56,293 bars**, 2018-01-02 14:30 UTC → 2026-08-26 16:00 UTC (RTH). QQQ/GLD/IWM/NVDA still missing. Next 15m hunt can use native session-equivalent RS vs SPY.

## Smoke (10 IB 15m names, 2022-01-01 → 2025-12-02)

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_trades.py --preset 15m --symbols AAPL,MSFT,AMZN,META,TSLA,AMD,GOOGL,NFLX,AVGO,JPM --workers 4 --load-workers 4 --start 2022-01-01
```

Bat: `scripts/research/run_channel_touch_15m_smoke.bat`

Wall clock **26.3s** (15m load 13.5s incl. cache; ALPACA 1d RS 8.4s; scan 4.1s).

| Metric | Value |
|--------|--------|
| Raw trades | 1134 |
| After in-channel + span<=10d + RS top1/day + 0.10% friction | **437** (10 symbols) |
| Win rate | 34.3% |
| Expectancy | **+0.50%** |
| PF | **1.93** |
| Median | **-0.62%** (fat tail, same shape as daily) |
| Avg hold | 38.7 **bars** (~1.5 sessions) |
| Exits | hard 220 / trail 212 / wide 3 |

CSV: `reports/ascending_channels/channel_touch_15m_trades_20260826_215119.csv`

**Not a promote.** 10-name smoke only; RS is daily fallback; `hold_days` column is bar count on 15m.

## Expansion (300 liquid IB 15m names, 2018-11-01 → 2025-12-02)

Native **IB 15m SPY** RS (126 sessions = 3276 bars). Ranked by ALPACA 1d ADV, intersect IB 15m (`_pick_15m_by_daily_adv`).

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --workers 4 --load-workers 8
```

Bat: `scripts/research/run_channel_touch_15m_n300.bat`

Wall clock **786.7s** (~13.1 min): pick ~2.4 min; load 481.2s; scan 155.6s (57,770 raw); RS 15.1s.

| Metric | 10-name smoke (2022+, daily RS) | 300-name (2018+, 15m SPY RS) |
|--------|----------------------------------|------------------------------|
| After filters | 437 | **1764** (268 symbols; ~1 RS-top1/session-day) |
| Win rate | 34.3% | 32.9% |
| Expectancy | +0.50% | **+0.15%** |
| PF | 1.93 | **1.24** |
| Median | −0.62% | **−0.70%** |
| Avg hold | 38.7 bars | 27.0 bars (~1 session) |

CSV: `reports/ascending_channels/channel_touch_15m_trades_20260826_234629.csv`

Edge thinned vs the mega-cap smoke (expected). Still fat-tailed (median negative). **Not a promote.**

Do **not** wire nightly 15m. Do **not** stream the IB 15m universe.

## Next

- Split-adjust check (NFLX 15m prints in the 20s–50s)
- Drop-top-N / bootstrap on the 1764-trade file before treating PF 1.24 as real
- Full IB 15m universe (~1,478) only if we want capacity vs this 300-name slice
