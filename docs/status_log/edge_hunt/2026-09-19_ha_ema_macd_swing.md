# Animesh / Dhan HA + 55 EMA + MACD swing (2026-09-19)

Claim-check of the Dhan masterclass swing system ([2YBmiyVmNNw](https://www.youtube.com/watch?v=2YBmiyVmNNw)). YouTube transcript was unavailable; rules follow the published summary. **No promote.**

Logic: `utils/research/ha_ema_macd_swing.py`. Runner: `scripts/research/run_ha_ema_macd_swing.py`. Strategy note: [ha_ema_macd_swing](../../strategies/ha_ema_macd_swing/README.md).

## Encoding (causal)

Daily (weekend-prep analog). Signals on **completed Heikin-Ashi**; fills/stops on **regular OHLC**.

- EMA(55) of High and EMA(55) of Low. Skip while HA close is inside the band; arm when HA close **breaks above** EMA High after a prior HA close at/below that rail.
- Bullish HA color flip (prior red, signal green).
- MACD(12,26,9) on HA close must be **> 0**. Same-bar zero-cross is a named variant, not the default.
- Fill **next open**. Stop = signal-bar EMA Low. Target = 3R. Gap through the stop fills and exits at the open.
- Exits (first hit): stop, 3R, **Friday close**, MACD line cross below signal (next open). Same-bar stop beats target.

Video-named variants only: rr=4, no Friday exit, MACD zero-cross, Friday-weekly resample (Friday exit off — the bar *is* Friday).

## Sample

ALPACA 1d panel **2217** names, 2017-01-03 warmup, eval **2018-01-01 → 2026-09-18**. PIT top-500 ADV, min price $10, min ADV $1M. Panel shape **1546 x 2217** (Alpaca coverage is ~2020+; **2018-19 n=0**). SPY IB buy-and-hold Sharpe **0.67** MDD **−34.1%**. Wall **320s**.

TimescaleDB examples before the universe run: SPY 2021-03-08 HA break → 2021-03-09 385.84 → Friday 394 (+2.11%); NVDA 2022-11-04 → 11-07 142.10 → Friday 163.31 (+14.9%); A 2021-03-01 → next-open stop −3.0%.

## Results

| Book | n | E | PF | geo | WR | median | hold | max conc |
|------|---|---|----|-----|----|--------|------|----------|
| Daily unconstrained (primary) | **4047** | **+0.23** | **1.20** | **1.26** | 50% | +0.01 | 1d | 47 |
| Daily occ-15 | 3483 | +0.21 | 1.18 | 1.23 | 50% | 0.00 | 1d | 15 |
| Daily rr=4 | 4046 | +0.23 | 1.20 | 1.25 | 50% | 0.00 | 1d | 47 |
| Daily no-Friday | 3741 | +0.40 | 1.18 | 1.22 | 37% | **−1.40** | 7d | 103 |
| Daily MACD zero-cross | 349 | +0.56 | 1.56 | 1.67 | 55% | +0.31 | 1d | 7 |
| Weekly unconstrained | 634 | +2.64 | 1.55 | 1.97 | 40% | **−2.79** | 7w | 57 |
| Weekly occ-15 | 263 | +2.22 | 1.50 | — | 39% | −2.30 | 7w | 15 |

Primary year buckets (geo skips n<30, so **only 2022-23 and 2024-26**):

| Bucket | n | E | PF |
|--------|---|---|----|
| 2018-2019 | 0 | — | — |
| 2020-2021 | 24 | +0.21 | 1.23 |
| 2022-2023 | 1362 | +0.47 | 1.45 |
| 2024-2026 | 2661 | +0.11 | 1.09 |
| FULL | 4047 | +0.23 | 1.20 |

Calendar 2022 n=356 E +0.96 PF 1.88; 2023 n=1006 E +0.30 PF 1.29. Drop-top-3 still +0.21 / 1.18. Friction **0.25** E **−0.02 PF 0.99**.

Occupancy-15 equal-dollar MTM vs SPY: total **+48%** Sharpe **0.51** MDD **−11.4%** invested 52% (SPY Sharpe 0.67 / MDD −34%). Fails Sharpe > 1.0 and Sharpe ≥ SPY.

Primary exits: **Friday 3042 / 4047 (75%)**, stop 793, target 89, MACD 66, gap_stop 39. Realized payoff **1.17** not 3:1 — Friday clips the swing. Inverse/bond ETFs (SHY/VXX/UVXY/…) are only **1.2%** of fills.

Weekly looks stronger on pooled E/PF but is fat-tailed (median −2.79) and the 2022-23 geo bucket is **2023 n=97 E +7.68 carrying 2022 n=2 E −9.90**. Do not promote.

MACD same-bar zero-cross is a small-n overlay (2022 n=27); do not fish it.

## Verdict

**No promote. Do not wire live / `/hot`.**

Daily with the video's Friday exit is a 1-day hold, not a swing. Gross PF 1.20 dies at 25 bps. Occupancy Sharpe loses to SPY. Weekly resample is a 2023 leftover, not a robust weekend system.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\run_ha_ema_macd_swing.py --examples-only
python scripts\research\run_ha_ema_macd_swing.py --weekly
```

Artifacts: `reports/research/ha_ema_macd_swing/2026-09-19/`.
