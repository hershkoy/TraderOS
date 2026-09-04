# 15m L3: real support touch (0%) vs 0.24% near-miss

`--preset 15m` sets detector `error_pct=0.24` (daily 1.2 / sqrt(26)). That same tolerance was also used for L3 tags: a wick counted if the low came within 0.24% of support without necessarily intersecting the rail.

New CLI `--touch-error-pct` separates fill/tag tolerance from detector pivot fitting (detector stays 0.24). `--touch-error-pct 0` requires `low <= support <= high` (true intersection). Purchaser is **signal-bar close** (`--realistic-fill`, default mode).

## Stack

`--preset 15m --n-symbols 300 --entry-mode l3_touch --min-l3-wait-bars 12 --realistic-fill --realistic-fill-mode signal-close --workers 4 --load-workers 8`

Causal H2, in-channel, span≤10d, RS top1, friction 0.10, 2018-11-01 → 2025-12-02. Detector `error_pct` unchanged at 0.24.

## Wall-clock

| Book | Cache | Scan+trade | Elapsed |
|------|-------|------------|---------|
| touch 0.24 (default) | 1.1s | 266s (raw 51774) | ~439s |
| touch 0.0 | 1.1s | 245s (raw 41325) | ~408s |

Raw tags drop ~20% when near-misses are rejected; RS top1 n stays ~1750 (occupancy reshuffles which names win the day).

## Results (RS top1, 0.10% friction)

| Book | n | WR% | E% | PF | 2018-19 E / PF |
|------|---|-----|-----|-----|----------------|
| Next-mid wait-12 (frozen `current_best`) | 1746 | — | **+0.09** | **1.16** | all + on optimistic era |
| Signal-close + touch **0.24** | 1755 | 29.3 | **+0.05** | **1.08** | **−0.14 / 0.76** |
| Signal-close + touch **0.0** | 1744 | 30.0 | **+0.13** | **1.22** | **+0.13 / 1.22** |

Year buckets touch 0.0: 2018-19 +0.13, 2020-21 +0.20, 2022-23 +0.08, 2024-26 +0.13 (all +). Touch 0.24 fails 2018-19.

Beyond-width A/B on touch 0.0 (still no hard promote of daily 0.25 onto 15m): off n=1744 E +0.13 PF 1.22; 0.25 n=1654 E +0.17 PF 1.29.

## Verdict

The 0.24% near-miss **was a soft assumption**, and requiring a **real rail intersection** lifts wait-12 under signal-close enough to beat the frozen next-mid book on E/PF and repair 2018-19. Prefer `--touch-error-pct 0` for 15m L3 research going forward. Do **not** set `--error-pct 0` (that would tighten detector pivot fitting).

**`current_best/15m_channel_touch.html` replaced 2026-09-04** with signal-close + touch 0% (IB SPY overlay from 2018-11-06; trade times RTH). Stamp: `...15m_l3_wait12_touch0_signal_close_20260904_172151.html`. Stable trades: `reports/ascending_channels/channel_touch_15m_l3_wait12.csv`.

## Artifacts

- Touch 0.24: `reports/ascending_channels/2026-09-04/channel_touch_15m_trades_20260904_165626.csv`
- Touch 0.0: `reports/ascending_channels/2026-09-04/channel_touch_15m_trades_20260904_170332.csv`
- HTML: `reports/ascending_channels/current_best/15m_channel_touch.html`
