# 1d H2 close-confirm + formation containment (2026-09-07)

Not a promote. Follow-up to the ADM 2021-11-23 review: many "channels" are mathematically parallel but not classically contained (price spiked far above the eventual rail before H2).

## What changed

1. **`formation_beyond_width`** — max `(high - resist) / width` from L1 through H2 (formation only). Detector unchanged; post-filter `--max-formation-beyond-width`.
2. **Close-cross fill modes** — `purchase_close_cross_15m(..., fill_mode=...)`:
   - `signal-close` = buy the **confirm 15m close** (last RTH allowed)
   - `next-mid` / `next-open` = prior behavior (15:45 confirm cancels)
3. Feature CSV always written for `--intraday-trigger close-cross` (confirm-bar + stock entry features).

## ADM smoke (proof)

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --symbols ADM --shakeout-breakout --intraday-trigger close-cross --realistic-fill --realistic-fill-mode signal-close --max-formation-beyond-width 0.25 --workers 1
```

| Date | Confirm ET | Fill (confirm close) | formation_beyond | Kept @ 0.25? |
|------|------------|----------------------|------------------|--------------|
| 2021-04-27 | 10:30 | 62.44 | high | **dropped** |
| **2021-11-23** | **14:00** | **66.96** | **1.8765** | **dropped** |
| 2022-01-04 | 09:30 | 68.70 | high | **dropped** |
| 2022-03-02 | 10:15 | 79.80 | ok | kept |
| 2026-01-15 | 14:15 | 66.34 | high | dropped |
| 2026-04-30 | 10:30 | 75.34 | ok | kept |

Nov 23 is a real H2 close-confirm (15m closed above the daily rail), **not** hot-cross. Formation overshoot ~1.88× width (May/Jun 2021 spikes) correctly kills it.

Artifacts: `reports/ascending_channels/2026-09-07/channel_touch_close_cross_features_20260907_234339.csv`, `channel_touch_h2_resist_break_20260907_234339.csv`.

## Full-universe run

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --intraday-trigger close-cross --realistic-fill --realistic-fill-mode signal-close --max-formation-beyond-width 0.25 --workers 4 --load-workers 8
```

Wall-clock **~1425s** (~24m). IB 15m panels 1478/2216. Log: `logs/scanners/channel_touch_h2_close_cross_form025_20260907.log`.

| Book | n | E % | PF | Notes |
|------|---|-----|-----|-------|
| Raw resist-break (confirm-close) | 6666 | +0.04 | 1.01 | ~55% hard-stop |
| + formation≤0.25 | 2790 | +0.21 | 1.07 | drops ~58% of raw |
| + span≤365 + form (unique) | **1947** | **+0.24** | **1.07** | 2022-23 **E −1.07 PF 0.70** |
| + RS top1 | 999 | +0.28 | 1.08 | optional |

Year buckets on unique span365+form: 2018-19 +0.37/1.12; 2020-21 +0.55/1.17; **2022-23 −1.07/0.70**; 2024-26 +0.61/1.19.

Features: `reports/ascending_channels/2026-09-08/channel_touch_close_cross_features_20260908_000732.csv`. Trades: `channel_touch_full_h2_break_span365_unique_20260908_000734.csv`.

Versus `current_best` hot-cross **n=4079 E −0.19 PF 0.94**: slightly better on E/PF and smaller, but still far from a promote (flat overall, 2022-23 fails, median trade negative, ~56% hard-stop).

**No promote.** Formation filter is still useful quality hygiene (kills ADM-class non-channels).
