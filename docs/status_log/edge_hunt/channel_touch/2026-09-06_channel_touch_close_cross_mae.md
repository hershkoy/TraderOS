# 1d close-cross + trail MAE (RDWR microscope, 2026-09-06)

Diagnostic, not a promote. `--intraday-trigger close-cross`: 1d H2 arms on prior completed daily bars; first RTH 15m **close > daily rail**; fill is the **next** 15m mid. `--trail-mae` walks 15m from the fill with **no hard stop**: 10% trail, **18% when 15m volume > the prior 20-bar mean**.

The TV RDWR boxes are the report columns:

- **max_profit** = MFE (highest 15m high vs fill) while that trail kept the trade open (the ~20% ruler). Also `max_profit_atr_15m` / `max_profit_atr_1d`.
- **mae** = min 15m low from fill through the MFE bar — the stop you needed to *reach* that peak (the ~4.8% red box). Also `mae_atr_15m` / `mae_atr_1d`.
- **trail_only_gain_pct** = realized exit after giving back the trail width (peak 29% with a 10% trail exits ~+16%).

## Feature CSV

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --symbols RDWR --shakeout-breakout --intraday-trigger close-cross --realistic-fill --realistic-fill-mode next-mid --trail-mae --workers 1
```

**`reports/ascending_channels/2026-09-07/channel_touch_close_cross_mae_features_20260907_021740.csv`** — confirm-bar features plus the MFE/MAE columns. Occupancy and `wild_low_to_mid` skip rows are included (`skip_reason` set; outcome cols empty).

RDWR fills (occupancy-honest later span365 unique is 3 of these 6):

| confirm ET | fill px | max_profit % | max_profit ATR 1d | MAE % | MAE ATR 1d | trail exit |
|------------|---------|--------------|-------------------|-------|------------|------------|
| 2020-02-11 10:00 | 26.62 | 0.45 | 0.29 | 0.53 | 0.34 | trail −9.59 |
| 2021-08-18 14:15 | 33.12 | 20.11 | 9.25 | 2.63 | 1.21 | trail +8.10 |
| 2024-10-14 10:00 | 24.05 | 1.23 | 0.62 | 7.13 | 3.62 | wide −16.99 |
| 2025-06-05 11:15 | 24.41 | 29.37 | 14.19 | 1.93 | 0.93 | trail +16.44 |
| 2025-09-18 11:00 | 26.93 | 3.99 | 2.02 | 2.51 | 1.27 | trail −6.41 |
| 2026-05-15 13:30 | 28.03 | 13.88 | 4.34 | 1.89 | 0.59 | trail +2.49 |

**RDWR 2025-06-13 never prints** (Jun-5 already filled). The chart ruler ~20% / ~4.8% is a hand-picked entry; this fill’s peak while the 10/18 trail held is **+29.4%** with MAE **1.93%** to get there (trail gives back to +16.4%).

No promote. No volume→stop formula from n=6.

Follow-on 2026-09-07: 50-name then 300-name CSVs + year-split ridge — still no promote. See [MAE ridge](2026-09-07_channel_touch_close_cross_mae_ridge.md).
