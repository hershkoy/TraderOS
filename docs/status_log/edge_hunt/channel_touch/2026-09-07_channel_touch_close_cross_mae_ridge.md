# Close-cross MAE ridge (50 then 300 names, year split)

Diagnostic, not a promote. Confirm-bar features (known when the 15m closes above the daily rail) predict that trade’s `mae_pct` (stop needed to reach trail MFE). Train years pick L2 + a pad on predicted MAE; later years are the test. Stop path: survive to `trail_only_gain_pct` if realized MAE is strictly inside the clipped stop, else `-stop`. Gate is E/PF, not win rate. Same 2018–2026 window as the H2 stack (not a 2006 start).

## CSVs

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --n-symbols 50 --shakeout-breakout --intraday-trigger close-cross --realistic-fill --realistic-fill-mode next-mid --trail-mae --workers 4 --load-workers 8
python scripts\research\backtest_channel_touch_h2_break.py --n-symbols 300 --shakeout-breakout --intraday-trigger close-cross --realistic-fill --realistic-fill-mode next-mid --trail-mae --workers 4 --load-workers 8
python scripts\research\fit_close_cross_mae_regression.py --csv reports\ascending_channels\2026-09-07\channel_touch_close_cross_mae_features_STAMP.csv
```

`--n-symbols` now caps 1d as well as 15m; RDWR is forced into the cap. 50-name list is RDWR plus the next 49 alphabetically from the raw unique set (A-heavy; 8 names had no IB 15m).

| Panel | Feature CSV | IB 15m | Rows / fills / symbols with fills |
|-------|-------------|--------|-----------------------------------|
| 50 names | `reports/ascending_channels/2026-09-07/channel_touch_close_cross_mae_features_20260907_022706.csv` | 42/50 | 1534 / 213 / 36 |
| 300 names | `reports/ascending_channels/2026-09-07/channel_touch_close_cross_mae_features_20260907_023105.csv` | 234/300 | 8031 / 1127 / 212 |

Close-cross unique span365 + ATR k=2 (not the trail-only MAE book): 50-name **n=108 E −0.59 PF 0.81**; 300-name **n=618 E +0.23 PF 1.07**. Do not compare those to hot-cross full-universe n=4079.

## Year split

Train `buy_date` strictly before **2023-01-01**. Inner grid on train uses val year **2022** (max test-stop E, then PF) over `l2` in {0.1, 0.5, 1, 5, 10} and MAE `pad` in {1.0, 1.1, 1.25, 1.5, 2.0}. Predicted stop is clipped to 1.5%–8%. Features are z-scored on train only.

## 50-name holdout (train n=85, test n=128)

Inner 2022 had **n=12** — it picked **l2=0.1 pad=1.25**. Holdout 2023–26:

| Policy | n | E | PF |
|--------|---|---|-----|
| Trail-only (no hard stop) | 128 | −0.79 | 0.82 |
| Ridge stop | 128 | −1.96 | 0.49 |
| Train-median MAE * pad | 128 | −1.86 | 0.50 |
| Fixed 3% stop | 128 | −1.97 | 0.48 |

RMSE train 2.21 / test 3.19 vs intercept 2.83. Coefs (z-scored) were unstable: `close_over_rail_pct` −2.23 vs `open_vs_rail_pct` +2.11.

## 300-name holdout (train n=415, test n=712)

Re-grid on 300-name train: **l2=0.5 pad=2.0**. Frozen 50-name recipe **l2=0.1 pad=1.25** transferred worse.

| Policy | n | E | PF |
|--------|---|---|-----|
| Trail-only (no hard stop) | 712 | **+1.28** | **1.33** |
| Ridge re-grid stop (pad 2.0) | 712 | −0.11 | 0.97 |
| Frozen 50-name stop (pad 1.25) | 712 | −1.19 | 0.69 |
| Train-median MAE * pad | 712 | −0.79 | 0.81 |
| Fixed 3% stop | 712 | −0.89 | 0.77 |

Expanding folds with frozen 300-name l2/pad: 2022 and 2023 stop-path both lose to trail; 2024–26 trail E +1.67 PF 1.44 vs ridge stop E +0.23 PF 1.06. RMSE ~2.66 vs intercept 2.76 — the MAE label is barely linear in confirm features.

Trail-only numbers are occupancy-unfiltered fills (n=1127), not unique-symbol/day. They are the right comparison for a stop overlay on this MAE CSV. They are not a book.

## Verdict

Confirm-bar ridge does **not** produce a dynamic stop that beats unconstrained 10/18 trail or a dumb 3% stop on later years. Pad-2 (almost the clip ceiling) is the least-bad grid point because it rarely fires — and still loses E/PF. Do not wire this into `/hot` or nightly. ATR k=2 (1.5%–6%) stays the keeper exit. 1d `current_best` stays losing hot-cross n=4079 E −0.19 PF 0.94.

Fit artifacts: `reports/ascending_channels/2026-09-07/close_cross_mae_ridge_20260907_022721_*` (50-name), `_023113_*` (300 re-grid), `_023114_*` (300 frozen 50-name recipe).
