# 1d H2 unique book — regression filter for losing trades (2026-09-05)

Yes, **entry features are already collected** (`entry_features=True` on the H2 scan). The current_best unique CSV has RSI, SMA distance, %B, close_loc, volume, squeeze, geometry, wait, and RS. SPY regime columns were missing on that file; this run attached **prior-session** SPY 20d return / SMA50 / ATR%%.

## Leak

`feature_asof` is the **fill-day daily bar** (`2019-06-07 00:00` with no `buy_time`). Current_best 1d is **next-mid**, so that session's close / RSI / %B / range / SMA distance / same-day volume are not known at the 15m fill. Honest features are rails, wait, calendar, extra-vs-parent, and prior-session SPY.

Nightly close-fills *can* use the signal day's close. That is a different book; this filter was scored on the HTML next-mid unique file.

## Setup

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\filter_channel_touch_losers.py
```

Book: `channel_touch_full_h2_break_span365_unique_20260905_135126.csv` **n=1239 E +2.27 PF 1.92** (extras 584). Expanding yearly OOS after the 60% buy-date quantile (first fold **2024-02-08**), purge + 21d embargo. Holdout cutoff **2023-01-01**. Ridge predicts `gain_pct_net` (winsorized train y); logistic predicts P(win). `skip0` = pred>=0 or p>=0.5; `thr` = train-only max-PF quantile (keep at least 35%).

## Univariate

All |Spearman vs gain| **< 0.07**. Month +0.06, width −0.05, ATR/squeeze_mom −0.06. `is_extra` is ~0 (mean E still differs; rank correlation on a fat tail is not the mean). There is no strong linear ranking of losers vs winners at entry.

## Expanding WF (2024-02 → 2026-08, n_oos=497, all E +2.11 PF 1.84)

| set | model | rule | kept n | kept E | kept PF | spearman | auto-verdict |
|-----|-------|------|--------|--------|---------|----------|--------------|
| honest_geom | ridge | skip0 | 338 | +2.23 | 1.90 | +0.045 | research_only |
| honest_geom | ridge | thr | 143 | +3.72 | 2.71 | +0.045 | research_only |
| honest_geom | logistic | skip0/thr | 489 / 351 | +2.03 / +1.80 | 1.81 / 1.72 | +0.11 | no_promote (drops the better trades) |
| honest_full | ridge | thr | 131 | +2.76 | 2.09 | **−0.016** | research_only |
| leaky_close | all | | | | | ~0.03–0.05 | no lift vs honest; logistic thr **worse** |

2024 fold for honest_geom ridge skip0 is **worse** (kept E +0.71 vs +0.80). Ridge thr 2024 drop-top-3 PF **0.97**. Expanding OOS never sees 2018–2023.

## Holdout 2023-01-01 (train 553 / test 686, all E +2.96 PF 2.24)

| set | model | rule | kept n | kept E | kept PF |
|-----|-------|------|--------|--------|---------|
| honest_geom | ridge | skip0 | 451 | **+3.21** | **2.36** |
| honest_geom | ridge | thr | 221 | +2.91 | 2.29 |
| honest_full | ridge | skip0 | 457 | +3.32 | 2.43 |
| leaky_close | logistic | thr | 233 | +2.20 | 1.95 |

The late expanding-WF ridge-thr spike does **not** reproduce on 2023+. Skip0's holdout lift is small and the first expanding year goes the other way. Fill-day close features do not help (and can hurt).

## Gate

Need OOS kept to beat equal-dollar on **E and PF**, more than one year including 2020–21 or a 2023 holdout, drop-top-3 PF>1, Spearman not noise.

**No promote.** Do not replace `current_best/1d_channel_touch.html`. Do not wire a score gate into nightly. Same conclusion as the 2026-08-29 ridge *sizing* study and the 15m logistic/MLP: the snapshot does not rank this fat-tailed book well enough to skip losers.

## Code

- [`utils/research/channel_touch_entry_model.py`](../../../utils/research/channel_touch_entry_model.py): `HONEST_1D_*`, `fit_keep_fold`, `expanding_keep_walk_forward`
- [`scripts/research/filter_channel_touch_losers.py`](../../../scripts/research/filter_channel_touch_losers.py)
- Tests: `tests/unit/test_channel_touch_entry_model.py`

## Artifacts

- `reports/ascending_channels/2026-09-05/channel_touch_1d_loser_filter_*_20260905_181221.csv`
