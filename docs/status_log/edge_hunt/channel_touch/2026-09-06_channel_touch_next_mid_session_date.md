# 1d next-mid rebuilt with `_as_session_date` — 2026-09-06

The 2026-09-05 unique next-mid + shakeout HTML (**n=1239 E +2.27 PF 1.92**) joined IB 15m on the **wrong session** (midnight UTC converted to New York). This rescan uses `_as_session_date` (calendar date, no TZ convert).

Nightly still uses the **unrealistic** same-bar rail clip. This is a research overlay, **not** `current_best`.

## Setup

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --realistic-fill --realistic-fill-mode next-mid --shakeout-breakout --workers 4 --load-workers 8
```

2018-11-01 → 2026-08-27, 2216 ALPACA 1d, IB prefix, window 504/252, unique-symbol/day, span365, wait-6, shakeout-breakout, ATR k=2, 0.25% friction. IB 15m on 1478/2216. Wall-clock **3855s** (1d cache 19s + 15m 235s + scan 3600s).

## Results (`gain_pct_net`)

| Book | n | E | PF | WR | med | hard_stop |
|------|---|---|----|-----|-----|-----------|
| Unique span365 | **1986** | **+2.24** | **1.893** | 37.6 | −2.74 | 744 |
| Unique parent / extra | 1247 / 739 | — | — | — | — | — |
| RS top1 | 916 | +2.09 | 1.805 | 37.6 | −2.71 | 338 |
| Cap 2/day by wait (post-filter) | 1395 | +2.29 | 1.908 | — | — | — |

Year buckets (unique): 2018-19 E +2.19 PF 2.06; 2020-21 +1.97 / 1.78; 2022-23 +1.62 / 1.56; 2024-26 +2.70 / 2.14. All +.

Drop-top-3: n=1983 E **+2.02** PF **1.805**.

Vs frozen books (same sleeve, not a same-occupancy remap):

| Fill | Unique n | E | PF |
|------|----------|---|----|
| Close-fill unrealistic rail-clip + shakeout | 4558 | +2.62 | 2.10 |
| Next-mid + shakeout **pre-`_as_session_date`** | 1239 | +2.27 | 1.92 |
| **Next-mid + shakeout (this run)** | **1986** | **+2.24** | **1.893** |
| Open-cross + shakeout | 2845 | +2.06 | 1.775 |
| Hot-cross lerp85 (no EOD close) | 4079 | −0.19 | 0.94 |

Mondays and correct-session 15m **raise occupancy** vs the pre-fix book; E/PF stay about the same. The crowding cap-2 lift to n=933 E +3.01 PF 2.25 was a **pre-fix artifact** — honest cap-2 is n=1395 E +2.29 PF 1.91 (barely above uncapped). Do not wire cap-2 from the old numbers.

## RDWR

Raw next-mid has **no 2025-06-13** row (honest Friday wild-cancel, as predicted). Unique span365 has **no RDWR** (the remaining RDWR raw fills are span>365). CTF BUY **24.64** was the wrong-session Thursday blend, not a live Friday fill.

## Decision

Treat **n=1986 E +2.24 PF 1.89** as the honest next-mid + shakeout unique book. Do **not** copy into `current_best/`. Nightly stays the clip. `1d_unrealistic/1d_channel_touch.html` is still the pre-fix overlay until regenerated from this CSV. Cap-2 wait is no longer a soft-promote on the old +3.01 / 2.25 figures.

CSV: `reports/ascending_channels/2026-09-06/channel_touch_full_h2_break_span365_unique_20260906_065735.csv`.
