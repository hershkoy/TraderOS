# 1d open-cross realistic fill — 2026-09-05

RDWR 2025-06-13 showed the daily clip/next-mid problem: Alpaca daily gapped through resist (O=L=24.65) so the book bought **24.64**, while IB 15m opened **24.37** (still under the rail) and the first open above resist was **09:45 ET**. CTF stamped midnight UTC, which snapped to the 09:30 candle.

## Rule

Daily signal is unchanged (close above resist after H2, shakeout-breakout on). Realistic fill `--realistic-fill-mode open-cross`:

1. On that session's IB 15m, take the first RTH bar whose **open is already above resist**.
2. Fill at that bar's **close** (known when the 15m completes).
3. Last RTH bar is allowed. No next-mid, no wild-bar cancel.
4. Skip the trade if no 15m opens above resist (or no IB 15m).
5. `buy_time` is that 15m bar (UTC). Daily occupancy/exits stay on the daily bar.

Live-plausible: you see the 15m open already through the rail, wait for that bar to close, buy the close. You do not buy a bar that opened below.

RDWR: 09:30 O 24.37 skipped; 09:45 O 25.19 / C **25.71**. Same exit 2025-07-11 @ 28.251, gain **+9.88%** vs HTML next-mid **+14.66%** at 24.64.

## Setup

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --realistic-fill --realistic-fill-mode open-cross --shakeout-breakout --workers 4 --load-workers 8
```

2018-11-01 → 2026-08-27, 2216 ALPACA 1d, IB prefix, window 504/252, unique-symbol/day, span365, 0.25% friction. IB 15m on 1478/2216. Wall-clock **4186s** (1d cache 26s + 15m 246s + scan 3913s).

## Results (gain_pct_net)

| Book | n | E | PF | WR | med | hard_stop |
|------|---|---|----|----|-----|-----------|
| Unique span365 | **2845** | **+2.06** | **1.775** | 38.4 | −2.86 | 1085 |
| Unique parent / extra | 1906 / 939 | — | — | — | — | — |
| RS top1 | 1103 | +1.71 | 1.621 | 37.9 | −2.88 | 425 |

Year buckets (unique): 2018-19 E +2.41 PF 2.10; 2020-21 +1.99 / 1.76; 2022-23 **+1.09 / 1.36**; 2024-26 +2.45 / 1.94. All +.

Vs frozen books (same sleeve, not a same-occupancy remap):

| Fill | Unique n | E | PF |
|------|----------|---|----|
| Close-fill optimistic (2026-09-03 causal) | 3364 | +2.40 | 2.02 |
| Next-mid + shakeout (current_best, pre-`_as_session_date` fix) | 1239 | +2.27 | 1.92 |
| **Open-cross + shakeout (this run)** | **2845** | **+2.06** | **1.775** |

Open-cross keeps more gap-through days than next-mid (no wild cancel) and prices them at the 15m close instead of the daily low / prior-session blend.

CSV: `reports/ascending_channels/2026-09-05/channel_touch_full_h2_break_span365_unique_20260905_203414.csv`.

## Decision

**Do not promote.** E/PF lose the next-mid HTML book (and optimistic close-fill). Nightly stays **close fills**. Do not replace `current_best/1d_channel_touch.html`. `--realistic-fill-mode open-cross` stays a research switch.

The fill itself is the honest gap-through purchaser (RDWR 25.71 @ 09:45 ET, not 24.64 on the 09:30 candle).
