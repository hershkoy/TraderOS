# 1d buy-now hot-cross (15m high >= daily rail) — 2026-09-06

Live model: arm on daily H2 from **prior** completed bars (`causal_h2`). Buy during RTH when last crosses the daily rail. Do **not** wait for that session's daily close. `/hot` analog is armed + Alpaca last. `open-cross` is the wrong live model when the 15m **opens under** the rail then trades through.

`--intraday-trigger hot-cross` is **not** `--realistic-fill`. It changes **when** the buy exists (first 15m `high >= resist` after wait), not a post-EOD reprice of the clip.

## Fill (A0 lerp85)

On the trigger 15m: `fill = rail + 0.85 * (close - rail)` if close >= rail, else rail; clamp to `[low, high]`. Bounds A3=rail and A1=15m close were not full-rescanned (ATR stop depends on entry, but gap remaps already sat at PF ~0.92–0.95).

A **15m gap** is trigger `open >= rail` (not a daily gap). RDWR 2025-06-13 Alpaca 1d O=L=24.65 looks like a daily gap; IB 15m 09:30 ET still opened **24.35** under rail **24.64** (`gap_15m=False`). 5m is not the universe fill clock.

## Setup

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --intraday-trigger hot-cross --hot-cross-fill lerp85 --shakeout-breakout --workers 4 --load-workers 8
```

2018-11-01 → 2026-08-27, 2216 ALPACA 1d, IB prefix, window 504/252, unique-symbol/day, span365, wait-6, shakeout-breakout, ATR k=2, 0.25% friction. IB 15m on 1478/2216 (738 daily names skipped). First hung run recoerced the full 15m history every wait day; `index_rth_15m_by_session` once per symbol. Wall-clock **596.5s** (1d cache 19s + 15m 237s + scan 338s).

## Results (`gain_pct_net`)

| Book | n | E | PF | WR | med | hard_stop |
|------|---|---|----|-----|-----|-----------|
| Raw resist-break (no span) | 6997 | −0.21 | 0.938 | 26.0 | −3.97 | 4076 |
| Unique span365 | **4079** | **−0.19** | **0.944** | 25.8 | −3.97 | 2430 |
| Unique parent / extra | 3328 / 751 | — | — | — | — | — |
| RS top1 | 1342 | −0.05 | 0.985 | 26.5 | −4.37 | 800 |

Year buckets (unique): 2018-19 E −0.55 PF 0.82; 2020-21 +0.02 / 1.01; 2022-23 **−0.90 / 0.75**; 2024-26 +0.04 / 1.01. Not all +.

Drop-top-3: n=4076 E **−0.28** PF **0.917** (tails were carrying a flat book, not hiding an edge).

Gap splits on the unique book (same occupancy; G-gap-open is same-exit approx):

| Split | n | E | PF |
|-------|---|---|----|
| G-under (15m open < rail) | 3265 | −0.18 | 0.946 |
| G-gap lerp (open >= rail) | 814 | −0.24 | 0.935 |
| G-skip (drop 15m gaps) | 3265 | −0.18 | 0.946 |
| G-gap-open (gaps fill at open) | 4079 | −0.27 | 0.922 |
| G-wild skip (open already >1% through rail) | 3650 | −0.21 | 0.938 |

Without the daily close-above-resist gate the book buys every armed name that **tags** the rail on a 15m high. Hard-stop rate is ~60% (2430/4079). Fill-price tweaks cannot turn PF 0.94 into ~1.5.

## RDWR

Unique (and raw) have **no 2025-06-13** row. Occupancy from **2025-06-05** 10:30 ET: wait=16, rail 24.42, 15m O 24.21 / H 24.46 / C 24.41, fill **24.42** (`gap_15m=False`), trail exit **2025-07-11** (same exit as the clip book's Jun-13 trade). That is an intra-day rail poke, not the EOD close-above-resist day. Clip/CTF BUY **24.64** on Jun-13 never prints here because the position is already open.

Other unique RDWR rows: 2024-10-14 09:30 ET 24.09 (stop); 2024-10-21 shakeout 24.20 (stop); 2024-12-02 15:45 ET 24.76 (stop); 2025-07-31 09:30 ET gap-open 26.40 (stop).

## Decision

**Do not promote.** Live-executable, but E < 0, PF < 1, 2022-23 fails, drop-top-3 worse. Do **not** grade vs clip n=4558 E +2.62 PF 2.10. Nightly stays the **unrealistic** same-bar rail clip. Do not wire `/hot` last-cross as a *winning* 1d book. Skip cap-2 `wait_bars` and A3/A1 full rescans.

**`current_best/` 1d slot (2026-09-06 evening):** this book is the live-executable baseline even though it loses. HTML: `reports/ascending_channels/current_best/1d_hot_cross.html` (same bytes as `1d_channel_touch.html`). That is the number to beat. See [current_best copy](2026-09-06_channel_touch_current_best_hot_cross.md).

CSV: `reports/ascending_channels/2026-09-06/channel_touch_full_h2_break_span365_unique_20260906_055000.csv`.
