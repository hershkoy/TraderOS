# 1d H2 A/B: rail tick vs 1.2% buffer, next-open vs same-bar — 2026-09-05

RDWR Jun 9 2025 closed **above the painted rail** (+0.69%) but below the H2 break rule (`close > resist * 1.012`). Question: drop the 1.2% buffer, and/or fill at the **next session open** after an EOD close (nightly-honest MOO).

Detector **pivot fitting stays `error_pct=1.2`**. Break/tag buffer is `--touch-error-pct` (default = 1.2). Fill `--realistic-fill-mode next-open` buys the next daily **open** (no IB 15m join). Occupancy and ATR/trail start on that next bar; same-day stop after the open is allowed.

## 2x2 (unique span365 + shakeout, 0.25% friction)

2018-11-01 → 2026-08-27, 2216 ALPACA 1d, IB prefix, window 504/252. Wall-clock **289s** (first scan 104s cache miss, then ~62s each).

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --workers 4 --load-workers 8
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --touch-error-pct 0 --workers 4 --load-workers 8
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --realistic-fill --realistic-fill-mode next-open --workers 4 --load-workers 8
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --realistic-fill --realistic-fill-mode next-open --touch-error-pct 0 --workers 4 --load-workers 8
```

Or `scripts\research\run_ab_h2_next_open.bat`.

| Cell | Unique n | E | PF | WR | med | hard_stop | parent / extra |
|------|----------|---|----|----|-----|-----------|----------------|
| **B12 close-fill** (nightly path) | **4558** | **+2.62** | **2.098** | 42.0 | −1.98 | 1558 | 3119 / 1439 |
| B0 close-fill | 5649 | +1.38 | 1.492 | 34.7 | −3.18 | 2560 | 3850 / 1799 |
| B12 next-open | 4894 | +0.45 | 1.141 | 30.0 | −3.73 | 2577 | 3171 / 1723 |
| B0 next-open | 5804 | +0.36 | 1.114 | 29.4 | −3.72 | 3104 | 3854 / 1950 |

B12 close-fill matches the 2026-09-05 shakeout combined book. All year buckets stay + only on that cell. B12 next-open **2020-21 E −0.07 PF 0.98**.

RS top1 (optional): B12 close n=1316 E +2.83 PF 2.17; B0 close +1.71 / 1.60; B12 next-open +0.56 / 1.17; B0 next-open +0.80 / 1.25.

CSVs (`reports/ascending_channels/2026-09-05/` unique):

- B12 close: `channel_touch_full_h2_break_span365_unique_20260905_230923.csv`
- B0 close: `..._231025.csv`
- B12 next-open: `..._231128.csv`
- B0 next-open: `..._231230.csv`

## RDWR

| Cell | Fill | Net |
|------|------|-----|
| B12 close | 2025-06-13 @ **24.65** (same-bar rail clip) | +14.36% |
| B0 close | 2025-06-09 @ **24.46** (tick-above on the poke day) | +15.24% |
| B12 next-open | 2025-06-16 @ **27.36** (Monday MOO after Fri 26.58 close) | +3.01% |
| B0 next-open | 2025-06-10 @ **24.525** (user's next-open after Jun 9 close) | +14.94% |

Jun 9 next-open works **on this name**. Tick-above + next-open adds ~1k weaker breaks (failed pokes) and next-open after a real 1.2% gap (Jun 13→16) pays 27.36 instead of 24.65.

## Decision

**Do not promote.** Keep `--touch-error-pct` default 1.2 on daily H2. Nightly stays **close fills** (same-bar rail clip) + shakeout. Do not replace `current_best/1d_channel_touch.html`. `--realistic-fill-mode next-open` stays a research switch.
