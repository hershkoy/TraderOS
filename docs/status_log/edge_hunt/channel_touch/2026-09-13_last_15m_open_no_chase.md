# Last-15m next-open, no chase if gapped (2026-09-13)

Same-list overlay on last-15m-open-mid + 15m N+1 mid **n=2297 E +0.27 PF 1.09** (geo 1.095; fric0.25 E +0.02 PF 1.01). Occupancy not re-walked. **No promote.**

Clock: last-15m H2 is known at 15:45/16:00 ET. Instead of filling that mid, wait for the **next RTH 09:30 ET**. Gap-up is tick vs the signal session's last 15m **close** (not 15m `gap_15m` = open >= rail on a trigger bar). Next-day resist is the projected rail (`channel_pos` / width + 1 bar of slope).

Pre-registered:

- **chase_moo** — buy the next 09:30 open even if it gapped (control)
- **skip_gap** — MOO only when there is no gap-up and the open still holds the rail
- **pullback** — same MOO, else same-session wait: 50% retrace of (open − rail) (or reclaim if the open is already below), then a green / hammer / engulfing 15m whose **low and close** hold at/above the rail. Fill that bar's **close**. Cancel the gap-up wait if a later 15m closes below the rail. No next-day wait.

Runner: `scripts/research/overlay_last_15m_open_no_chase.py`. Helpers: `utils/research/last_15m_open_no_chase.py`.

## Overnight gaps

On the n=2297 book, next 09:30 vs prior last-15m close:

| | |
|--|--|
| Gap-up share | **55.6%** (1277) |
| Median gap | **+0.13%** |
| Gap > 0.5% / > 1% | 30.7% / 16.1% |
| Gap-down | 42.5% |

Most "gaps" are small overnight ticks. Do not threshold-fish 0.5/1.0; tick gap-up was the pre-registered gate.

## Results

Gross, then friction 0.25. Grade vs last-15m **n=2297 E +0.27 PF 1.09**.

| Book | n | E | PF | geo | fric0.25 E / PF | 2022 n / E / PF | 2022-23 E / PF |
|------|---|----|-----|-----|-----------------|-----------------|----------------|
| Last-15m baseline | 2297 | +0.27 | 1.09 | 1.095 | +0.02 / 1.01 | 117 / **−2.37 / 0.41** | −0.90 / 0.73 |
| **chase_moo** | 2297 | **−0.01** | 1.00 | 1.023 | −0.26 / 0.92 | 117 / **−2.77 / 0.34** | −1.08 / 0.69 |
| **skip_gap** | 553 | **−0.02** | 0.99 | 1.031 | −0.27 / 0.92 | 21 / **−3.16 / 0.34** | −1.56 / 0.56 |
| **pullback** | 1236 | +0.16 | 1.05 | 1.086 | **−0.09 / 0.97** | 69 / **−2.96 / 0.30** | −0.90 / 0.73 |

Year buckets on pullback (gross): 2018-19 E +1.28 PF 1.51; 2020-21 +0.65 / 1.21; 2022-23 **−0.90 / 0.73**; 2024-26 +0.13 / 1.04 (weaker than last-15m +0.48 / 1.16). skip_gap **2024-26 E −0.38 PF 0.87**.

Pullback fill mix: MOO **553** + confirm **683** (green 510 / hammer 109 / engulfing 64). Skips: no_confirm 786, closed_below_rail 275. Confirm-only sleeve n=683 E +0.30 PF 1.10 is not a policy (the user idea is MOO plus wait).

Winner-$ on pullback skips (n_dropped=1061): win$ **3820** vs loser$ saved **3337** (dollar_ratio 1.14, winner_frac 0.32, not the blunt flag). 2024-26 dollar_ratio **1.36**. 2022 leftover n=69 is redder than first-chase.

Smoke: AMPL 2024-02-09 still fills a 15m hammer and still **−21.5%**. VST 2021/2022 gap-up then close below rail (skipped; 2022 VST was a small last-15m winner). TARS 2023-07-20/27 gap-up, no same-session holding confirm.

## Decision

Buying the next open instead of the last-15m mid **removes the 1d edge** (chase_moo E −0.01). Skipping gap-ups without a wait (skip_gap) is worse and turns 2024-26 red. Waiting for a 15m pullback that holds the rail recovers some of the skipped names but still loses E/PF/geo/friction vs last-15m, and 2022 stays a failed-breakout slaughter. **Do not promote. Do not swap `current_best/1d_channel_touch.html`.** Do not occupancy-walk a gap-% cap.

Wall-clock **294s** (IB 15m cache hit 940 names, then 66s policy walk).

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_open_no_chase.py --workers 8
```

Artifacts: `reports/ascending_channels/2026-09-13/last_15m_open_no_chase_{chase_moo,skip_gap,pullback,summary}.csv`.
