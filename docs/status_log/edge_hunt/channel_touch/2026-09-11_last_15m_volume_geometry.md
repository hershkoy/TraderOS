# Last-15m volume-delta / geometry overlays (2026-09-11)

Same-list A/B (occupancy not re-walked) on the LAUR-fixed last-15m-open-mid + 15m N+1 mid book **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01; geo-mean year PF 1.095). HTML `…165800.html`. **No promote.** Does not beat hot-cross n=4079 E −0.19 PF 0.94 as a live 1d clock. 2022-23 stays negative on every overlay.

This is **session volume-delta** (15m close-in-range buy/sell, summed over RTH), not classical POC/VAH/VAL profile.

## Smoke (AMPL / VST / TARS)

| Name | Fill | pos | form | Overlay that matches the chart |
|------|------|-----|------|--------------------------------|
| **AMPL** 2024-02-09 | 14.17, sell 11.15 (−21.3%) | 1.03 | 0.04 | Near-rail; real channel. Textbook Evening Doji Star does **not** print. Consecutive 15m-sum `sell_pct>=55%` does **not** print before the Feb 21 gap (14.07 → 9.22). ATR already sells 09:45 ET. Daily Volume Delta on the screenshot is not the 15m-sum. |
| **VST** 2021-02-18 | 23.20 (−19.6%) | 1.007 | **1.82** | June 2020 peak. `formation_beyond_width` 1.82. Skip. |
| **TARS** 2023-07-20 | 23.24 (−19.4%) | **1.492** | 0.37 | Extended first print *and* formation 0.37. Drops at pos 1.25; **kept** at 1.50. Form 0.25 also drops it, so delayed 2nd-close (C, form-ok only) never sees it. |

## Overlay A — early exit

Pre-registered: Evening Doji Star (fill day may be candle 1; gap-up required) **or** two consecutive post-fill RTH sessions with 15m-sum sell_pct >= 55%. Decision at that session close; fill next 15m mid if strictly earlier than the book's ATR/trail.

| | n | E | PF | geo year PF | WR | 2022-23 E/PF |
|--|---|---|----|-------------|----|--------------|
| Baseline | 2297 | +0.27 | 1.086 | 1.095 | 31.4 | −0.90 / 0.73 |
| **A** | 2297 | +0.46 | 1.192 | **1.298** | 40.6 | **−0.27 / 0.90** |

Used 1119 / 2297 (doji_star 41, seller_sessions 1078). Winner-cut: 80 winners flip to losers; 292 losers flip to winners; winner $ **−2129** vs loser $ **+2569**. Fat winners get clipped; 2020-21 E falls +0.46 → +0.23. 2022-23 still fails. Occupancy not re-walked (would free slots earlier). **No promote.**

## Overlay B — skip

| Overlay | n | E | PF | geo | winners dropped | losers dropped | winner $ / loser $ saved |
|---------|---|---|----|-----|-----------------|----------------|--------------------------|
| pos <= 1.25 | 2161 | +0.29 | 1.095 | 1.110 | 43 | 93 | 442 / 465 |
| pos <= 1.50 | 2248 | +0.31 | 1.100 | 1.106 | 17 | 32 | 93 / 181 |
| form <= 0.25 | 1267 | +0.13 | 1.043 | 1.025 | 325 | 705 | **3664 / 3223** |
| form 0.25 + pos 1.25 | 1199 | +0.14 | 1.046 | 1.026 | 348 | 750 | 3885 / 3443 |

Pos 1.25 drops TARS Jul-20; 1.50 keeps it. Form 0.25 drops VST 2021 and **costs more winner dollars than it saves** (same pattern as close-confirm form 0.25: hygiene, not an edge). 2022-23 fails on every row. **No promote.**

## Overlay C — delayed 2nd close

Forced on form-ok names skipped by pos 1.25: wait for a close at/below the rail, then a 2nd close above with session buy_pct >= 50. Occupancy not re-walked.

13 delayed fills (55 no-fill skips). Delayed sleeve E **−2.82**. Combined n=1212 E +0.11 PF 1.035 vs form+pos1.25 skip n=1199 E +0.14. TARS Jul-20 is form 0.37 so it never enters C. **No promote.**

## Occupancy

Skipped. A lifts geo-mean PF but wrecks winner dollars. B form hurts E/PF. B pos is a small pooled bump with 2022-23 still red. A full unique H2 rescan would not change the promote gate.

Wall-clock: first 15m load 940 symbols **451s** TimescaleDB (cache miss), overlay apply ~300s; cache-hot reload 3.4s.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_volume_geometry.py --symbols AMPL,VST,TARS
python scripts\research\overlay_last_15m_volume_geometry.py --workers 8 --delayed-second-close
```

Artifacts: `reports/ascending_channels/2026-09-11/last_15m_overlay_A_early_exit.csv`, `last_15m_overlay_B_skip_summary.csv`, `last_15m_overlay_C_delayed.csv`.
