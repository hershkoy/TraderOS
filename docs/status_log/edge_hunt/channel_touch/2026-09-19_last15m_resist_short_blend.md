# Last-15m H2 + 15m short H3+ occupancy blend (2026-09-19)

Hypothesis: portfolio-overlay the `current_best` last-15m long H2 breakout with the 15m short H3+ resist fade so shorts patch 2022. Not a same-channel long-then-short switch.

Harness: generalized [`utils/research/h2_l3_blend_occupancy.py`](../../../utils/research/h2_l3_blend_occupancy.py) (`SLEEVE_SHORT`, `a_cap`/`b_cap`); runner [`scripts/research/overlay_last15m_resist_short.py`](../../../scripts/research/overlay_last15m_resist_short.py).

Inputs: `reports/ascending_channels/channel_touch_1d_last_15m.csv` (n=2297) + `2026-09-16/channel_touch_resist_short_unique_20260916_211434.csv` (n=7004).

## Research frictions (long 0.00 / short 0.10)

| Book | n | long/short | E | PF | geo | 2022 E/PF | 2023 E/PF | 2022-23 E/PF |
|------|---|------------|---|----|-----|-----------|-----------|--------------|
| Long alone | 2297 | 2297/0 | +0.27 | 1.09 | 1.10 | **−2.37 / 0.41** | −0.33 / 0.90 | −0.90 / 0.73 |
| Short alone | 7004 | 0/7004 | +0.21 | 1.47 | 1.44 | +0.26 / 1.57 | +0.20 / 1.45 | +0.23 / 1.52 |
| Concat no occ | 9301 | 2297/7004 | +0.22 | 1.20 | 1.22 | −0.00 / 1.00 | +0.07 / 1.06 | +0.03 / 1.03 |
| Unique nocap | 9137 | 2286/6851 | +0.23 | 1.20 | 1.22 | −0.01 / 0.99 | +0.08 / 1.07 | +0.04 / 1.04 |
| UNIFIED 8 | 3225 | 546/2679 | +0.27 | 1.28 | 1.28 | −0.00 / 1.00 | +0.17 / 1.16 | +0.06 / 1.06 |
| UNIFIED 15 | 4848 | 907/3941 | +0.37 | 1.39 | 1.42 | −0.01 / 0.98 | +0.24 / 1.22 | +0.08 / 1.09 |
| UNIFIED 20 | 5723 | 1135/4588 | +0.33 | 1.34 | 1.37 | −0.00 / 1.00 | +0.31 / 1.30 | +0.12 / 1.13 |
| DEDICATED 15+5 unique | 6550 | 958/5592 | +0.29 | 1.34 | 1.34 | −0.05 / 0.95 | +0.21 / 1.25 | +0.09 / 1.10 |

2022 calendar mix under concat is **117 long + 1031 short**. Equal-dollar per fill makes the short scratch book dominate; pooled 2022 ~flat is **not** a fixed long book.

Unified FCFS cap 8 skips **1747** long fills (late last-15m arrival vs short occupancy) — same starvation mode as H2+L3.

## Equalized friction 0.25% (both sleeves)

| Book | E | PF | 2022 E/PF | 2023 E/PF | 2022-23 E/PF |
|------|---|----|-----------|-----------|--------------|
| Concat | +0.07 | 1.06 | **−0.16 / 0.84** | −0.10 / 0.92 | −0.13 / 0.88 |
| UNIFIED 15 | +0.20 | 1.19 | **−0.17 / 0.81** | +0.06 / 1.05 | −0.09 / 0.92 |
| DEDICATED 15+5 | +0.12 | 1.13 | **−0.21 / 0.79** | +0.05 / 1.05 | −0.08 / 0.92 |

Under equal friction, **2022 stays red** on every walk. Short sleeve alone is known-fragile at 0.25% (prior note E +0.06).

## Verdict

**No promote.**

- Concat / unique almost “fixes” 2022 only by n-weighting shorts over failed breakouts (mixed payoff scales; H2 avg win ~9% vs short ~1.6%).
- Occupancy walks do not turn 2022 green once friction is equalized.
- Same failure class as the 2026-09-13 H2+L3 blend: do not promote on pooled E while 2022-23 stays ≤~1 PF under honest stress.
- Number to beat remains last-15m alone **E +0.27 PF 1.09**. Do not wire shorts into `/hot` or nightly from this.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last15m_resist_short.py
```

CSV: `reports/ascending_channels/2026-09-19/last15m_resist_short_blend_occupancy_summary.csv`.
