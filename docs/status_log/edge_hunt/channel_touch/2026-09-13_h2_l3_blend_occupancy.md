# H2+L3 occupancy walk: unified FCFS vs dedicated sleeves (2026-09-13)

Timestamp greedy walk on stored last-15m H2 + 15m L3 wait-12 fills (n=2297 + 1744). Unique-symbol single account unless `_indep`. Equal-dollar **per taken fill** (L3 avg win ~2.5% vs H2 ~9% — pooled E is mixed-scale). **No promote.**

Runner: `scripts/research/overlay_h2_l3_blend_occupancy.py`. Logic: `utils/research/h2_l3_blend_occupancy.py`.

## Hypothesis vs what the walk does

The crowding note was: a global cap of 15–20 will lock L3 out because H2 got there first and holds ~16d. Last-15m H2 fills at **15:45 ET**. L3 fires any RTH 15m bar and is usually flat by the next session (median hold 0.81d, unconstrained max concurrent **7**). FCFS is arrival-time, not “H2 owns the book.”

## Baselines

| Book | n | H2 / L3 | E | PF | geo | 2022 E / PF | 2023 E / PF | 2022-23 |
|------|---|---------|---|----|-----|-------------|-------------|---------|
| H2 unconstrained | 2297 | 2297 / 0 | +0.27 | 1.09 | 1.10 | −2.37 / 0.41 | −0.33 / 0.90 | −0.90 / 0.73 |
| L3 unconstrained | 1744 | 0 / 1744 | +0.23 | 1.43 | 1.44 | **+0.05 / 1.09** | **+0.30 / 1.56** | +0.18 / 1.32 |
| Concat, no occ | 4041 | 2297 / 1744 | +0.25 | 1.13 | 1.14 | −0.74 / 0.57 | −0.04 / 0.98 | −0.32 / 0.83 |
| Unique-symbol, no cap | 3955 | 2287 / 1668 | +0.26 | 1.13 | 1.15 | −0.74 / 0.57 | −0.06 / 0.97 | −0.33 / 0.83 |

Same-name overlap drops 86 fills (76 L3, 10 H2). Overlaying the logs without a cap does **not** fix 2022-23: equal-trade mix of H2 disasters and L3 scratch.

## 1. Unified FCFS (one max_open)

| Cap | n | H2 / L3 | E | PF | geo | L3 kept | cap-skip H2 / L3 | 2022 E | 2023 E | 2022-23 PF |
|-----|---|----------|---|----|-----|---------|------------------|--------|--------|------------|
| **8** | 1697 | 587 / 1110 | +0.32 | 1.22 | 1.28 | 64% | **1705 / 614** | −0.53 | +0.48 | 0.94 |
| **15** | 2250 | 923 / 1327 | +0.54 | 1.35 | 1.41 | **76%** | 1368 / **381** | −0.65 | +0.43 | 0.93 |
| **20** | 2538 | 1128 / 1410 | +0.53 | 1.32 | 1.37 | 81% | 1162 / 296 | −0.72 | +0.39 | 0.91 |

L3 is **not** locked out entirely at 15–20 (381 / 1744 skipped for cap at 15). Melt-up H2 holds do crowd some L3. The bigger FCFS effect at cap 8 is the reverse: **H2 is starved** (1705 cap skips) because the last-15m print arrives when L3 already sits on up to 7 of 8 names.

2022 stays red on every cap. 2023 goes green because the walk keeps more L3 than H2 that year, not because H2 resist-chase started working.

H2 winner-$ at cap 15: drop 1374 names, 2020-21 win$ 1235 vs lose$ 1428 (not blunt). That is the known `max_open` crowding cost on H2, mixed with L3’s smaller dollars.

## 2. Dedicated sleeves (L3 reserved)

Unique-across = one lot per name. Independent = two books may double a name. L3 unconstrained max concurrent is 7, so a reserve of 5 almost never binds.

| Book | n | H2 / L3 | E | PF | geo | L3 sleeve_full | 2022 E | 2023 E | 2022-23 PF |
|------|---|---------|---|----|-----|----------------|--------|--------|------------|
| 15+5 unique | 2666 | 960 / **1706** | +0.43 | 1.29 | 1.35 | **5** | −0.66 | +0.25 | 0.89 |
| 15+5 indep | 2699 | 960 / 1739 | +0.43 | 1.29 | 1.35 | 5 | −0.66 | +0.28 | 0.90 |
| 12+3 unique | 2482 | 807 / 1675 | +0.39 | 1.28 | 1.32 | 39 | −0.55 | +0.18 | 0.89 |
| 8+4 unique | 2309 | 599 / 1710 | +0.22 | 1.18 | 1.21 | 11 | −0.50 | −0.01 | 0.82 |

Reserved slots **do** what they claim: L3 keep-rate ~96–98% vs unified 15’s 76%. Independent vs unique-across is noise (same-name overlap is small). H2 taken count tracks the H2 cap (same crowding as unified on that sleeve). 2022 is still a failed H2 sleeve plus scratch L3. Dedicated 8+4 pooled E **loses** to last-15m +0.27 because it throws away most H2 winners and overweight L3’s smaller R.

## Verdict

**No promote.** Neither occupancy rule is a regime blend that “lets structure allocate capital”:

- Concat / unique-symbol overlay leaves 2022-23 red.
- Unified 15–20 does not lock L3 out; it skips ~20% of L3 and a lot of H2. Cap 8 starves **H2** (late last-15m arrival).
- Dedicated 15+5 keeps L3 funded and still has 2022 E −0.66 / 2022-23 PF 0.89.
- Pooled E/PF lifts (unified 15 +0.54 / 1.35) mix two payoff scales and a H2 `max_open` crowding filter. Do not grade that vs last-15m +0.27 / 1.09. Geo of year buckets is not a pass while 2022-23 PF &lt; 1.
- 15m L3 wait-12 stays the hostile-regime **book**, not a sleeve you bolt onto last-15m under a shared cap.

Number to beat remains last-15m **n=2297 E +0.27 PF 1.09**. Do not cap-fish 9/12/18.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_h2_l3_blend_occupancy.py
```

CSV: `reports/ascending_channels/2026-09-13/h2_l3_blend_occupancy_summary.csv`.
