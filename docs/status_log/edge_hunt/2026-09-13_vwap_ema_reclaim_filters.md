# VWAP reclaim selectivity overlays (2026-09-13)

Follow-up to [first book](2026-09-13_vwap_ema_reclaim.md). Same next-mid list, **fric 0.10**, occupancy not re-walked except daily top1. **No promote.**

Question: n=227915 is almost every session's first 9/VWAP cross. Can a ~100x filter (video "the trade I wait for every day") salvage E/PF?

Pre-registered (not a fitted grid): min unit vs 0.1% friction, morning signal <11:00 ET, LOD grind age>=4 (1h), one name/day ranked by unit_pct.

Wall **38s**. Artifacts: `reports/vwap_ema_reclaim/2026-09-13/vwap_ema_reclaim_filter_overlay.csv`.

## Results (fric 0.10)

| Cut | n | factor | E | PF | geo | 2022 E/PF | drop-top-3 E |
|-----|---|--------|---|----|-----|-----------|--------------|
| baseline | 227915 | 1x | −0.10 | 0.67 | 0.66 | −0.10 / 0.69 | −0.10 |
| unit>=1.5 | 47524 | 5x | −0.07 | 0.84 | 0.83 | −0.06 / 0.87 | −0.08 |
| unit>=3 | 7244 | 31x | −0.02 | 0.97 | 0.94 | +0.00 / 1.00 | −0.03 |
| **unit>=5** | **867** | **263x** | **+0.01** | **1.01** | **0.92** | −0.14 / 0.84 | **−0.01** |
| morning<11 | 24596 | 9x | −0.10 | 0.65 | 0.64 | −0.14 / 0.61 | −0.10 |
| lod_age>=4 | 209357 | 1.1x | −0.10 | 0.67 | 0.66 | −0.10 / 0.70 | −0.10 |
| unit>=3 + morning | 324 | 704x | −0.18 | 0.79 | 0.71 | −0.14 / 0.85 | −0.22 |
| **daily top1** | **1959** | **116x** | −0.11 | 0.85 | 0.83 | **−0.36 / 0.59** | −0.12 |
| top1 + unit>=2 | 1940 | 118x | −0.11 | 0.85 | 0.84 | −0.36 / 0.60 | −0.12 |
| top1 + morning | 1910 | 119x | −0.11 | 0.79 | 0.77 | −0.06 / 0.90 | −0.11 |

Every skip vs baseline is **winner-$ blunt** (drop ~45% winners). lod_age>=4 barely cuts: the 15m book already waits at least one bar, and most crosses are already a 1h+ grind.

unit>=5 is the only pooled-green slice. It is **not** a promote: geo 0.92 from 2018-19 PF **0.61**; drop-top-3 flips E negative; 2022 PF 0.84; 83% EOD (717/867) so the measured-move almost never completes; n=867 is small-n noise vs the 228k parent.

Daily top1 is the literal 100x "one trade a day." It **loses more** than the parent after friction (2022 E −0.36). Ranking the biggest LOD does not pick the 3:1 payoff — 73% still EOD.

## Verdict

**No promote.** You can cut n by ~100x. That does not create an edge after 0.1% friction.

The detector fires because a 9 EMA / VWAP cross after some session low is common, not rare. Selectivity (bigger unit, morning, one name/day) does not recover the video's 3:1 once fills are next-mid and costs are 0.1%. Do not threshold-fish unit>=4.x. Do not occupancy-walk these lists onto last-15m.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\overlay_vwap_ema_filters.py
```
