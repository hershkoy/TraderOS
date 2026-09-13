# Choppy-regime confirm overlays (2026-09-13)

Follow-up to [refine](2026-09-13_choppy_regime_refine.md). **No promote.** Do not mix with SMA50 threshold numbers or the SMA200 / VIX-skip first book.

Question: can price-action confirmation salvage occupancy clustering (RSI-2 max concurrent 185) and 2023 pullback knives (time-stop-6 still PF 0.70)?

Rules (SMA50, no VIX sit-out, CPI-eve on, fric 0.10, ALPACA 1d, same 1527x2217 panel):

1. **RSI cross-back.** Arm when Wilder RSI(2) < 5 above SMA50. Buy the later MOC where RSI reclaims **5** or **10**. Cancel if close loses SMA50 or wait > 8 sessions. Occupancy ranks by **higher** RSI at reclaim (stronger bounce), not the still-crashing tail.
2. **Pullback close-loc 0.7.** Same 3-LL / SMA100 / SMA5 / ATR limit / time-stop 6, but the setup bar must close in the top 30% of that day's range.

Wall **48.2s**. Artifacts: `reports/choppy_regime/confirm/2026-09-13/`.

## Results vs SMA50 threshold refine

| Book | n | E | PF | geo | max conc | 2022 n/E/PF | occ-15 fric0.10 PF |
|------|---|---|----|-----|----------|-------------|-------------------|
| SMA50 threshold (refine) | 3674 | +0.61 | **1.55** | 1.54 | **185** | 194 / +1.08 / 2.57 | **1.30** |
| SMA50 occ-15 (refine) | 2082 | +0.47 | 1.38 | 1.33 | 15 | 105 / +0.89 / 2.07 | 1.30 |
| xback5 unconstrained | 2978 | +0.27 | 1.30 | 1.28 | **110** | 165 / +0.38 / 1.54 | — |
| xback5 occ-15 | 2236 | +0.30 | 1.33 | 1.28 | 15 | 121 / +0.44 / 1.69 | **1.21** |
| xback10 unconstrained | 2954 | +0.26 | 1.30 | 1.25 | 102 | 165 / +0.22 / 1.31 | — |
| xback10 occ-15 | 2233 | +0.32 | 1.37 | 1.29 | 15 | 121 / +0.40 / 1.63 | 1.25 |
| Pullback time-6 (refine) | 4564 | +0.27 | 1.24 | 1.06 | 104 | 111 / +0.82 / 2.92 | 1.19 |
| Pullback loc0.7 | 1104 | **+0.01** | **1.01** | 0.86 | 49 | 34 / +0.99 / 5.90 | **0.93** |
| Pullback loc0.7 occ-15 | 1021 | +0.01 | 1.01 | 0.83 | 15 | 20 / +0.66 / 3.63 | 0.93 |

Calendar **2023**: xback5 n=634 E +0.16 PF 1.19 (occ-15 n=487 E +0.09 PF 1.10). Pullback loc0.7 n=340 E **−0.82 PF 0.49** (worse than time-stop-6 n=1193 E −0.40 PF 0.70). Drop-top-3 on loc0.7 goes **negative** (E −0.05).

Year buckets (geo skips n<30): xback 2022-23 PF 1.25 / 1.18 vs threshold 1.52. Loc0.7 2022-23 **PF 0.55**.

## Verdict

**No promote.** Confirmation does not salvage this path.

Cross-back does what it says on clustering: max concurrent **185 → 110**, n down ~19% (3674 → 2978). It does **not** isolate a higher-probability 15-name book. Occupancy-15 + 0.10% friction is **PF 1.21 vs threshold 1.30**. Median trade **+0.61 vs +1.15**: waiting for RSI to reclaim buys after the snapback has started. Win rate fell (68% → 65.5%), so this is not a 20% whipsaw cut on this panel. 2022 stays green but weaker. Cap 15 is still a cluster day; 110 concurrent is not live equal-dollar.

Pullback close-near-high is a **2023-worse / winner-$** overlay: n collapses, 2023 PF 0.49, pooled PF ~1.00, drop-top-3 negative. Closing strong after three lower lows did not mean the mean held. Do not require loc 0.7 on this book.

Keep SMA50 (refine). Do not drop the MA. Do not occupancy-walk onto last-15m. Naive threshold + these two confirms are exhausted for a production sleeve.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\run_choppy_regime.py --confirm-only --outdir reports\choppy_regime\confirm
```
