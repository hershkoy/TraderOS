# ROC rebound ETF robustness grid (2026-09-22)

18 liquid ETFs (SPY/QQQ/IWM/DIA/VTI/MDY + sectors + GLD/EFA/EEM). Overlays fixed (not fished). Wall-clock ~81s.

## Friction 0.10% RT (primary)

| Variant | n | WR | PF | E | MDD | geo10 | drop3 PF/E |
|---------|---|----|----|---|-----|-------|------------|
| A0 baseline | 2205 | 71.1 | 1.69 | +1.04 | -66 | 1.78 | 1.67 / +1.01 |
| **A1 SMA200** | 1190 | 72.9 | **2.05** | **+1.25** | **-35** | **2.11** | 2.02 / +1.22 |
| A2 vol-scale | 1426 | 69.5 | 1.70 | +1.03 | -58 | 2.04 | 1.67 / +0.99 |
| **A3 SMA200+vol** | 781 | 73.2 | **2.38** | **+1.42** | **-32** | **2.30** | 2.31 / +1.36 |
| T20 on A0 | 2536 | 68.1 | 1.55 | +0.84 | -64 | 1.46 | — |
| T40 on A0 | 2274 | 70.9 | 1.67 | +1.00 | -67 | 1.76 | — |
| T20 on A3 | 836 | 68.2 | 1.77 | +0.89 | -37 | 1.71 | winner-$ reject |
| T40 on A3 | 807 | 73.4 | 2.19 | +1.25 | -33 | 2.04 | below A3 |

A3 eras (fric 0.10): 1993–2004 PF 2.77; 2005–2015 PF 2.16; 2016–2026 PF 2.49.

Zero-friction same ranking (A3 PF 2.51 E +1.52 MDD -31).

## Verdict

- **A1 / A3 improve** baseline on PF, E, MDD, and era PFs; drop-top-3 does not collapse A3 → not a pure winner-$ clip.
- **Vol-scale alone (A2)** ≈ flat vs A0 after costs — skip.
- **Time-stops** hurt (T20 especially clips expectancy). Do not promote.
- **Soft-interesting ETF sleeve:** A3 (SMA200 + vol-scaled entry, fixed +3% ROC exit, no time-stop). Still not a Sharpe>1 edge-hunt promote / not live-wired — pooled multi-ETF book is research-only; occupancy not walked.

Artifacts: `reports/research/roc_rebound/roc_rebound_etf_grid.csv`  
Runner: `scripts/research/run_roc_rebound_etf_grid.py`
