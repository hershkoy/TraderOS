# ROC rebound expanding walk-forward (2026-09-22)

18 ETFs, friction **0.10%** RT. Protocols:

1. **Frozen A3** — SMA200 + vol-scaled entry; score each OOS calendar year 2005–2026.
2. **Variant-select WF** — expanding IS picks best of {A0,A1,A2,A3} by IS PF; apply to next year.
3. **Holdout block** — IS ≤2015-12-31, OOS ≥2016-01-01.

Wall-clock ~26s (Yahoo cache warm).

## Expanding annual OOS (frozen A3)

| Metric | Value |
|--------|-------|
| Pooled OOS n | 694 |
| Pooled PF / E | **2.34** / +1.40% |
| drop-top-3 PF / E | 2.27 / +1.33% |
| geo year-bucket (min10) | 2.30 |
| Folds with trades | 22/22 |
| % folds PF≥1 | **86.4%** |
| Median fold PF | **3.05** |
| Worst fold PF | **0.013** (2008, n=3) |

Fragile OOS years (PF &lt; 1): **2008** (0.01, n=3), **2011** (0.52), **2020** (0.55). Soft years: 2022 PF 1.24, 2023 PF 1.10.

## Variant-select WF

IS picked **A3 every fold** (22/22) → identical to frozen A3. Selecting after seeing the grid does not change the OOS path here; A3 was also the IS winner before 2016.

## Holdout IS≤2015 / OOS≥2016

| Variant | IS PF | OOS n | OOS PF | OOS E |
|---------|-------|-------|--------|-------|
| A0 | 1.63 | 872 | 1.79 | +1.06 |
| A1 | 1.95 | 533 | 2.18 | +1.35 |
| A2 | 1.59 | 618 | 1.89 | +1.13 |
| **A3** | **2.27** | **370** | **2.49** | **+1.56** |

IS would select A3 → OOS PF 2.49 (passes the pre-declared holdout).

## Verdict

**`SOFT_PASS_OOS_FRAGILE_YEAR`**

- Clears sample / holdout / drop-top-3 / friction gates for an ETF research sleeve.
- Does **not** clear “stable every year”: crash/chop folds (2008/2011/2020) can print PF ≪ 1; pooled MDD on concatenated OOS trades still large (~−69% sequential — not a live book).
- Still not a Sharpe>1 live promote; no single-slot occupancy vs SPY run yet.

Runner: `scripts/research/run_roc_rebound_walkforward.py`  
Artifacts: `reports/research/roc_rebound/roc_rebound_walkforward_summary.json`, `roc_rebound_wf_*_folds.csv`
