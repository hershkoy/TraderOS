# ROC capital-constraint fixes vs SPY (2026-09-22)

Holdout **2018–2026**, fric 0.10%. Goal: repair the single-slot FCFS failure (Sharpe 0.12, MDD −34%).

## Fixes tested

| ID | Stack |
|----|--------|
| B0 | A3 + deepest-ROC daily rank, 1 slot |
| B1 | + SPY>SMA200 regime |
| B2 | + relative weakness (stock ROC < SPY ROC) |
| B3 | + time-stop 7 |
| B4 | regime + time-stop 7, 1 slot |
| B5 | B4 + min entry_roc ≤ −5% |
| B6/B7 | B4 with 4 / 5 slots |
| B8 | B6 + rel-weak |
| B9 | B6 + idle cash Rf 4% annual |

## Holdout results

| Variant | occ n | PF | E | Sharpe | MDD | inv | vs SPY Sharpe 0.72 / MDD −25% | Gate |
|---------|-------|----|---|--------|-----|-----|-------------------------------|------|
| B0 rank1 | 56 | 1.20 | +0.34 | 0.12 | −34% | 53% | worse | FAIL |
| B1 regime | 40 | 1.52 | +0.65 | 0.24 | −16% | 46% | worse | FAIL |
| B2 relWeak | 53 | 1.09 | +0.17 | 0.05 | −32% | 51% | worse | FAIL |
| **B3 tstop7** | 99 | **1.80** | +0.75 | **0.68** | **−11%** | 33% | MDD better; Sharpe≈SPY | FAIL |
| B4 regime+tstop | 74 | 1.78 | +0.69 | 0.59 | −11% | 29% | MDD better | FAIL |
| B5 deep−5% | 33 | 2.19 | +1.24 | 0.59 | −11% | 13% | thin n | FAIL |
| B6 4-slot | 187 | 1.53 | +0.53 | 0.48 | −11% | 18% | MDD better | FAIL |
| B7 5-slot | 211 | 1.51 | +0.52 | 0.46 | −10% | 16% | MDD better | FAIL |
| B8 +relWeak 4s | 180 | 1.55 | +0.57 | 0.50 | −10% | 17% | MDD better | FAIL |
| **B9 4s+Rf4%** | 187 | 1.53 | +0.53 | **0.93** | **−9%** | 18% | Sharpe>SPY; MDD better | **FAIL** (Sharpe≤1) |

SPY holdout: Sharpe **0.72**, MDD **−25.4%**.

## What worked / didn’t

- **Ranking alone (B0)** does not fix cross-day slot lockout — same FAIL as FCFS.
- **Time-stop 7** is the real velocity fix: MDD −34% → ~−11%, Sharpe 0.12 → 0.68.
- **SPY regime** helps MDD vs B0 but adds little once time-stop is on.
- **Rel-weak alone** hurts; with regime+tstop it’s neutral.
- **4–5 slots** smooth MDD slightly but dilute Sharpe vs 1-slot tstop.
- **Rf on idle cash (B9)** lifts Sharpe to **0.93** (beats SPY) and MDD −9%, but still fails Phase-5 absolute **Sharpe>1**.

## Verdict

**`SOFT_IMPROVE_NO_GATE`**

Capital fixes (especially time-stop + Rf) make the sleeve **much less toxic** under occupancy, and B9 **beats SPY on Sharpe and MDD**, but nothing clears Sharpe>1. Still **no promote / not live-wired**.

Runner: `scripts/research/run_roc_rebound_capital_fix.py`  
Artifacts: `reports/research/roc_rebound/roc_rebound_capital_fix.csv`
