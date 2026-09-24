# ROC A3 single-slot occupancy vs SPY (2026-09-22)

Frozen A3 (SMA200 + vol-scale, fric 0.10%). Pre-declared holdout **2018-01-01 → 2026-12-31**. Single-slot FCFS occupancy (unique symbol; same-day rank by more-negative entry ROC). Equity: 100% invested when in a trade, cash when flat; daily MTM. Gate: Sharpe>1 **and** Sharpe≥SPY **and** MDD better than SPY.

Wall-clock ~18s.

## Trade books

| Book | n | PF | E |
|------|---|----|---|
| A3 unconstrained (full) | 781 | 2.38 | +1.42% |
| A3 occupancy max_open=1 (full) | **151** (skipped 630) | **1.42** | +0.63% |
| Holdout unconstrained | 319 | 2.15 | — |
| **Holdout occupancy** | **56** | **1.20** | **+0.34%** |

## Equity vs SPY (Phase-5 gates)

| Window | Sleeve Sharpe / MDD / inv | SPY Sharpe / MDD | Gate |
|--------|---------------------------|------------------|------|
| Full | 0.15 / −34% / 52% | 0.75 / −25% | **FAIL** |
| IS pre-2018 | 0.20 / −21% / 52% | 0.77 / −20% | **FAIL** |
| **Holdout 2018–26** | **0.12 / −34% / 53%** | **0.72 / −25%** | **FAIL** |

Fails all three legs: Sharpe≪1, Sharpe≪SPY, MDD worse than SPY.

## Verdict

**`FAIL_HOLDOUT`**

Pooled multi-ETF PF ~2.3 / walk-forward OOS PF ~2.3 **do not upgrade** under capital realism. Forcing one slot collapses holdout PF to ~1.2 and produces a cash-heavy path that loses the Sharpe/MDD race to SPY buy-and-hold.

Still valid as a **research / claim-check** of the QS rebound idea on indexes; **not** a promote vs SPY, not live-wired.

Runner: `scripts/research/run_roc_rebound_occupancy.py`  
Artifacts: `reports/research/roc_rebound/roc_rebound_occupancy_summary.json`, `roc_rebound_occupancy_vs_spy.csv`
