# ROC rebound universe sample (2026-09-22)

Seeded random **50** names: 12 ETF + 19 large (SPX/NDX ∩ top ADV) + 19 smaller (lower-ADV non-index). Same 7d ROC ±3% next-open rules. Slippage (round-turn friction): ETF **0.10%**, large **0.25%**, small **0.40%**. Start 2006-01-01; wall-clock ~66s.

Symbols (seed=42): see `reports/research/roc_rebound/roc_rebound_universe_summary.json`.

## Pooled trades (equal-weight fills, not a live portfolio)

| Bucket | syms | n | WR | PF | E | med | geoPF (year min10) |
|--------|------|---|----|----|---|-----|---------------------|
| etf | 12 | 1114 | 68.6% | **1.54** | +0.89% | +2.29 | 1.69 |
| large | 19 | 1227 | 65.6% | **1.34** | +0.84% | +2.25 | 1.33 |
| small | 19 | 1245 | 61.9% | **1.12** | +0.45% | +2.19 | 1.16 |
| all | 50 | 3586 | 65.3% | **1.27** | +0.72% | +2.26 | 1.27 |

Pooled “final equity / MDD” compounds every trade in series — **not** realistic multi-name capital (ignore for promotion).

## Per-symbol (median)

| Bucket | med n | med E | med PF | % with PF>1 |
|--------|-------|-------|--------|-------------|
| etf | 103 | +1.04% | 1.57 | 91.7% |
| large | 62 | +0.70% | 1.43 | 84.2% |
| small | 69 | +0.86% | 1.34 | 78.9% |

Worst names (small): EVGO/PUK/MCHB PF &lt; 0.75. Best: PECO PF 3.3, XLU 2.1, MRK 2.0.

## Verdict

Edge **weakens off SPY** once slippage is on: ETFs still the strongest bucket (PF ~1.5), large marginal (~1.3), small barely clears 1.0 after 40 bp RT. Not a universe promote. SPY claim-check stays the clean reference.

Runner: `scripts/research/run_roc_rebound_universe.py`.
