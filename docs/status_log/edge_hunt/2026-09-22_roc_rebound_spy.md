# ROC rebound SPY claim-check (2026-09-22)

Source: Quantified Strategies video [SF1OtXR5crQ](https://www.youtube.com/watch?v=SF1OtXR5crQ) — 7d ROC < −3% buy next open; ROC > +3% sell next open; no stop/TP.

## Rebuild

Yahoo SPY 1993-01-29 → 2026-09-16, wall-clock ~8s.

| Book | n | WR | PF | E | eq | MDD |
|------|---|----|----|---|----|-----|
| Claimed | 146 | 71.2% | 2.40 | — | ~9.15 | ~29% |
| Ours fric0 | 147 | 71.43% | 2.399 | +1.65% | 9.11 | 29.0% |
| Ours fric0.25 | 147 | 69.39% | 2.115 | +1.40% | 6.34 | 30.3% |
| TimescaleDB 2010+ | 61 | 78.69% | 4.619 | +2.28% | 3.78 | 11.3% |

Eras (ours): 1993–2004 n=55 PF 2.71; 2005–2015 n=51 PF 2.08; 2016–2026 n=41 PF 2.45 — matches video within 1 trade.

Length 4–11 and exit +1…+5 all stay PF > ~1.8.

## Verdict

**Claim replicates.** In-sample only; MDD ~29%; n~147 thin. **No promote** as a Sharpe>1 edge-hunt sleeve — keep as documented research / optional SPY bounce overlay candidate.

Artifacts: `reports/research/roc_rebound/`; strategy doc `docs/strategies/roc_rebound/README.md`.
