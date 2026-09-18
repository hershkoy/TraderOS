# QS Five Swing Strategies — claim check (2026-09-18)

**Verdict:** Combined and solo sleeves **largely replicate** the Quantified Strategies article/video on Yahoo SPY ~1993–2026. Treat as selective mean-reversion sleeves (not Sharpe>1 promote).

**Source:** [article](https://www.quantifiedstrategies.com/swing-trading-strategies-for-beginners/) / video KuXV0LRfJx8  
**Code:** `utils/research/qs_five_swing.py`, `scripts/research/generate_qs_five_swing_report.py`  
**Report:** `reports/research/qs_five_swing/qs_five_swing_spy.html`  
**Wall-clock:** ~12.8s (fric 0 / 0.03 / 0.25 on 8465 Yahoo bars)

## Combined (article cost ~0.03%/trade)

| | Claimed | Ours (fric 0.03) | Ours (fric 0) |
|--|---------|------------------|---------------|
| n | 975 | 958 | 958 |
| E %/trade | 0.38 | 0.39 | 0.42 |
| CAGR % | 11.1 | 11.31 | 12.26 |
| Exposure % | 41 | 40.3 | 40.3 |
| MDD % | 23 | 23.8 | 23.8 |
| $100k final | ~3.2M | 3.62M | 4.82M |

Losing years (trade-sum): ours fric 0 shows **2018 −14.4%** only (claim also cites 2008 −1%).

## Solo (fric 0.03) vs highlights

- S1: CAGR 8.6 / exp 18 / MDD 24 vs claim ~8 / 18 / 23
- S2: n=310 CAGR 7.6 / exp 12 / MDD 17 vs claim n=271 ~7.2 / 11 / 16
- S3: CAGR 7.9 / exp 22 vs claim ~8 / 21
- S4: CAGR 4.1 (article: weaker complement; no hard numbers)
- S5: CAGR 3.0 / MDD 8.9 vs video ~3.3 / 8

## Notes

- S1 band uses **2.5 ×** 25d avg range under 10d high (article; not 1×).
- Combined priority S1→S5; fills at fric 0: S4 294, S1 252, S3 148, S5 140, S2 124.
- Friction 0.25% RT: combined CAGR ~4.5% PF ~1.36 — still green, not a free lunch.
- No live wiring this pass.
