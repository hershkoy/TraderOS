# Cash-flow screen v3 claim-check

**Verdict: no_promote** — primary top-30 + relative momentum is thin (≤4 thick years) and does not clear a durable SPY/RSP edge; trade median still lags SPY.

## Spec (requested structural fix vs v2)

- Soft floors → eligibility pool, then **top-N / year** by
  `rank(FCF yield) + rank(cash OP ROA) + rank(12-1 mom)`
- Relative 12-1 momentum: stock > SPY (skip last 21d)
- CapEx/OCF < 35% **or** cash OP ROA > 15%
- Hold 1y; EW annual portfolio rebalance + per-trade book
- Benchmarks: SPY (TimescaleDB) + RSP (Yahoo)
- Headline portfolio CAGR uses years with **n_names ≥ 10** (single-name cohorts excluded)

## Books

### A. Primary — top-30 + rel mom (`2026-10-03/`)

| | |
|---|---|
| Eligible / selected | 1372 / 80 |
| Trades | n=80 med=11.49% mean=17.19% WR=60% PF=2.64; spyMed=20.62% rspMed=11.83%; **xsSPY=−9.13% xsRSP=−0.34%** |
| Portfolio (all years) | 4y CAGR 13.12% vs SPY 13.30% / RSP 10.88% (xsSPY −0.18%) |
| Portfolio (n≥10) | **3y** CAGR 24.93% vs SPY 22.05% / RSP 15.80% (2023–25; 2022 was n=1) |
| Wall-clock | 189s |

Year path (rel): 2022 n=1 −16%; 2023 n=30 +22.6% (SPY +31.4%); 2024 n=30 +28.9%; 2025 n=19 +23.4%.

### B. Ablation — top-25 + rel mom (`2026-10-03_top25/`)

- Eligible/selected 1372 / 66
- Trades n=66 med=10.7% xsSPY=−10.0% xsRSP=−2.0%
- Portfolio 4y CAGR 13.4% ≈ SPY 13.55%; thick window still short
- Same story as A — no promote

### C. Ablation — top-30, **no** rel mom (`2026-10-03_norel/`)

| | |
|---|---|
| Eligible / selected | 4827 / 113 |
| Trades | n=113 med=11.3% mean=24.0% WR=60% PF=3.79; spyMed=17.5% rspMed=9.8%; **xsSPY=−6.2% xsRSP=+1.5%** |
| Portfolio (all years) | 6y CAGR **34.5%** vs SPY 22.0% / RSP 17.7% — **inflated** |
| Portfolio (n≥10) | **4y** CAGR ≈24.2% vs SPY ≈20.7% / RSP ≈12.3% |

Inflation source: 2020 **AEO alone** +100%, 2021 **AAPL alone** +24.6%. Thick years 2022–25 still beat SPY CAGR on this short window, but 2023 loses to SPY (+22% vs +31%), trade median still lags SPY, and n_years is too thin for a Sharpe>1 / occupancy promote.

## Decision

1. **Trade-median vs SPY-median** remains a lose (Bessembinder / equal-weight vs cap-weight) even after top-N.
2. **Relative momentum** shrinks the book and kills early thin years; it does **not** create a median edge vs SPY.
3. **EW portfolio** can look competitive vs RSP and sometimes SPY on thick years, but sample is 3–4 annual cohorts — not enough for edge-hunt promote.
4. Status: **no_promote** (same family as v1/v2). Do not wire live.

Reports: `reports/research/cashflow_four_filters_v3/2026-10-03{,_top25,_norel}/`
Harness: `utils/research/cashflow_four_filters_v3.py` / `scripts/research/claimcheck_cashflow_four_filters_v3.py`
