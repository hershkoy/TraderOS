# Cash-flow screen v2 claim-check (2026-10-03)

**Verdict: no_promote**

1y annual-rebalance book with FCF yield, shareholder yield, cash OP, CapEx gate, and optional 12m momentum still **lags SPY on median**.

## Spec (locked)

| Gate | Rule |
|------|------|
| Hold | **1 year**; unique symbol per `filed` calendar year |
| CapEx | CapEx/OCF < 35% |
| FCF yield | FCF / EV > **5%** (EV = mkt_cap + debt − cash) |
| Shareholder yield | (net buybacks + dividends TTM) / mkt_cap > **1%** |
| Cash OP ROA | (OpInc − (NI − OCF)) / Assets > **6%** (OCF fallback if OpInc/NI missing) |
| Momentum | 12m price return > 0 (ablation without mom also run) |
| Sector | SIC exclude FIN/EN/UTIL/MAT/RE |

Code: `utils/research/cashflow_four_filters_v2.py`, `scripts/research/claimcheck_cashflow_four_filters_v2.py`

## Results (unique symbol-year, full 1y hold)

| Book | n | syms | median | mean | WR | PF | SPY median | excess med |
|------|---|------|--------|------|-----|-----|------------|------------|
| **v2 + 12m mom** | 538 | 294 | **6.9%** | 15.6% | 57% | 2.42 | **18.7%** | **-11.8** |
| v2 no momentum | 1145 | 435 | **7.5%** | 24.7% | 59% | 3.59 | **18.2%** | **-10.7** |

Momentum shrinks n and does **not** lift median vs SPY (slightly worse excess). Fat right tail still inflates mean/PF.

Wall ~170s/run (companyfacts enrich + price panel).

## vs v1

- v1 5y full: n=7 only (unusable for claim)
- v1 3y: med 25% vs SPY 70%
- v2 1y: much larger n; still loses to SPY on median

## Artifacts

- `reports/research/cashflow_four_filters_v2/2026-10-03/` (+mom)
- `reports/research/cashflow_four_filters_v2/2026-10-03_nomom/` (ablation)

## Decision

**Do not promote.** Larger tradeable sample does not fix the edge vs SPY; momentum overlay does not rescue median underperformance on this panel.
