# Cash-flow four-filter claim-check (2026-10-03)

**Verdict: no_promote**

Video claim (median **+86%** over 5y vs market **+66%**) does not clear on our SEC companyfacts + ALPACA 1d book. Full-horizon 5y sample is tiny; larger 3y / partial books underperform SPY median.

## Setup

- Screen: OCF/debt > 0.30, CapEx/OCF < 0.35, net buybacks TTM / mkt_cap > 0.02, FCF/assets > 0.12
- SIC exclude FIN / EN / UTIL / MAT / RE
- Entry = next 1d close on/after SEC `filed`; exit = close on/after filed + hold
- Filter 3 mkt_cap = shares x close as-of filed (recomputed when stored mkt_cap null)
- Primary book: unique symbol per calendar year of `filed`
- Code: `utils/research/cashflow_four_filters.py`, `scripts/research/claimcheck_cashflow_four_filters.py`

## Results

| Book | n (unique yr) | syms | median | mean | WR | SPY median | excess med |
|------|---------------|------|--------|------|-----|------------|------------|
| **5y full hold** (claim) | 7 | 3 | **70.8%** | 89.5% | 100% | 39.5% | +31.3 |
| 3y full hold | 161 | 100 | **24.9%** | 56.3% | 70% | **70.5%** | **-45.5** |
| 5y allow-partial (to last bar) | 473 | 187 | **11.0%** | 49.4% | 61% | **37.8%** | **-26.8** |

Claim reference: median 86% / market 66% over 5y.

### Why 5y full is n=7

Only filings that can complete 5y before the last ALPACA bar (~2026) enter (~filed through mid-2021). After all four filters, that window collapses to **3 names / 7 symbol-years**. Median 70.8% beats SPY on that micro-sample but is not a claim replication (n<<30, not 86%).

### 3y / partial

Larger books show the screen's recent cohort **lags SPY on median** (fat right tail lifts mean/PF while median stays weak — same fat-tail pattern as other edge-hunt rejects).

## Artifacts

- `reports/research/cashflow_four_filters/2026-10-03_h5/` (5y full)
- `reports/research/cashflow_four_filters/2026-10-03/` (3y full)
- `reports/research/cashflow_four_filters/2026-10-03_partial5y/` (partial)
- Wall-clock ~75s per run (price panel load dominates)

## Decision

**Do not promote.** Not enough completed 5y trades to validate the 86% claim; on the tradeable larger samples the screen loses to SPY on median.
