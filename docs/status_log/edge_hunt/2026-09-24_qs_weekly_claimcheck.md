# QS weekly strategies claim-check

- Generated: 2026-09-24T20:32:50.130130+00:00
- Wall clock: 63.3s
- Sources: [weekly trading strategy](https://www.quantifiedstrategies.com/weekly-trading-strategy/), [SPY down-week](https://www.quantifiedstrategies.com/sp500-down-week-trading-strategy/), [#69 shop](https://www.quantifiedstrategies.com/product/strategy69/)

## 1) SPY down-week (free rules)

Rules: if Friday close < prior Friday close, long for the next week.

| Book | n | WR | PF | Avg% | CAGR | Sharpe | MDD | Exposure | Gate |
|------|---|----|----|------|------|--------|-----|----------|------|
| Claimed | — | — | — | 0.44 | 9.3% | — | 36.0% | 42.0% | CLAIM |
| SPY_down_week_claim_close | 766 | 59.27% | 1.485 | 0.394 | 8.37% | 0.584 | 39.45% | 43.49% | FAIL (Sharpe<=1.0) |
| SPY_down_week_next_open | 766 | 58.49% | 1.452 | 0.357 | 7.58% | 0.566 | 43.44% | 43.49% | FAIL (Sharpe<=1.0) |

SPY buy&hold on same window (claim fill book): CAGR 8.87% Sharpe 0.478 MDD 56.47%.

## 2) Strategy #69 reconstructions (paywalled — ROC proxy)

Universe SPY/EFA/GLD/TLT. Rank by N-period ROC; buy lowest (Substack teaser) or highest (classic momentum); hold one period. Not the exact #69 variable.

### Claimed

- Weekly article: n=728 CAGR 15.1% avg 0.4% MDD 43.0%
- Monthly shop: n=179 CAGR 10.1% avg 1.2% WR 58.0% PF 1.9 MDD 21.0%

### Top by Sharpe (n>=30)

| name | n | WR | PF | Avg% | CAGR | Sharpe | MDD | keep% | Gate |
|------|---|----|----|------|------|--------|-----|-------|------|
| rot69_monthly_lb3_highest | 259 | 55.98% | 1.634 | 0.853 | 9.43% | 0.536 | 32.78% | 57.9 | FAIL (Sharpe<=1.0) |
| rot69_weekly_lb3_lowest | 1137 | 57.7% | 1.29 | 0.228 | 10.73% | 0.535 | 41.06% | 54.5 | FAIL (Sharpe<=1.0) |
| rot69_weekly_lb1_lowest | 1139 | 57.42% | 1.265 | 0.213 | 9.79% | 0.483 | 46.79% | 24.1 | FAIL (Sharpe<=1.0) |
| rot69_weekly_lb12_highest | 1128 | 54.43% | 1.211 | 0.161 | 7.37% | 0.431 | 35.46% | 78.1 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| rot69_monthly_lb6_highest | 256 | 54.3% | 1.491 | 0.683 | 7.26% | 0.415 | 34.86% | 65.2 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| rot69_monthly_lb2_highest | 260 | 56.92% | 1.491 | 0.673 | 7.14% | 0.408 | 33.75% | 44.2 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| rot69_weekly_lb8_highest | 1132 | 54.06% | 1.201 | 0.15 | 6.8% | 0.402 | 43.48% | 71.4 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| rot69_monthly_lb1_highest | 261 | 54.79% | 1.469 | 0.66 | 6.91% | 0.397 | 38.32% | 28.4 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| rot69_weekly_lb6_highest | 1134 | 54.14% | 1.194 | 0.145 | 6.5% | 0.387 | 36.52% | 68.3 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| rot69_monthly_lb12_highest | 250 | 56.8% | 1.419 | 0.673 | 6.82% | 0.369 | 33.88% | 75.2 | FAIL (Sharpe<=1.0, Sharpe<SPY) |

## Verdict

**SPY down-week: claim roughly replicates** (avg ~0.39% vs 0.44%, CAGR ~8.4% vs 9.3%, MDD ~39% vs 36%, exposure ~43% vs 42%). Honest next-open fill is a bit weaker. Beats SPY Sharpe on the full Yahoo window but **fails Sharpe>1** — **no promote**.

**#69: not verified / no promote.** ROC lowest/highest proxies never clear Sharpe>1; none hit the article's weekly 15.1% CAGR. Monthly highest-lb3 (~9.4% CAGR, MDD ~33%) is in the shop's ballpark but is a reconstruction, not the paid variable. Substack "buy lowest" does not unlock a gate pass.

Artifacts: `reports/research/qs_weekly/`; harness `utils/research/qs_weekly.py` / `scripts/research/claimcheck_qs_weekly.py`.

