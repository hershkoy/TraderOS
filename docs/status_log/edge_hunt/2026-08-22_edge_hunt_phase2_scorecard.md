# Edge hunt Phase 2 - dual momentum vs SPY buy-and-hold

Date: 2026-08-22

## Window / data

- Primary eval: `2011-01-03` -> `2025-11-25` (IB SPY/EFA/BIL + TRADINGVIEW SHY)
- Sensitivity: `2018-11-01` -> `2025-11-25` (same providers)
- Rule: month-end pick SPY vs EFA by ~12m return; if winner 12m <= 0 hold SHY or BIL
- Gate: Sharpe >= SPY and/or better MDD with CAGR within ~2pp (or higher CAGR)

Harness: `scripts/research/run_dual_momentum_phase2.py`

## Scorecard A - 2011-01 -> 2025-11

| name | CAGR | Sharpe | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 11.87% | 0.69 | -34.10% | 0.35 | 17.28% | 431.47% | 100% | +0.00% | +0.00 | BENCHMARK |
| SPY_SMA200_abs_mom | 8.37% | 0.67 | -25.85% | 0.32 | 12.52% | 231.04% | 83% | -3.50% | -0.02 | FAIL |
| DualMom_SPY_EFA_SHY | 5.43% | 0.35 | -34.10% | 0.16 | 15.41% | 119.75% | 99% | -6.44% | -0.33 | FAIL |
| DualMom_SPY_EFA_BIL | 6.24% | 0.43 | -34.10% | 0.18 | 14.67% | 146.42% | 95% | -5.63% | -0.26 | FAIL |

Notes:
- DualMom_SPY_EFA_SHY: rebalances=179; hold_days mostly SPY (2766), SAFE 561, EFA 400
- DualMom_SPY_EFA_BIL: rebalances=172; hold_days SPY 2661, SAFE 561, EFA 357
- Max DD matches SPY (-34.1%): absolute filter did not exit before the sample's worst equity drawdown (strategy spent most days in SPY)

## Scorecard B - 2018-11 -> 2025-11 (Phase 1 window)

| name | CAGR | Sharpe | MaxDD | Calmar | Vol | TotRet | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 13.64% | 0.68 | -34.10% | 0.40 | 20.07% | 146.80% | BENCHMARK |
| SPY_SMA200_abs_mom | 8.42% | 0.67 | -25.85% | 0.33 | 12.61% | 77.10% | FAIL |
| DualMom_SPY_EFA_SHY | 7.07% | 0.40 | -34.10% | 0.21 | 17.55% | 62.01% | FAIL |
| DualMom_SPY_EFA_BIL | 7.28% | 0.42 | -34.10% | 0.21 | 17.52% | 64.27% | FAIL |

## Promotion

**No Phase-2 candidate cleared gates.**

Dual momentum underperformed SPY on CAGR and Sharpe on both windows, and did not reduce max drawdown vs SPY in this sample. SMA200 timing cut MDD (-25.9% vs -34.1%) but still failed the CAGR-within-2pp gate.

### Overall edge-hunt verdict (Phase 1 + 2)

| Candidate | vs SPY |
|-----------|--------|
| SMA200 abs mom | Better/equal MDD sometimes; lower CAGR - FAIL |
| XS 12-1 momentum | Worse - FAIL |
| Weekly BigVol portfolio | Sharpe tie, much better MDD, CAGR ~5pp short - near-miss sleeve |
| Dual mom SPY/EFA/SHY|BIL | Worse CAGR+Sharpe, same MDD - FAIL |

**No strategy cleared the stated SPY-beating gates on available data.** Closest useful finding remains Weekly BigVol as a **lower-drawdown sleeve** (not a CAGR beater).

### Optional next experiments (not promoted)

1. QQQ as risk-on leg instead of SPY (beta bet; needs honesty vs SPY benchmark)
2. Dual mom with end-of-month confirmation delay / 6m lookback sensitivity
3. Accept BigVol as risk-reduced satellite + SPY core (portfolio construction, not single-strategy edge)
4. Longer pre-2010 history if another source can extend SPY/EFA (TV dump pattern)

Artifacts: `reports/edge_hunt_phase2/`, `reports/edge_hunt_phase2_2018/`
