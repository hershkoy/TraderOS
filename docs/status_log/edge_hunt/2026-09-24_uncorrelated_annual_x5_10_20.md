# Uncorrelated annual sleeve (x=5,10,20)

- Generated: 2026-09-24T21:30:27
- Eval: 2023-01-01 -> 2025-11-26
- Provider: ALPACA; liquid_n=300; cost_bps_rt=10.0
- Panel symbols loaded: 2217; close shape=(979, 2217)
- Wall clock: load=4m 46s total=4m 47s

## Performance

| name | CAGR | Sharpe | Sortino | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | trades | WR | PF | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 21.87% | 1.41 | 2.36 | -19.00% | 1.15 | 15.55% | 77.25% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| UncorrAnnual_x5_L300_gross | 27.03% | 0.24 | 1.47 | -41.31% | 0.65 | 111.21% | 99.87% | 100% | +5.16% | -1.16 | 3 | 33% | 9.20 | FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) |
| UncorrAnnual_x5_L300_net | 26.93% | 0.24 | 1.46 | -41.36% | 0.65 | 111.21% | 99.41% | 100% | +5.06% | -1.16 | 3 | 33% | 9.20 | FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) |
| UncorrAnnual_x10_L300_gross | 31.79% | 0.53 | 2.72 | -27.58% | 1.15 | 59.91% | 122.29% | 100% | +9.92% | -0.88 | 3 | 100% | 99.00 | FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) |
| UncorrAnnual_x10_L300_net | 31.69% | 0.53 | 2.72 | -27.64% | 1.15 | 59.91% | 121.80% | 100% | +9.82% | -0.88 | 3 | 100% | 99.00 | FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) |
| UncorrAnnual_x20_L300_gross | 25.24% | 0.55 | 2.95 | -20.86% | 1.21 | 45.98% | 91.82% | 100% | +3.37% | -0.86 | 3 | 67% | 46.09 | FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) |
| UncorrAnnual_x20_L300_net | 25.15% | 0.55 | 2.94 | -20.92% | 1.20 | 45.98% | 91.40% | 100% | +3.28% | -0.86 | 3 | 67% | 46.09 | FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) |

## Annual returns

| year | SPY_buy_hold | UncorrAnnual_x5_L300_net | UncorrAnnual_x10_L300_net | UncorrAnnual_x20_L300_net |
| --- | --- | --- | --- | --- |
| 2023 | 24.81% | 131.82% | 111.12% | 84.15% |
| 2024 | 23.30% | -4.54% | -0.07% | -2.48% |
| 2025 | 15.18% | -9.89% | 5.14% | 6.58% |

## Year picks by x

## x=5

- Gate (net): **FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY)**
- Notes: x=5 liquid_n=300 cost_bps_rt=10.0; years_with_picks=3/3; avg_mean_pair_corr=-0.069
- Picks CSV: `D:/WORK/Projs/IB/backTraderTest/reports/research/uncorrelated_annual/picks_x5_L300_20260924_213026.csv`
- Equity CSV: `D:/WORK/Projs/IB/backTraderTest/reports/research/uncorrelated_annual/equity_x5_L300_20260924_213026.csv`

| lookback_year | hold_year | n_eligible | n_picks | mean_pairwise_corr | symbols | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2023 | 13 | 5 | -0.1461 | UUP,SPY,UVXY,GLD,AEO | corr_n=13 |
| 2023 | 2024 | 300 | 5 | -0.0090 | MCK,TEAM,SJM,SMCI,TLT | corr_n=300 |
| 2024 | 2025 | 300 | 5 | -0.0533 | GIS,ASML,IBKR,TLT,TPL | corr_n=300 |

## x=10

- Gate (net): **FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY)**
- Notes: x=10 liquid_n=300 cost_bps_rt=10.0; years_with_picks=3/3; avg_mean_pair_corr=0.001
- Picks CSV: `D:/WORK/Projs/IB/backTraderTest/reports/research/uncorrelated_annual/picks_x10_L300_20260924_213026.csv`
- Equity CSV: `D:/WORK/Projs/IB/backTraderTest/reports/research/uncorrelated_annual/equity_x10_L300_20260924_213026.csv`

| lookback_year | hold_year | n_eligible | n_picks | mean_pairwise_corr | symbols | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2023 | 13 | 10 | -0.0082 | UUP,SPY,UVXY,GLD,AEO,VXX,A,TLT,USO,AAPL | corr_n=13 |
| 2023 | 2024 | 300 | 10 | 0.0249 | MCK,TEAM,SJM,SMCI,TLT,PGR,NTES,MNST,PSX,ULTA | corr_n=300 |
| 2024 | 2025 | 300 | 10 | -0.0145 | GIS,ASML,IBKR,TLT,TPL,MCK,NVDA,DXCM,MMM,TSCO | corr_n=300 |

## x=20

- Gate (net): **FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY)**
- Notes: x=20 liquid_n=300 cost_bps_rt=10.0; years_with_picks=3/3; avg_mean_pair_corr=0.041
- Picks CSV: `D:/WORK/Projs/IB/backTraderTest/reports/research/uncorrelated_annual/picks_x20_L300_20260924_213026.csv`
- Equity CSV: `D:/WORK/Projs/IB/backTraderTest/reports/research/uncorrelated_annual/equity_x20_L300_20260924_213026.csv`

| lookback_year | hold_year | n_eligible | n_picks | mean_pairwise_corr | symbols | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2023 | 13 | 13 | 0.0620 | UUP,SPY,UVXY,GLD,AEO,VXX,A,TLT,USO,AAPL,SHY,DBC,IEF | corr_n=13 |
| 2023 | 2024 | 300 | 20 | 0.0524 | MCK,TEAM,SJM,SMCI,TLT,PGR,NTES,MNST,PSX,ULTA,ARGX,T,CME,ANET,DG,LLY,MRNA,HUM,FCNCA,PDD | corr_n=300 |
| 2024 | 2025 | 300 | 20 | 0.0100 | GIS,ASML,IBKR,TLT,TPL,MCK,NVDA,DXCM,MMM,TSCO,CME,WMT,CTAS,PDD,KR,HUM,VZ,ANF,GILD,COP | corr_n=300 |

## Design

- Lookback = prior calendar year daily returns among history-ok then top-ADV names.
- Selection = greedy minimize mean pairwise Pearson corr.
- Hold = equal-weight next calendar year; lag-1 implement + RT costs.
- ALPACA daily is thin before ~2022; use `--ib-prefix` + early `--start` for longer history.
