# Deep Pullback universe scan (2026-09-17)

QS rules on TimescaleDB daily (ALPACA + IB prefix). Equal-weight trade pool (not portfolio occupancy).

- Friction: **0.00%** round-turn
- Symbols: **306** (indexes=6 big=150 random=150)
- Start: `2006-01-01T00:00:00`  seed=42
- Wall clock: load **27.8s** / backtest **12.6s** / total **40.3s**

## Cohort books

| Cohort | n | E%% | PF | geoPF(n>=30) | geoPF(n>=10) | WR%% | median%% | syms w/ trades |
|--------|--:|---:|---:|-------------:|-------------:|----:|---------:|---------------:|
| index | 484 | 0.471 | 1.955 | 1.49 | 1.351 | 73.97 | 0.868 | 6/6 |
| big | 3060 | 0.907 | 1.745 | 2.118 | 2.118 | 67.55 | 1.278 | 149/150 |
| random | 2612 | 0.61 | 1.41 | 1.376 | 1.376 | 66.46 | 1.179 | 150/150 |
| all | 6156 | 0.747 | 1.585 | 1.678 | 1.678 | 67.59 | 1.173 | 305/306 |

## Indexes

| Sym | n | E%% | PF | WR%% | median | bars | range |
|-----|--:|---:|---:|----:|-------:|-----:|-------|
| MDY | 93 | 0.353 | 1.592 | 75.27 | 0.813 | 5209 | 2006-01-03 -> 2026-09-17 |
| VTI | 91 | 0.491 | 2.094 | 75.82 | 0.884 | 5209 | 2006-01-03 -> 2026-09-17 |
| IWM | 86 | 0.472 | 1.791 | 69.77 | 0.856 | 5209 | 2006-01-03 -> 2026-09-17 |
| DIA | 82 | 0.168 | 1.304 | 70.73 | 0.565 | 5209 | 2006-01-03 -> 2026-09-17 |
| QQQ | 66 | 0.696 | 2.513 | 72.73 | 1.158 | 4021 | 2010-08-26 -> 2026-08-21 |
| SPY | 66 | 0.758 | 4.288 | 80.3 | 0.95 | 3760 | 2010-01-04 -> 2026-09-03 |

## Year buckets (all)

- **2018-2019**: n=312 E=0.982 PF=2.297 WR=71.79
- **2020-2021**: n=1893 E=1.08 PF=1.913 WR=70.89
- **2022-2023**: n=1280 E=0.262 PF=1.162 WR=62.34
- **2024-2026**: n=2395 E=0.738 PF=1.552 WR=66.1

## Artifacts

- `D:/WORK/Projs/IB/backTraderTest/reports/research/deep_pullback/2026-09-17/deep_pullback_universe_2026-09-17_per_symbol.csv`
- `D:/WORK/Projs/IB/backTraderTest/reports/research/deep_pullback/2026-09-17/deep_pullback_universe_2026-09-17_trades.csv`
- `D:/WORK/Projs/IB/backTraderTest/reports/research/deep_pullback/2026-09-17/deep_pullback_universe_2026-09-17_summary.json`

Treated as a selective sleeve check, not Sharpe>1 edge-hunt promote.
