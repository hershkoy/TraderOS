# Edge hunt Phase 1 - scorecard vs SPY buy-and-hold

Date: 2026-08-22

## Window / data

- Evaluation window: `2018-11-01` -> `2025-11-25`
- SPY ALPACA 1d bars used: 2018-11-01 -> 2025-11-25 (n=1343). Earlier than ~2018-11 may be unavailable on current Alpaca feed.
- Edge gate: Sharpe >= SPY and/or better MDD with CAGR within ~2pp of SPY (or higher CAGR).
- Cross-sectional results use current SPX membership (survivorship bias) - optimistic.

## Scorecard

| name | CAGR | Sharpe | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 13.75% | 0.73 | -25.38% | 0.54 | 18.89% | 148.53% | 100% | +0.00% | +0.00 | BENCHMARK |
| SPY_SMA200_abs_mom | 3.60% | 0.33 | -25.88% | 0.14 | 11.05% | 28.43% | 65% | -10.15% | -0.40 | FAIL |
| XS_mom_12_1_gross | 6.16% | 0.34 | -29.03% | 0.21 | 18.37% | 52.55% | 43% | -7.59% | -0.39 | FAIL |
| XS_mom_12_1_net_10bps | 6.02% | 0.33 | -29.09% | 0.21 | 18.37% | 51.18% | 43% | -7.73% | -0.40 | FAIL |
| WeeklyBigVol_portfolio | 8.82% | 0.73 | -12.51% | 0.71 | 12.07% | 81.77% | 48% | -4.93% | +0.00 | FAIL |

## Notes per candidate

- **SPY_buy_hold**: 100% SPY
- **SPY_SMA200_abs_mom**: Month-end close>SMA200; cash=0%; next-day position
- **XS_mom_12_1_gross**: universe=501 (SPX_intersect_ALPACA_1d); survivorship bias (current SPX list); top_n=20; cost=10bps RT; rebalances=29 [gross]
- **XS_mom_12_1_net_10bps**: universe=501 (SPX_intersect_ALPACA_1d); survivorship bias (current SPX list); top_n=20; cost=10bps RT; rebalances=29 [net]
- **WeeklyBigVol_portfolio**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164

## Timings

- load_spy: 0.0s
- bh: 0.0s
- abs: 0.0s
- xs: 14.1s
- bigvol: 4.4s
- total: 18.5s

## Promotion

**No Phase-1 candidate cleared gates.**

Near-miss (much better MDD, but CAGR too far below SPY):
- `WeeklyBigVol_portfolio`: CAGR=8.82% Sharpe=0.73 MDD=-12.51% (invested 48%)

Next data to add for Phase 2 dual momentum: daily **SHY** (or BIL) and **EFA**/**VXUS**, plus ideally longer SPY history pre-2018 if available.
