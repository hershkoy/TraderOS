# Edge hunt Phase 6 - Portfolio overlays (vol-target / CTA / low-vol)

Date: 2026-08-22

## Objective

- Gate (unchanged): Sharpe **> 1.0**, Sharpe **>= SPY**, and **MDD better than SPY**.
- Shift from single-stock signal fishing (Phase 5 FAIL) to portfolio construction: vol targeting, multi-asset trend, low-vol basket.
- Incumbent: Phase 4 KEEP `Blend_SPY70_BV30`.

## Window / data

- SPY IB n=4000; BIL=yes; macros=['BIL', 'DBC', 'GLD', 'IEF', 'SPY', 'TLT', 'USO', 'UUP']; CTA assets=['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC']
- Full eval: `2018-01-02` -> `2025-11-25`
- IS: `2018-01-01` -> `2022-12-31`; OOS: `2023-01-01` -> `2025-11-26`
- VIX term-structure overlay: **skipped** (no VIX/VXV / VIX futures in TimescaleDB).
- Quality/low-vol: **realized-vol only** (no ROE/FCF fundamentals in `market_data`).
- Gate column in the table compares every row to **full-sample** SPY; formal promotion uses full-sample rows only. OOS is judged vs `SPY_buy_hold_OOS` in Promotion.

## Scorecard

| name | CAGR | Sharpe | Sortino | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | trades | WR | PF | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 12.37% | 0.63 | 0.98 | -34.10% | 0.36 | 19.61% | 151.15% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| SPY_buy_hold_IS | 7.32% | 0.34 | 0.52 | -34.10% | 0.21 | 21.62% | 42.29% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| SPY_buy_hold_OOS | 21.87% | 1.41 | 2.36 | -19.00% | 1.15 | 15.55% | 77.25% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| VolTarget_12pct_20d_cap1 | 9.63% | 0.79 | 1.24 | -15.94% | 0.60 | 12.15% | 106.67% | 80% | -2.74% | +0.16 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VolTarget_12pct_20d_cap1_IS | 5.70% | 0.46 | 0.70 | -15.94% | 0.36 | 12.28% | 31.90% | 76% | -1.62% | +0.13 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VolTarget_12pct_20d_cap1_OOS | 16.90% | 1.42 | 2.33 | -13.18% | 1.28 | 11.91% | 57.12% | 89% | -4.97% | +0.01 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VolTarget_10pct_20d_cap1 | 8.38% | 0.78 | 1.21 | -13.49% | 0.62 | 10.75% | 88.76% | 73% | -3.99% | +0.15 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VolTarget_10pct_20d_cap1_IS | 5.08% | 0.47 | 0.71 | -13.49% | 0.38 | 10.82% | 28.08% | 68% | -2.24% | +0.13 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VolTarget_10pct_20d_cap1_OOS | 14.43% | 1.36 | 2.21 | -11.98% | 1.20 | 10.63% | 47.70% | 80% | -7.44% | -0.05 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VolTarget_12pct_20d_cap1p5 | 9.89% | 0.73 | 1.12 | -16.69% | 0.59 | 13.54% | 110.59% | 93% | -2.48% | +0.10 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VolTarget_12pct_20d_cap1p5_IS | 5.97% | 0.44 | 0.66 | -16.69% | 0.36 | 13.66% | 33.58% | 88% | -1.35% | +0.10 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VolTarget_12pct_20d_cap1p5_OOS | 17.14% | 1.29 | 2.05 | -15.23% | 1.13 | 13.34% | 58.08% | 100% | -4.73% | -0.12 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VolTarget_12pct_60d_cap1 | 7.72% | 0.60 | 0.91 | -20.62% | 0.37 | 12.94% | 79.92% | 78% | -4.65% | -0.03 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VolTarget_12pct_60d_cap1_IS | 3.61% | 0.27 | 0.41 | -20.62% | 0.18 | 13.27% | 19.37% | 74% | -3.71% | -0.07 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VolTarget_12pct_60d_cap1_OOS | 15.31% | 1.24 | 1.99 | -15.31% | 1.00 | 12.35% | 51.04% | 86% | -6.56% | -0.17 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| CTA_sma200_sleeve | -1.17% | -0.18 | -0.26 | -25.84% | -0.05 | 6.54% | -8.86% | 100% | -13.54% | -0.81 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| CTA_sma200_sleeve_IS | 0.65% | 0.10 | 0.14 | -10.04% | 0.07 | 6.33% | 3.30% | 100% | -6.67% | -0.24 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| CTA_sma200_sleeve_OOS | -4.38% | -0.64 | -1.03 | -19.77% | -0.22 | 6.88% | -12.15% | 100% | -26.25% | -2.04 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| Blend_SPY70_CTA30 | 9.39% | 0.64 | 0.99 | -25.44% | 0.37 | 14.72% | 103.15% | 100% | -2.98% | +0.01 | 0 | - | - | FAIL (Sharpe<=1.0) |
| Blend_SPY70_CTA30_IS | 5.49% | 0.35 | 0.53 | -25.44% | 0.22 | 15.61% | 30.59% | 100% | -1.83% | +0.01 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| Blend_SPY70_CTA30_OOS | 16.58% | 1.27 | 2.12 | -16.76% | 0.99 | 13.02% | 55.90% | 100% | -5.29% | -0.13 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| LowVol60_top50 | 2.21% | 0.28 | 0.40 | -12.88% | 0.17 | 7.91% | 18.87% | 40% | -10.16% | -0.35 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| LowVol60_top50_IS | 1.36% | 0.23 | 0.31 | -8.75% | 0.16 | 5.85% | 6.97% | 5% | -5.96% | -0.11 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| LowVol60_top50_OOS | 3.75% | 0.36 | 0.58 | -11.79% | 0.32 | 10.56% | 11.26% | 100% | -18.12% | -1.05 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |

## Annual returns

| year | SPY_buy_hold | VolTarget_12pct_20d_cap1 | VolTarget_10pct_20d_cap1 | VolTarget_12pct_20d_cap1p5 | VolTarget_12pct_60d_cap1 | CTA_sma200_sleeve | Blend_SPY70_CTA30 | LowVol60_top50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | -7.01% | -5.53% | -3.81% | -4.00% | -5.92% | 0.00% | -4.91% | 0.00% |
| 2019 | 28.79% | 20.00% | 16.01% | 18.85% | 19.31% | 0.00% | 19.70% | 0.00% |
| 2020 | 16.16% | 10.64% | 9.76% | 12.87% | -0.09% | 0.00% | 11.90% | 0.00% |
| 2021 | 27.04% | 21.12% | 17.50% | 19.46% | 23.10% | -2.61% | 20.05% | 0.00% |
| 2022 | -19.48% | -13.18% | -10.99% | -13.18% | -13.53% | 6.07% | -14.60% | 6.97% |
| 2023 | 24.29% | 18.68% | 15.94% | 19.18% | 17.96% | -8.76% | 16.44% | -0.93% |
| 2024 | 23.30% | 20.29% | 16.61% | 19.28% | 21.38% | -2.13% | 18.58% | 10.26% |
| 2025 | 15.18% | 9.76% | 9.00% | 10.90% | 5.27% | -1.20% | 12.66% | 1.74% |

## Notes per candidate

- **SPY_buy_hold**: 100% IB SPY
- **SPY_buy_hold_IS**: IS
- **SPY_buy_hold_OOS**: OOS
- **VolTarget_12pct_20d_cap1**: vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00
- **VolTarget_12pct_20d_cap1_IS**: IS vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00
- **VolTarget_12pct_20d_cap1_OOS**: OOS vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00
- **VolTarget_10pct_20d_cap1**: vol_target=10% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.73 max_w=1.00
- **VolTarget_10pct_20d_cap1_IS**: IS vol_target=10% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.73 max_w=1.00
- **VolTarget_10pct_20d_cap1_OOS**: OOS vol_target=10% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.73 max_w=1.00
- **VolTarget_12pct_20d_cap1p5**: vol_target=12% lookback=20d cap=1.5x rebal=daily cash=BIL cost=5bps RT; avg_w=0.93 max_w=1.50
- **VolTarget_12pct_20d_cap1p5_IS**: IS vol_target=12% lookback=20d cap=1.5x rebal=daily cash=BIL cost=5bps RT; avg_w=0.93 max_w=1.50
- **VolTarget_12pct_20d_cap1p5_OOS**: OOS vol_target=12% lookback=20d cap=1.5x rebal=daily cash=BIL cost=5bps RT; avg_w=0.93 max_w=1.50
- **VolTarget_12pct_60d_cap1**: vol_target=12% lookback=60d cap=1x rebal=weekly cash=BIL cost=5bps RT; avg_w=0.78 max_w=1.00
- **VolTarget_12pct_60d_cap1_IS**: IS vol_target=12% lookback=60d cap=1x rebal=weekly cash=BIL cost=5bps RT; avg_w=0.78 max_w=1.00
- **VolTarget_12pct_60d_cap1_OOS**: OOS vol_target=12% lookback=60d cap=1x rebal=weekly cash=BIL cost=5bps RT; avg_w=0.78 max_w=1.00
- **CTA_sma200_sleeve**: CTA sma200 equal_w n=7 assets=['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC'] rebal=monthly cost=10bps RT; avg_gross_exp=0.84
- **CTA_sma200_sleeve_IS**: IS CTA sma200 equal_w n=7 assets=['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC'] rebal=monthly cost=10bps RT; avg_gross_exp=0.84
- **CTA_sma200_sleeve_OOS**: OOS CTA sma200 equal_w n=7 assets=['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC'] rebal=monthly cost=10bps RT; avg_gross_exp=0.84
- **Blend_SPY70_CTA30**: spy70/cta30; CTA sma200 equal_w n=7 assets=['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC'] rebal=monthly cost=10bps RT; avg_gross_exp=0.84
- **Blend_SPY70_CTA30_IS**: IS spy70/cta30; CTA sma200 equal_w n=7 assets=['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC'] rebal=monthly cost=10bps RT; avg_gross_exp=0.84
- **Blend_SPY70_CTA30_OOS**: OOS spy70/cta30; CTA sma200 equal_w n=7 assets=['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC'] rebal=monthly cost=10bps RT; avg_gross_exp=0.84
- **LowVol60_top50**: low_vol60d hold_n=50 liquid=500 inv_vol=True cost=10bps RT; rebalances=38
- **LowVol60_top50_IS**: IS low_vol60d hold_n=50 liquid=500 inv_vol=True cost=10bps RT; rebalances=38
- **LowVol60_top50_OOS**: OOS low_vol60d hold_n=50 liquid=500 inv_vol=True cost=10bps RT; rebalances=38

## Timings

- load_spy_bil: 0.0s
- sim_VolTarget_12pct_20d_cap1: 0.0s
- sim_VolTarget_10pct_20d_cap1: 0.0s
- sim_VolTarget_12pct_20d_cap1p5: 0.0s
- sim_VolTarget_12pct_60d_cap1: 0.0s
- load_macros: 8.0s
- sim_cta: 0.0s
- load_panel: 8.8s
- sim_lowvol: 0.7s
- total: 17.5s

## Data gaps

- CTA assets used: ['SPY', 'TLT', 'GLD', 'USO', 'UUP', 'IEF', 'DBC']

- VIX term-structure overlay skipped: no VIX/VXV in market_data.

## Promotion

**No Phase-6 full-sample candidate cleared the risk-adjusted gate.**

Near-miss (Sharpe>=SPY and better MDD, but Sharpe<=1.0):
- `VolTarget_12pct_20d_cap1`: CAGR=9.63% Sharpe=0.79 MDD=-15.94%
- `VolTarget_10pct_20d_cap1`: CAGR=8.38% Sharpe=0.78 MDD=-13.49%
- `VolTarget_12pct_20d_cap1p5`: CAGR=9.89% Sharpe=0.73 MDD=-16.69%
- `Blend_SPY70_CTA30`: CAGR=9.39% Sharpe=0.64 MDD=-25.44%

Next: tighten vol-target (or blend with Phase 4 BigVol) to push full-sample Sharpe above 1.0; do not re-open Phase 5 stock-signal grids. VIX curve still needs futures data.

### OOS robustness (vs SPY_buy_hold_OOS)

OOS passers: `VolTarget_12pct_20d_cap1_OOS` Sharpe=1.42 MDD=-13.18%

