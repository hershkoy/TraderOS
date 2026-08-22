# Edge hunt Phase 6c - VIX term-structure overlay

Date: 2026-08-22

## Objective

- Unblock Phase 6 VIX curve overlay using IB `VIX` + `VIX3M` daily closes.
- Contango (VIX < VIX3M) = risk-on SPY; backwardation = risk-off (BIL).
- Also test soft weights and multiply by Phase 6 near-miss vol-target.
- Gate: Sharpe > 1.0, Sharpe >= SPY, MDD better than SPY.

## Window / data

- SPY IB n=4000; VIX IB n=4000; VIX3M IB n=4000; BIL=yes; cost=5bps RT
- Full eval: `2018-01-02` -> `2025-11-25`
- IS: 2018-01-01 -> 2022-12-31; OOS: 2023-01-01 -> 2025-11-26

## Scorecard

| name | CAGR | Sharpe | Sortino | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | trades | WR | PF | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 12.37% | 0.63 | 0.98 | -34.10% | 0.36 | 19.61% | 151.15% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| SPY_buy_hold_IS | 7.32% | 0.34 | 0.52 | -34.10% | 0.21 | 21.62% | 42.29% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| SPY_buy_hold_OOS | 21.87% | 1.41 | 2.36 | -19.00% | 1.15 | 15.55% | 77.25% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| VIXCurve_binary | 10.63% | 0.73 | 1.14 | -30.06% | 0.35 | 14.59% | 121.96% | 91% | -1.74% | +0.10 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VIXCurve_binary_IS | 4.96% | 0.31 | 0.48 | -30.06% | 0.16 | 15.78% | 27.30% | 89% | -2.37% | -0.02 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VIXCurve_binary_OOS | 21.36% | 1.74 | 2.95 | -10.29% | 2.08 | 12.25% | 75.09% | 95% | -0.52% | +0.34 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VIXCurve_binary_min20 | 10.77% | 0.73 | 1.14 | -30.06% | 0.36 | 14.72% | 124.32% | 92% | -1.60% | +0.10 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VIXCurve_binary_min20_IS | 5.18% | 0.32 | 0.49 | -30.06% | 0.17 | 15.97% | 28.66% | 90% | -2.14% | -0.01 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VIXCurve_binary_min20_OOS | 21.36% | 1.74 | 2.95 | -10.29% | 2.08 | 12.25% | 75.09% | 95% | -0.52% | +0.34 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VIXCurve_binary_min25 | 11.04% | 0.72 | 1.12 | -30.06% | 0.37 | 15.29% | 128.58% | 95% | -1.33% | +0.09 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VIXCurve_binary_min25_IS | 6.72% | 0.41 | 0.63 | -30.06% | 0.22 | 16.42% | 38.38% | 93% | -0.60% | +0.07 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VIXCurve_binary_min25_OOS | 19.11% | 1.46 | 2.35 | -12.19% | 1.57 | 13.10% | 65.89% | 97% | -2.76% | +0.05 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VIXCurve_soft | 12.24% | 0.66 | 1.02 | -31.61% | 0.39 | 18.59% | 148.79% | 99% | -0.13% | +0.03 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VIXCurve_soft_IS | 7.38% | 0.36 | 0.56 | -31.61% | 0.23 | 20.41% | 42.66% | 99% | +0.06% | +0.02 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VIXCurve_soft_OOS | 21.36% | 1.43 | 2.36 | -18.16% | 1.18 | 14.92% | 75.12% | 100% | -0.51% | +0.03 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VIXCurve_bin_x_VT12 | 8.55% | 0.78 | 1.22 | -19.76% | 0.43 | 10.91% | 91.18% | 76% | -3.82% | +0.15 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VIXCurve_bin_x_VT12_IS | 4.37% | 0.40 | 0.60 | -19.76% | 0.22 | 10.91% | 23.82% | 70% | -2.95% | +0.06 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VIXCurve_bin_x_VT12_OOS | 16.30% | 1.49 | 2.46 | -10.09% | 1.62 | 10.91% | 54.82% | 85% | -5.57% | +0.09 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VIXCurve_soft_x_VT12 | 9.53% | 0.80 | 1.24 | -15.97% | 0.60 | 11.96% | 105.18% | 80% | -2.84% | +0.17 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VIXCurve_soft_x_VT12_IS | 5.67% | 0.47 | 0.71 | -15.97% | 0.36 | 12.06% | 31.70% | 75% | -1.65% | +0.13 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VIXCurve_soft_x_VT12_OOS | 16.67% | 1.42 | 2.32 | -12.76% | 1.31 | 11.78% | 56.22% | 88% | -5.21% | +0.01 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| VIXCurve_bin20_x_VT12 | 8.57% | 0.78 | 1.19 | -19.76% | 0.43 | 11.05% | 91.42% | 76% | -3.80% | +0.14 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VIXCurve_bin20_x_VT12_IS | 4.40% | 0.40 | 0.59 | -19.76% | 0.22 | 11.13% | 23.98% | 71% | -2.92% | +0.06 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VIXCurve_bin20_x_VT12_OOS | 16.30% | 1.49 | 2.46 | -10.09% | 1.62 | 10.91% | 54.82% | 85% | -5.57% | +0.09 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |

## Annual returns

| year | SPY_buy_hold | VIXCurve_binary | VIXCurve_binary_min20 | VIXCurve_binary_min25 | VIXCurve_soft | VIXCurve_bin_x_VT12 | VIXCurve_soft_x_VT12 | VIXCurve_bin20_x_VT12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | -7.01% | -8.23% | -11.11% | -7.67% | -7.90% | -3.40% | -5.90% | -6.98% |
| 2019 | 28.79% | 17.45% | 20.13% | 24.39% | 28.27% | 11.98% | 19.66% | 14.14% |
| 2020 | 16.16% | 26.54% | 29.09% | 29.09% | 18.22% | 15.35% | 11.33% | 17.67% |
| 2021 | 27.04% | 23.94% | 23.94% | 23.94% | 26.97% | 18.49% | 21.06% | 18.49% |
| 2022 | -19.48% | -24.69% | -24.69% | -24.69% | -19.55% | -16.25% | -13.22% | -16.25% |
| 2023 | 24.29% | 24.29% | 24.29% | 24.29% | 24.29% | 18.68% | 18.68% | 18.68% |
| 2024 | 23.30% | 16.71% | 16.71% | 19.72% | 22.95% | 14.00% | 20.02% | 14.00% |
| 2025 | 15.18% | 20.20% | 20.20% | 11.02% | 14.12% | 14.12% | 9.37% | 14.12% |

## Notes

- **SPY_buy_hold**: 100% IB SPY
- **SPY_buy_hold_IS**: IS
- **SPY_buy_hold_OOS**: OOS
- **VIXCurve_binary**: vix_curve mode=binary thresh=1 cash=BIL cost=5bps RT; avg_w=0.91 frac_back=0.09
- **VIXCurve_binary_IS**: IS vix_curve mode=binary thresh=1 cash=BIL cost=5bps RT; avg_w=0.91 frac_back=0.09
- **VIXCurve_binary_OOS**: OOS vix_curve mode=binary thresh=1 cash=BIL cost=5bps RT; avg_w=0.91 frac_back=0.09
- **VIXCurve_binary_min20**: vix_curve mode=binary thresh=1 minVIX=20 cash=BIL cost=5bps RT; avg_w=0.92 frac_back=0.09
- **VIXCurve_binary_min20_IS**: IS vix_curve mode=binary thresh=1 minVIX=20 cash=BIL cost=5bps RT; avg_w=0.92 frac_back=0.09
- **VIXCurve_binary_min20_OOS**: OOS vix_curve mode=binary thresh=1 minVIX=20 cash=BIL cost=5bps RT; avg_w=0.92 frac_back=0.09
- **VIXCurve_binary_min25**: vix_curve mode=binary thresh=1 minVIX=25 cash=BIL cost=5bps RT; avg_w=0.95 frac_back=0.09
- **VIXCurve_binary_min25_IS**: IS vix_curve mode=binary thresh=1 minVIX=25 cash=BIL cost=5bps RT; avg_w=0.95 frac_back=0.09
- **VIXCurve_binary_min25_OOS**: OOS vix_curve mode=binary thresh=1 minVIX=25 cash=BIL cost=5bps RT; avg_w=0.95 frac_back=0.09
- **VIXCurve_soft**: vix_curve mode=soft thresh=1 cash=BIL cost=5bps RT; avg_w=0.99 frac_back=0.09
- **VIXCurve_soft_IS**: IS vix_curve mode=soft thresh=1 cash=BIL cost=5bps RT; avg_w=0.99 frac_back=0.09
- **VIXCurve_soft_OOS**: OOS vix_curve mode=soft thresh=1 cash=BIL cost=5bps RT; avg_w=0.99 frac_back=0.09
- **VIXCurve_bin_x_VT12**: vix_curve mode=binary thresh=1 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.76 frac_back=0.09
- **VIXCurve_bin_x_VT12_IS**: IS vix_curve mode=binary thresh=1 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.76 frac_back=0.09
- **VIXCurve_bin_x_VT12_OOS**: OOS vix_curve mode=binary thresh=1 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.76 frac_back=0.09
- **VIXCurve_soft_x_VT12**: vix_curve mode=soft thresh=1 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.80 frac_back=0.09
- **VIXCurve_soft_x_VT12_IS**: IS vix_curve mode=soft thresh=1 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.80 frac_back=0.09
- **VIXCurve_soft_x_VT12_OOS**: OOS vix_curve mode=soft thresh=1 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.80 frac_back=0.09
- **VIXCurve_bin20_x_VT12**: vix_curve mode=binary thresh=1 minVIX=20 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.76 frac_back=0.09
- **VIXCurve_bin20_x_VT12_IS**: IS vix_curve mode=binary thresh=1 minVIX=20 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.76 frac_back=0.09
- **VIXCurve_bin20_x_VT12_OOS**: OOS vix_curve mode=binary thresh=1 minVIX=20 vt=12%/20d/cap1 cash=BIL cost=5bps RT; avg_w=0.76 frac_back=0.09

## Timings

- load: 20.2s
- sim_VIXCurve_binary: 0.0s
- sim_VIXCurve_binary_min20: 0.0s
- sim_VIXCurve_binary_min25: 0.0s
- sim_VIXCurve_soft: 0.0s
- sim_VIXCurve_bin_x_VT12: 0.0s
- sim_VIXCurve_soft_x_VT12: 0.0s
- sim_VIXCurve_bin20_x_VT12: 0.0s
- total: 20.3s

## Promotion

**No Phase-6c full-sample candidate cleared the risk-adjusted gate.**

Near-miss (Sharpe>=SPY and better MDD, but Sharpe<=1.0):
- `VIXCurve_soft_x_VT12`: CAGR=9.53% Sharpe=0.80 MDD=-15.97%
- `VIXCurve_bin_x_VT12`: CAGR=8.55% Sharpe=0.78 MDD=-19.76%
- `VIXCurve_bin20_x_VT12`: CAGR=8.57% Sharpe=0.78 MDD=-19.76%
- `VIXCurve_binary_min20`: CAGR=10.77% Sharpe=0.73 MDD=-30.06%
- `VIXCurve_binary`: CAGR=10.63% Sharpe=0.73 MDD=-30.06%
- `VIXCurve_binary_min25`: CAGR=11.04% Sharpe=0.72 MDD=-30.06%
- `VIXCurve_soft`: CAGR=12.24% Sharpe=0.66 MDD=-31.61%

Next: keep Phase 6b `Blend_VT60_BV40` as research near-KEEP; do not fish VIX thresholds further unless a clear structural edge appears.

### OOS robustness (vs SPY_buy_hold_OOS)

OOS passers: `VIXCurve_binary_min20_OOS` Sharpe=1.74 MDD=-10.29%, `VIXCurve_binary_OOS` Sharpe=1.74 MDD=-10.29%, `VIXCurve_bin20_x_VT12_OOS` Sharpe=1.49 MDD=-10.09%, `VIXCurve_bin_x_VT12_OOS` Sharpe=1.49 MDD=-10.09%, `VIXCurve_binary_min25_OOS` Sharpe=1.46 MDD=-12.19%, `VIXCurve_soft_OOS` Sharpe=1.43 MDD=-18.16%, `VIXCurve_soft_x_VT12_OOS` Sharpe=1.42 MDD=-12.76%
