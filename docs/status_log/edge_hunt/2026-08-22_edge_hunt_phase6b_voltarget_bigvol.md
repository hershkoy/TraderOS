# Edge hunt Phase 6b - VolTarget SPY + BigVol blend

Date: 2026-08-22

## Objective

- Replace raw SPY in Phase 4 `Blend_SPY70_BV30` with Phase 6 near-miss `VolTarget_12pct_20d_cap1`.
- BigVol sleeve frozen: alloc=10%, max_pos=15, stop=15%, MA10 exit, 10bps RT.
- Gate: Sharpe > 1.0, Sharpe >= SPY, MDD better than SPY.

## Window / data

- SPY IB n=4000; BIL=yes; BigVol setups=weekly_bigvol_full_setups_20260822_104401.csv; VT=12%/20d/cap1
- Full eval: `2018-01-02` -> `2025-11-25`
- IS: 2018-01-01 -> 2022-12-31; OOS: 2023-01-01 -> 2025-11-26

## Scorecard

| name | CAGR | Sharpe | Sortino | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | trades | WR | PF | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 12.37% | 0.63 | 0.98 | -34.10% | 0.36 | 19.61% | 151.15% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| SPY_buy_hold_IS | 7.32% | 0.34 | 0.52 | -34.10% | 0.21 | 21.62% | 42.29% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| SPY_buy_hold_OOS | 21.87% | 1.41 | 2.36 | -19.00% | 1.15 | 15.55% | 77.25% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| VolTarget_12pct_20d_cap1 | 9.63% | 0.79 | 1.24 | -15.94% | 0.60 | 12.15% | 106.67% | 80% | -2.74% | +0.16 | 0 | - | - | FAIL (Sharpe<=1.0) |
| VolTarget_12pct_20d_cap1_IS | 5.70% | 0.46 | 0.70 | -15.94% | 0.36 | 12.28% | 31.90% | 76% | -1.62% | +0.13 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| VolTarget_12pct_20d_cap1_OOS | 16.90% | 1.42 | 2.33 | -13.18% | 1.28 | 11.91% | 57.12% | 89% | -4.97% | +0.01 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| BigVol_sleeve_10bps | 7.63% | 0.77 | 1.23 | -12.66% | 0.60 | 9.93% | 78.76% | 0% | -4.74% | +0.14 | 0 | - | - | FAIL (Sharpe<=1.0) |
| Blend_SPY70_BV30 | 11.09% | 0.73 | 1.14 | -25.44% | 0.44 | 15.23% | 129.43% | 100% | -1.28% | +0.10 | 0 | - | - | FAIL (Sharpe<=1.0) |
| Blend_VT70_BV30 | 9.06% | 0.90 | 1.42 | -13.02% | 0.70 | 10.08% | 98.30% | 100% | -3.31% | +0.27 | 0 | - | - | FAIL (Sharpe<=1.0) |
| Blend_VT70_BV30_IS | 4.12% | 0.46 | 0.69 | -12.44% | 0.33 | 9.01% | 22.33% | 100% | -3.20% | +0.12 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| Blend_VT70_BV30_OOS | 18.25% | 1.56 | 2.59 | -13.02% | 1.40 | 11.69% | 62.44% | 100% | -3.62% | +0.15 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| Blend_VT80_BV20 | 9.25% | 0.86 | 1.36 | -13.70% | 0.68 | 10.72% | 101.09% | 100% | -3.12% | +0.23 | 0 | - | - | FAIL (Sharpe<=1.0) |
| Blend_VT80_BV20_IS | 4.66% | 0.46 | 0.70 | -13.70% | 0.34 | 10.13% | 25.52% | 100% | -2.66% | +0.12 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| Blend_VT80_BV20_OOS | 17.78% | 1.53 | 2.52 | -13.07% | 1.36 | 11.66% | 60.57% | 100% | -4.09% | +0.12 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |
| Blend_VT60_BV40 | 8.86% | 0.93 | 1.48 | -12.96% | 0.68 | 9.53% | 95.51% | 100% | -3.51% | +0.30 | 0 | - | - | FAIL (Sharpe<=1.0) |
| Blend_VT60_BV40_IS | 3.57% | 0.45 | 0.69 | -11.09% | 0.32 | 7.86% | 19.14% | 100% | -3.75% | +0.12 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| Blend_VT60_BV40_OOS | 18.74% | 1.58 | 2.64 | -12.96% | 1.45 | 11.86% | 64.40% | 100% | -3.13% | +0.17 | 0 | - | - | PASS (Sharpe>1 + Sharpe>=SPY + MDD) |

## Annual returns

| year | SPY_buy_hold | VolTarget_12pct_20d_cap1 | BigVol_sleeve_10bps | Blend_SPY70_BV30 | Blend_VT70_BV30 | Blend_VT80_BV20 | Blend_VT60_BV40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | -7.01% | -5.53% | 0.00% | -4.91% | -3.87% | -4.43% | -3.32% |
| 2019 | 28.79% | 20.00% | 0.00% | 19.70% | 13.76% | 15.82% | 11.73% |
| 2020 | 16.16% | 10.64% | 0.00% | 11.90% | 7.72% | 8.72% | 6.70% |
| 2021 | 27.04% | 21.12% | 0.00% | 20.67% | 15.74% | 17.61% | 13.79% |
| 2022 | -19.48% | -13.18% | 0.00% | -15.68% | -10.28% | -11.31% | -9.16% |
| 2023 | 24.29% | 18.68% | 6.92% | 20.27% | 15.80% | 16.81% | 14.73% |
| 2024 | 23.30% | 20.29% | 35.30% | 25.77% | 23.69% | 22.48% | 24.99% |
| 2025 | 15.18% | 9.76% | 23.57% | 17.03% | 13.18% | 11.98% | 14.44% |

## Notes

- **SPY_buy_hold**: 100% IB SPY
- **SPY_buy_hold_IS**: IS
- **SPY_buy_hold_OOS**: OOS
- **VolTarget_12pct_20d_cap1**: vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00
- **VolTarget_12pct_20d_cap1_IS**: IS vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00
- **VolTarget_12pct_20d_cap1_OOS**: OOS vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00
- **BigVol_sleeve_10bps**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_SPY70_BV30**: Phase4-style raw SPY 70% + BigVol 30%
- **Blend_VT70_BV30**: VT70%/BV30%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT70_BV30_IS**: IS VT70%/BV30%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT70_BV30_OOS**: OOS VT70%/BV30%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT80_BV20**: VT80%/BV20%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT80_BV20_IS**: IS VT80%/BV20%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT80_BV20_OOS**: OOS VT80%/BV20%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT60_BV40**: VT60%/BV40%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT60_BV40_IS**: IS VT60%/BV40%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_VT60_BV40_OOS**: OOS VT60%/BV40%; vol_target=12% lookback=20d cap=1x rebal=daily cash=BIL cost=5bps RT; avg_w=0.80 max_w=1.00; sleeve=setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164

## Timings

- load_spy_bil: 0.0s
- sim_voltarget: 0.0s
- sim_bigvol: 2m 31s
- total: 2m 31s

## Promotion

**No Phase-6b full-sample candidate cleared the gate.**

Near-miss (Sharpe>=SPY and better MDD, Sharpe<=1.0):
- `Blend_VT60_BV40`: CAGR=8.86% Sharpe=0.93 MDD=-12.96%
- `Blend_VT70_BV30`: CAGR=9.06% Sharpe=0.90 MDD=-13.02%
- `Blend_VT80_BV20`: CAGR=9.25% Sharpe=0.86 MDD=-13.70%
- `VolTarget_12pct_20d_cap1`: CAGR=9.63% Sharpe=0.79 MDD=-15.94%
- `BigVol_sleeve_10bps`: CAGR=7.63% Sharpe=0.77 MDD=-12.66%
- `Blend_SPY70_BV30`: CAGR=11.09% Sharpe=0.73 MDD=-25.44%

### OOS vs SPY_buy_hold_OOS

OOS passers: `VolTarget_12pct_20d_cap1_OOS` Sharpe=1.42, `Blend_VT70_BV30_OOS` Sharpe=1.56, `Blend_VT80_BV20_OOS` Sharpe=1.53, `Blend_VT60_BV40_OOS` Sharpe=1.58
