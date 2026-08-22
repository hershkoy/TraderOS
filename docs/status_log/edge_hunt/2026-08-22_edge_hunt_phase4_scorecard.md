# Edge hunt Phase 4 - Blend_SPY70_BV30 robustness

Date: 2026-08-22

## Locked settings

- Blend: **70% SPY / 30% BigVol** (frozen)
- Sleeve: alloc=10%, max_pos=15, MA10 exit
- Costs test: **10 bps RT** on sleeve trades only
- Walk-forward IS: `2018-11-01` -> `2022-06-30`
- Walk-forward OOS: `2022-07-01` -> `2025-11-26`
- Stress: stop **20%** + 10bps RT
- Setups source: `weekly_bigvol_full_setups_20260822_104401.csv`

## Data note

Weekly BigVol setups CSV confirms begin ~2023-04. The IS half (through 2022-06) has an idle sleeve (30% cash at 0% yield + 70% SPY). OOS is the binding robustness test for the sleeve; IS still validates blend behavior with no satellite signals.

## Scorecard

| name | CAGR | Sharpe | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold_full_0bps | 13.64% | 0.68 | -34.10% | 0.40 | 20.07% | 146.80% | 100% | +0.00% | +0.00 | BENCHMARK |
| BigVol_sleeve_full_0bps | 8.82% | 0.84 | -12.51% | 0.71 | 10.50% | 81.77% | 36% | -4.81% | +0.16 | FAIL |
| Blend70_30_full_0bps | 12.32% | 0.79 | -25.33% | 0.49 | 15.63% | 127.29% | 100% | -1.32% | +0.11 | PASS (Sharpe+MDD+CAGR) |
| SPY_buy_hold_full_10bps | 13.64% | 0.68 | -34.10% | 0.40 | 20.07% | 146.80% | 100% | +0.00% | +0.00 | PASS (partial gate) |
| BigVol_sleeve_full_10bps | 8.57% | 0.82 | -12.66% | 0.68 | 10.50% | 78.76% | 36% | -5.07% | +0.14 | FAIL |
| Blend70_30_full_10bps | 12.26% | 0.78 | -25.33% | 0.48 | 15.63% | 126.39% | 100% | -1.38% | +0.10 | PASS (Sharpe+MDD+CAGR) |
| SPY_buy_hold_IS | 9.43% | 0.42 | -34.10% | 0.28 | 22.62% | 39.06% | 100% | +0.00% | +0.00 | FAIL |
| BigVol_sleeve_IS | 0.00% | 0.00 | 0.00% | 0.00 | 0.00% | 0.00% | 0% | -9.43% | -0.42 | FAIL |
| Blend70_30_IS | 6.83% | 0.42 | -25.33% | 0.27 | 16.38% | 27.34% | 100% | -2.60% | +0.00 | FAIL |
| SPY_buy_hold_OOS | 18.28% | 1.08 | -19.00% | 0.96 | 16.92% | 77.06% | 100% | +0.00% | +0.00 | PASS (Sharpe+MDD+CAGR) |
| BigVol_sleeve_OOS | 18.61% | 1.23 | -12.66% | 1.47 | 15.12% | 78.76% | 76% | +0.33% | +0.15 | PASS (Sharpe+MDD+CAGR) |
| Blend70_30_OOS | 18.38% | 1.28 | -17.12% | 1.07 | 14.35% | 77.57% | 100% | +0.10% | +0.20 | PASS (Sharpe+MDD+CAGR) |
| SPY_buy_hold_stress20_10bps | 13.64% | 0.68 | -34.10% | 0.40 | 20.07% | 146.80% | 100% | +0.00% | +0.00 | PASS (partial gate) |
| BigVol_sleeve_stress20_10bps | 8.45% | 0.79 | -11.66% | 0.72 | 10.65% | 77.41% | 36% | -5.19% | +0.11 | FAIL |
| Blend70_30_stress20_10bps | 12.23% | 0.78 | -25.33% | 0.48 | 15.67% | 125.98% | 100% | -1.41% | +0.10 | PASS (Sharpe+MDD+CAGR) |

## Test results

### 1) Costs (full window, 10bps RT)

- Blend 0bps vs SPY: CAGR=12.32% Sharpe=0.79 MDD=-25.33% -> PASS (Sharpe+MDD+CAGR)
- Blend 10bps vs SPY: CAGR=12.26% Sharpe=0.78 MDD=-25.33% -> PASS (Sharpe+MDD+CAGR)
- Costs demote rule (CAGR>2pp below AND Sharpe<=SPY): **clear**

### 2) Walk-forward (10bps RT, stop 15%)

- IS blend vs IS SPY: CAGR=6.83% Sharpe=0.42 MDD=-25.33% -> FAIL
- OOS blend vs OOS SPY: CAGR=18.38% Sharpe=1.28 MDD=-17.12% -> **OOS_PASS (Sharpe>=SPY)**

### 3) Stop 20% stress (full window, 10bps RT)

- Blend stress vs SPY: CAGR=12.23% Sharpe=0.78 MDD=-25.33% -> PASS (Sharpe+MDD+CAGR)
- Stress demote rule: **clear**

## Notes per candidate

- **SPY_buy_hold_full_0bps**: 100% SPY [2018-11-01->2025-11-26]
- **BigVol_sleeve_full_0bps**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=0bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend70_30_full_0bps**: spy=70% bv=30%; stop=15%; cost_rt=0bps; sleeve_invested=36%
- **SPY_buy_hold_full_10bps**: 100% SPY [2018-11-01->2025-11-26]
- **BigVol_sleeve_full_10bps**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend70_30_full_10bps**: spy=70% bv=30%; stop=15%; cost_rt=10bps; sleeve_invested=36%
- **SPY_buy_hold_IS**: 100% SPY [2018-11-01->2022-06-30]
- **BigVol_sleeve_IS**: no setups in window [2018-11-01->2022-06-30]; sleeve idle cash (0% yield)
- **Blend70_30_IS**: spy=70% bv=30%; stop=15%; cost_rt=10bps; sleeve_invested=0%
- **SPY_buy_hold_OOS**: 100% SPY [2022-07-01->2025-11-26]
- **BigVol_sleeve_OOS**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend70_30_OOS**: spy=70% bv=30%; stop=15%; cost_rt=10bps; sleeve_invested=76%
- **SPY_buy_hold_stress20_10bps**: 100% SPY [2018-11-01->2025-11-26]
- **BigVol_sleeve_stress20_10bps**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=20% + MA10 exit; cost_rt=10bps; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=0 ma_exits=169
- **Blend70_30_stress20_10bps**: spy=70% bv=30%; stop=20%; cost_rt=10bps; sleeve_invested=36%

## Timings

- full_0bps: 4.4s
- full_10bps: 4.6s
- IS: 0.0s
- OOS: 10.2s
- stress20: 4.6s
- total: 23.9s

## Verdict

**KEEP promote** for `Blend_SPY70_BV30`.

Costs, OOS, and stop-20% stress did not trip demote rules. Treat as capital-allocation on BigVol sleeve — next work may improve sleeve expectancy, not blend weights.
