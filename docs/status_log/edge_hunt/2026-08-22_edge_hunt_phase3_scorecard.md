# Edge hunt Phase 3 - SPY core + Weekly BigVol satellite

Date: 2026-08-22

## Window / data

- Evaluation window: `2018-11-01` -> `2025-11-25`
- SPY bars 2018-11-01 -> 2025-11-25 (n=1776); BigVol setups=weekly_bigvol_full_setups_20260822_104401.csv; sleeve alloc=10% max_pos=15; idle sleeve cash earns 0%
- Rule: fixed capital split SPY B&H + BigVol sleeve; idle sleeve cash earns 0%.
- Default split 70/30; sensitivity 80/20 and 60/40.
- Gate: Sharpe >= SPY and/or better MDD with CAGR within ~2pp (or higher CAGR).

## Scorecard

| name | CAGR | Sharpe | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 13.64% | 0.68 | -34.10% | 0.40 | 20.07% | 146.80% | 100% | +0.00% | +0.00 | BENCHMARK |
| BigVol_sleeve_100pct | 8.82% | 0.84 | -12.51% | 0.71 | 10.50% | 81.77% | 36% | -4.81% | +0.16 | FAIL |
| Blend_SPY70_BV30 | 12.32% | 0.79 | -25.33% | 0.49 | 15.63% | 127.29% | 100% | -1.32% | +0.11 | PASS (Sharpe+MDD+CAGR) |
| Blend_SPY80_BV20 | 12.77% | 0.75 | -28.37% | 0.45 | 17.09% | 133.79% | 100% | -0.87% | +0.07 | PASS (Sharpe+MDD+CAGR) |
| Blend_SPY60_BV40 | 11.86% | 0.83 | -22.16% | 0.54 | 14.21% | 120.79% | 100% | -1.78% | +0.16 | PASS (Sharpe+MDD+CAGR) |

## Notes per candidate

- **SPY_buy_hold**: 100% SPY
- **BigVol_sleeve_100pct**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164
- **Blend_SPY70_BV30**: spy_weight=70% bigvol_weight=30%; sleeve_invested_mean=36%
- **Blend_SPY80_BV20**: spy_weight=80% bigvol_weight=20%; sleeve_invested_mean=36%
- **Blend_SPY60_BV40**: spy_weight=60% bigvol_weight=40%; sleeve_invested_mean=36%

## Timings

- load_spy: 6.8s
- bigvol_sleeve: 4.6s
- Blend_SPY70_BV30: 0.0s
- Blend_SPY80_BV20: 0.0s
- Blend_SPY60_BV40: 0.0s
- total: 11.4s

## Promotion

**Promoted (Phase 3):** `Blend_SPY70_BV30` (plan default) - PASS (Sharpe+MDD+CAGR)

- CAGR=12.32% (within ~1.3pp of SPY 13.64%)
- Sharpe=0.79 vs SPY 0.68
- MDD=-25.33% vs SPY -34.10%

**Also PASS (same signal, weight sensitivity only):**
- `Blend_SPY80_BV20`: CAGR 12.77%, Sharpe 0.75, MDD -28.37%
- `Blend_SPY60_BV40`: CAGR 11.86%, Sharpe 0.83, MDD -22.16% (best Sharpe / lowest MDD among the three)

Pure `BigVol_sleeve_100pct` still FAIL on CAGR gate (8.82% vs 13.64%) despite better Sharpe/MDD — the blend is what clears the bar.

Same BigVol signal across weights only — treat as **capital-allocation** result, not a new alpha. Next robustness (if continuing): walk-forward / costs on the BigVol sleeve, not further weight fishing.
