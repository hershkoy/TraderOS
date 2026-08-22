# Edge hunt Phase 5 - XS mom delta + swing mean reversion

Date: 2026-08-22

## Objective

- Gate: Sharpe **> 1.0**, Sharpe **>= SPY**, and **MDD better than SPY** (stricter than Phase 1-4). CAGR is reported, not the ranking objective.
- Hold: swing / multi-day primary; 12-1 monthly is a one-shot delta vs Phase 1 (PIT top-500 ADV + SPY>SMA200), not a lookback/top-N grid.
- Incumbent (not the gate): Phase 4 KEEP `Blend_SPY70_BV30`.

## Window / data

- ALPACA 1d panel 2205 names x 1343 dates; SPY n=2238 2017-01-03 -> 2025-11-25; IS freeze winner=`dump3_stock` (min_IS_trades=20)
- Full eval: `2018-01-02` -> `2025-11-25`
- IS: `2018-01-01` -> `2022-12-31` (clipped to available bars)
- OOS: `2023-01-01` -> `2025-11-26`
- Costs: 10 bps round-trip. Universe: ALPACA daily panel, PIT 30d dollar volume.
- Coverage caveat: ALPACA `1d` bars for most names begin ~2020-2022 (SPY itself from 2018-11). The wide panel index is the union of those dates, so 2018-2019 is mostly empty. SPY SMA200 is computed on IB SPY (2017+) then aligned.
- Do not re-run: vanilla SPX 12-1, SPY-only SMA200, dual mom, BigVol standalone, blend weights.

## Scorecard

| name | CAGR | Sharpe | Sortino | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | trades | WR | PF | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 12.37% | 0.63 | 0.98 | -34.10% | 0.36 | 19.61% | 151.15% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| XS_mom_12_1_PIT500_SMA200_10bps | 3.04% | 0.14 | 0.21 | -32.44% | 0.09 | 21.31% | 26.72% | 26% | -9.33% | -0.49 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| SPY_buy_hold_IS | 7.32% | 0.34 | 0.52 | -34.10% | 0.21 | 21.62% | 42.29% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| XS_mom_IS | 0.00% | 0.00 | 0.00 | 0.00% | 0.00 | 0.00% | 0.00% | 0% | -7.32% | -0.34 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| SPY_buy_hold_OOS | 21.87% | 1.41 | 2.36 | -19.00% | 1.15 | 15.55% | 77.25% | 100% | +0.00% | +0.00 | 0 | - | - | BENCHMARK |
| XS_mom_OOS | 8.53% | 0.24 | 0.37 | -32.44% | 0.26 | 35.21% | 26.72% | 71% | -13.34% | -1.16 | 0 | - | - | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_dump3_stock | 5.50% | 0.28 | 0.46 | -27.95% | 0.20 | 19.56% | 52.59% | 53% | -6.87% | -0.35 | 2154 | 51% | 1.12 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_dump3_spy200 | 1.47% | 0.10 | 0.15 | -27.02% | 0.05 | 15.12% | 12.24% | 42% | -10.90% | -0.53 | 1716 | 50% | 1.06 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_rsi2_stock | -5.06% | -0.41 | -0.59 | -46.37% | -0.11 | 12.46% | -33.66% | 55% | -17.43% | -1.04 | 2974 | 34% | 0.91 | FAIL (Sharpe<=1.0, Sharpe<SPY, MDD>=SPY) |
| MR_rsi2_spy200 | -4.49% | -0.45 | -0.63 | -33.17% | -0.14 | 9.98% | -30.40% | 47% | -16.86% | -1.08 | 2389 | 34% | 0.89 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_dump3_stock_IS | -0.95% | -0.09 | -0.13 | -17.64% | -0.05 | 10.91% | -4.66% | 25% | -8.27% | -0.43 | 333 | 50% | 0.96 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_dump3_spy200_IS | -0.18% | -0.11 | -0.13 | -5.14% | -0.04 | 1.66% | -0.91% | 12% | -7.50% | -0.45 | 46 | 59% | 0.89 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_rsi2_stock_IS | -3.68% | -0.49 | -0.68 | -28.96% | -0.13 | 7.52% | -17.08% | 29% | -11.01% | -0.83 | 511 | 34% | 0.75 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_rsi2_spy200_IS | -0.80% | -0.53 | -0.63 | -6.96% | -0.11 | 1.50% | -3.91% | 19% | -8.12% | -0.87 | 104 | 36% | 0.68 | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| MR_dump3_stock_OOS | 18.42% | 0.64 | 1.14 | -27.95% | 0.66 | 28.94% | 63.12% | 100% | -3.45% | -0.77 | 1827 | 51% | 1.15 | FAIL (Sharpe<=1.0) |

## Annual returns

| year | SPY_buy_hold | XS_mom_12_1_PIT500_SMA200_10bps | MR_dump3_stock | MR_dump3_spy200 | MR_rsi2_stock | MR_rsi2_spy200 |
| --- | --- | --- | --- | --- | --- | --- |
| 2018 | -7.01% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 2019 | 28.79% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 2020 | 16.16% | 0.00% | 0.19% | 0.19% | 2.68% | 2.68% |
| 2021 | 27.04% | 0.00% | 1.22% | 1.22% | -2.24% | -2.24% |
| 2022 | -19.48% | 0.00% | -5.99% | -2.29% | -17.39% | -4.27% |
| 2023 | 24.29% | -6.55% | 46.70% | 35.89% | -5.18% | -14.55% |
| 2024 | 23.30% | 23.53% | 1.25% | 1.25% | -6.74% | -6.56% |
| 2025 | 15.18% | 9.77% | 7.74% | -17.68% | -9.52% | -9.28% |

## Notes per candidate

- **SPY_buy_hold**: 100% IB/ALPACA SPY
- **XS_mom_12_1_PIT500_SMA200_10bps**: PIT top500 ADV30d min_px=5 min_adv=1000000; 12-1m top_n=20; SPY>SMA200; cost=10bps RT; rebalances_on=25/37
- **SPY_buy_hold_IS**: IS window
- **XS_mom_IS**: IS PIT top500 ADV30d min_px=5 min_adv=1000000; 12-1m top_n=20; SPY>SMA200; cost=10bps RT; rebalances_on=25/37
- **SPY_buy_hold_OOS**: OOS window
- **XS_mom_OOS**: OOS PIT top500 ADV30d min_px=5 min_adv=1000000; 12-1m top_n=20; SPY>SMA200; cost=10bps RT; rebalances_on=25/37
- **MR_dump3_stock**: variant=dump3_stock max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 611, 'time': 1027, 'stop': 501, 'eod': 15}
- **MR_dump3_spy200**: variant=dump3_spy200 max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 461, 'time': 859, 'stop': 381, 'eod': 15}
- **MR_rsi2_stock**: variant=rsi2_stock max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 1967, 'time': 915, 'stop': 77, 'eod': 15}
- **MR_rsi2_spy200**: variant=rsi2_spy200 max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 1587, 'time': 734, 'stop': 53, 'eod': 15}
- **MR_dump3_stock_IS**: IS variant=dump3_stock max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 106, 'time': 118, 'stop': 94, 'eod': 15}
- **MR_dump3_spy200_IS**: IS variant=dump3_spy200 max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 18, 'time': 21, 'stop': 7, 'eod': 0}
- **MR_rsi2_stock_IS**: IS variant=rsi2_stock max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 342, 'time': 141, 'stop': 13, 'eod': 15}
- **MR_rsi2_spy200_IS**: IS variant=rsi2_spy200 max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 68, 'time': 36, 'stop': 0, 'eod': 0}
- **MR_dump3_stock_OOS**: OOS freeze variant=dump3_stock max_pos=15 hold=8d stop=10% cost=10bps RT liquid=500 min_px=10; exits={'sma20': 501, 'time': 904, 'stop': 407, 'eod': 15}

## Timings

- list_symbols: 8.9s
- load_panel: 0.3s
- load_spy: 0.0s
- sim_dump3_stock: 1.3s
- sim_dump3_spy200: 1.2s
- sim_rsi2_stock: 1.6s
- sim_rsi2_spy200: 1.4s
- sim_dump3_stock_IS: 0.7s
- sim_dump3_spy200_IS: 0.6s
- sim_rsi2_stock_IS: 0.9s
- sim_rsi2_spy200_IS: 0.9s
- sim_winner_OOS: 1.2s
- total: 19.0s (swing screen, panel cache hit)
- first_panel_build (TimescaleDB, 2205 symbols, cache miss): 347s
- xs_mom_delta cached rerun: ~32s (pivot 2205 symbol parquets + sim)

## Phase 5a (12-1 PIT+SMA200)

- `XS_mom_12_1_PIT500_SMA200_10bps`: CAGR=3.04% Sharpe=0.14 MDD=-32.44% vs SPY 12.37% / 0.63 / -34.10%. **FAIL**.
- Invested 26% of days (SMA200 cash filter); 25/37 month-ends risk-on. IS is flat (0%) because ALPACA names and 12-1 lookback do not populate until ~2021+ and OOS is where the 25 on-months sit.
- **Family closed.** Do not iterate lookbacks or top-N.

## Phase 5b IS freeze

- Winner: `dump3_stock` IS Sharpe=-0.09 trades=333 OOS Sharpe=0.64 trades=1827 MDD=-27.95%

- Variants are named (dump vs RSI2, stock SMA20 vs SPY SMA200), not a parameter grid.

- Phase 5c (15m entry timing) only if a daily variant is close to the gate.

## Promotion

**No Phase-5 candidate cleared the risk-adjusted gate.**

Best swing variant `MR_dump3_stock` (3-day dump, exit SMA20 / 8d / 10% stop): CAGR=5.50% Sharpe=0.28 Sortino=0.46 MDD=-27.95%, 2154 trades, WR=51%, PF=1.12. Better MDD than SPY (-28% vs -34%) but Sharpe and CAGR both miss. OOS Sharpe=0.64 vs SPY OOS 1.41.

RSI2 pullbacks lose money (PF < 1). SPY>SMA200 overlay does not rescue either family.

Phase 5c (15m entry timing) is **not** warranted: daily MR is not close to Sharpe > 1.0. Incumbent remains Phase 4 **KEEP** `Blend_SPY70_BV30`.
