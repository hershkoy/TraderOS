# Channel-touch edge-v2 long-history sweep (2026-08-24)

## Goal

Re-baseline on ALPACA `1d` from **2018-11-01 → 2026-08-23**, then test five improvement hypotheses vs RS-top1 + squeeze-adaptive + 0.25% friction.

## Command

```bat
python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --edge-v2 --workers 4 --load-workers 8 --start 2018-11-01 --end 2026-08-23
```

Bat: `scripts/research/run_channel_touch_edge_v2.bat`

## Wall-clock

- Symbol list: ~12s
- OHLCV load (2217 symbols, mostly cold for new start): **~282s**
- 10 simulation scans × ~22s: **~220s**
- Total: **~512s (~8.5 min)**

## Long-history baseline (H0)

Same rules as 2026-08-23 refreshed scan (RS top1/day, squeeze 10%/18%, friction 0.25%).

| Metric | Value |
|--------|-------|
| Trades | 496 |
| Expectancy | **+1.63%** |
| Profit factor | **1.72** |
| Win rate | 26.0% |
| Hard stops | 326 / 496 (66%) |

### Year splits (H0, buy_date)

| Bucket | n | E | PF |
|--------|---|---|-----|
| 2018-2019 | 0 | — | — |
| 2020-2021 | 0 | — | — |
| 2022-2023 | 26 | **-0.13%** | 0.94 |
| 2024-2026 | 470 | **+1.73%** | 1.76 |
| FULL | 496 | +1.63% | 1.72 |

**Note:** Extending `--start` to 2018-11-01 did **not** expand the RS-top1 sample into 2018–2021. Under same-day RS top1, trade density is almost entirely 2024–2026 (panel fill + one entry/day). 2022–2023 is small and slightly negative. Edge is concentrated in the recent regime.

## Hypothesis results (net E / PF after 0.25% friction + RS top1)

| Scenario | n | E | PF | vs H0 | Verdict |
|----------|---|---|-----|-------|---------|
| H0_baseline | 496 | +1.63 | 1.72 | — | baseline |
| H1_atr_stop_k1.0 | 499 | +1.68 | 1.79 | slight lift | weak keep |
| H1_atr_stop_k1.5 | 498 | +1.89 | 1.74 | lift | **keep** |
| H1_atr_stop_k2.0 | 496 | **+2.29** | **1.82** | best ATR | **keep (best stop)** |
| H2_pivot_len_5 | 248 | +1.85 | 1.82 | lift, half sample | keep / monitor n |
| H2_pivot_len_10 | 387 | +1.05 | 1.47 | worse | reject |
| H2_entry_reclaim | 492 | +6.41 | 6.18 | outlier | **reject (look-ahead)** |
| H5_struct_exits | 498 | **-1.01** | 0.65 | kills edge | **reject** |
| H3_geometry | 293 | +1.74 | 1.83 | modest lift | soft keep |
| H4_spy_sma50 | 387 | +1.64 | 1.74 | flat | no lift |
| H3+H4 | 226 | +0.94 | 1.44 | worse | reject |
| H1k1.5+H4 | 389 | +1.87 | 1.75 | similar to H1 alone | optional |

### H2 reclaim look-ahead (important)

`entry_mode=reclaim` searches for close > support starting at `touch_index+1`, but the touch pivot is only confirmed at `touch_index+pivot_len`. That uses future knowledge of a pivot that is not yet confirmed live. **Do not promote reclaim** until retested with reclaim search starting only after pivot confirmation.

### H5 structure exits

Resistance tag + squeeze-fade tighten + 50d time stop cut avg win (15% → 6%) and turned expectancy negative. Reject as a bundle; do not combine with ATR stop.

## Recommended next cut

1. **Primary:** ATR hard stop `k=2.0` (clamped 1.5%–6%) on top of current RS-top1 + squeeze-adaptive trail.
2. **Secondary test:** `pivot_len=5` (valid shorter confirmation) — better E/PF but n≈248; check stability.
3. **Soft:** geometry filter alone (lower 40% of channel) — small lift, fewer trades.
4. **Skip:** hard ADV/ATR floors (prior), H5 struct exits, H4 alone, H3+H4, reclaim-until-fixed, pivot_len=10.

## Artifacts

- `reports/ascending_channels/channel_touch_edge_v2_20260824_015252.csv`
- `reports/ascending_channels/channel_touch_edge_v2_years_20260824_015252.csv`
- `reports/ascending_channels/channel_touch_trades_20260824_015252.csv`
- `reports/ascending_channels/channel_touch_trades_summary_20260824_015252.txt`

## Code changes

- [`scripts/research/backtest_channel_touch_trades.py`](scripts/research/backtest_channel_touch_trades.py): ATR stop, reclaim/pivot entry modes, geometry + SPY regime filters, structure exits, year splits, `--edge-v2`, default `--start 2018-11-01`
- [`scripts/research/run_channel_touch_edge_v2.bat`](scripts/research/run_channel_touch_edge_v2.bat)
