# Channel outside-area ratio vs losers (2026-09-12)

Same-list diagnostic on `current_best/1d_channel_touch.html` (last-15m-open-mid + 15m N+1 mid **n=2297 E +0.27 PF 1.09**, geo-year PF 1.095). Occupancy not re-walked. **No promote.**

The yellow region on a 3-touch chart is daily **lows below the support rail**. Channel area is the parallel band on a bar-index axis from **L1 through the buy day**. Ratio = `sum(max(0, support − low)) / (width × n_bars)`. Peak analog: `max(support − low) / width`. Rails from L1/L2 prices on the ALPACA+IB daily panel.

## Coverage

| | |
|--|--|
| Trades scored | **2297 / 2297** (every HTML row) |
| Unique symbols | 940 |
| Wall-clock | load 261.6s TimescaleDB (cache miss) + ratios 1.7s; wall 266.8s |
| Typical ratio | median **0.0002** (0.02% of the parallelogram); p90 0.014; p99 0.101 |
| Never below support | **35.2%** (809 trades) |

A visually large dip is still a small fraction of a long L1→buy parallelogram. TARS 2023-07-20 (the pharma undershoot example, −19.4%) is ratio **0.043**. TDS 2023-12-01 is 0.51 and still only one fat-tail name.

## Correlation with losing trades

Loser = `gain_pct <= 0` (1576 / 2297). **No ranking signal.**

| Feature | Spearman vs gain | Point-biserial vs loser | Mean winners | Mean losers |
|---------|------------------|-------------------------|--------------|-------------|
| **Below-support area ratio** | **+0.013** | **−0.008** | 0.0071 | 0.0066 |
| Peak undershoot / width | +0.014 | −0.002 | 0.135 | 0.134 |
| Bars below support | +0.003 | −0.005 | — | — |
| Existing `max_beyond_width` (overshoot above resist) | −0.012 | −0.022 | 0.641 | 0.603 |

Point-biserial is slightly **negative**: losers have a bit *less* yellow area, not more. Win rate / loser rate is flat across buckets (~66–70% losers; book WR 31.4%).

## Quintiles (area ratio)

Zero bucket separate; five qcut bins on the 1488 trades with ratio > 0.

| Bucket | n | Loser % | WR | E | PF |
|--------|---|---------|----|---|-----|
| 0 (no undershoot) | 809 | 69.0 | 31.0 | +0.07 | 1.02 |
| tiny (0–0.00017) | 298 | 69.8 | 30.2 | +0.18 | 1.06 |
| (0.00017–0.00083) | 297 | 67.7 | 32.3 | +0.57 | 1.19 |
| (0.00083–0.0031) | 298 | 68.5 | 31.5 | +0.61 | 1.19 |
| (0.0031–0.0101) | 297 | 66.0 | 34.0 | +0.63 | 1.22 |
| **top (>0.0101)** | 298 | 70.1 | 29.9 | **−0.12** | **0.96** |

Moderate undershoot (a real support tag after L1) has the best E/PF. Never testing support is meh. The fattest yellow blobs are slightly worse on E, **not** on loser rate. Fat winners live in that top bucket too: WBD 2025-10-30 **+24.8%** at ratio 0.36; PFSI 2020-07-27 **+18.8%** at 0.48. TDS −7.7% at 0.51.

## Skip overlays (occupancy not re-walked)

| Overlay | n | E | PF | geo year PF | 2022-23 E/PF | winners dropped | winner $ / loser $ |
|---------|---|---|----|-------------|--------------|-----------------|--------------------|
| Baseline | 2297 | +0.27 | 1.086 | 1.095 | −0.90 / 0.73 | — | — |
| Keep only ratio = 0 | 809 | +0.07 | 1.022 | 0.900 | −1.83 / 0.49 | 470 / 1018 | 5188 / 4632 |
| Keep <= median | 1149 | +0.01 | 1.003 | 0.961 | −1.69 / 0.53 | 367 / 781 | 4131 / 3530 |
| Drop top positive quintile | 1999 | +0.32 | 1.104 | **1.115** | −0.96 / 0.71 | 89 / 209 | 922 / 957 |
| Peak undershoot <= 0.25 | 1895 | +0.26 | 1.084 | 1.062 | −1.22 / 0.64 | 127 / 275 | 1362 / 1245 |
| Peak undershoot <= 0.50 | 2152 | +0.26 | 1.083 | 1.086 | −0.82 / 0.75 | 42 / 103 | 534 / 476 |

Drop-top is a small pooled bump. Winner dollars almost equal loser dollars saved. 2022-23 stays red. Winner-$ gate: do not occupancy-walk. **No promote.**

Never-below-support as a *keep* rule is worse than baseline (geo 0.90). A yellow blob is not a skip; a never-tested rail is not a keeper.

## Full-column correlations vs gain_pct

Same CSV, every numeric column vs `gain_pct`. Occupancy not re-walked. **No promote.**

Book is fat-tailed (median −3.43%, mean +0.27%, max +104%). Pearson is ~0 on causal columns. Spearman tops out at **|0.113|** (`atr_pct` = `atr_1d_pct`). The next three (`atr_15m_pct` −0.096, `open_above_rail_pct` −0.084, `range_pct` −0.077) are the same vol/extension cluster (pairwise Spearman 0.71–0.91). Yellow-area ratio stays **+0.013**.

ATR Spearman is negative because the k=2 stop is wider on high-vol names (loser ranks sit further left). Quintile E/PF still *favors* mid-to-high ATR: Q1 ATR 0.7–2.0% is n=460 WR 21% E −0.58 PF 0.78 geo 0.58; Q3–Q5 are E +0.67 to +0.81 / PF 1.19–1.25. Do not skip high ATR. Outcome leaks (`gain_pct_net` Spearman +0.77, `hold_days` +0.56) are tautological.

Artifacts: `channel_outside_area_gain_corr.json`, `channel_outside_area_gain_corr_quintiles.json`.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\analyze_channel_outside_area.py --workers 8
```

Artifacts: `reports/ascending_channels/2026-09-12/channel_outside_area_trades.csv`, `channel_outside_area_quintiles.csv`, `channel_outside_area_skip_overlays.csv`, `channel_outside_area_summary.json`.
