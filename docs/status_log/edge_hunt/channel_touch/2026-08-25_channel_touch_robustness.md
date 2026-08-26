# Channel-touch robustness (2026-08-25)

n=374, friction=0.25%

## Verdict

- PASS soft: after drop top-1, E=1.7018 PF=1.6107
- PASS soft: after drop top-3, E still 1.3202
- PASS soft: winsor 20% still E=0.5188
- OK: top-3 tail dependency 0.2476
- OK: bootstrap P(mean<0)=0.3%

## Scenarios

| scenario | n | E% | median% | trimmed5% | PF | win% |
|---|---|---|---|---|---|---|
| baseline | 374 | 2.6553 | -3.045 | -0.0618 | 1.9555 | 34.22 |
| drop_top_1 | 373 | 1.7018 | -3.05 | -0.1464 | 1.6107 | 34.05 |
| drop_top_3 | 371 | 1.3202 | -3.08 | -0.3074 | 1.4712 | 33.69 |
| drop_top_5 | 369 | 0.9876 | -3.1 | -0.4605 | 1.3506 | 33.33 |
| drop_top_5pct_winners | 355 | -0.4559 | -3.27 | -1.2296 | 0.8443 | 30.7 |
| winsor_cap_50 | 374 | 1.6372 | -3.045 | -0.0618 | 1.5891 | 34.22 |
| winsor_cap_30 | 374 | 1.0827 | -3.045 | -0.0618 | 1.3896 | 34.22 |
| winsor_cap_20 | 374 | 0.5188 | -3.045 | -0.1582 | 1.1867 | 34.22 |
| winsor_3x_avg_win | 374 | 1.5944 | -3.045 | -0.0618 | 1.5737 | 34.22 |
| exclude_BETR | 373 | 1.7018 | -3.05 | -0.1464 | 1.6107 | 34.05 |

## Tail dependency

- Top-3 / gross profit = **0.2476**
- Top trades: [{'stock': 'BETR', 'buy_date': '2025-08-21', 'gain_pct_net': 358.33}, {'stock': 'SANM', 'buy_date': '2026-04-07', 'gain_pct_net': 73.64}, {'stock': 'CALM', 'buy_date': '2024-05-22', 'gain_pct_net': 71.34}, {'stock': 'PRM', 'buy_date': '2025-04-08', 'gain_pct_net': 69.32}, {'stock': 'APA', 'buy_date': '2026-01-29', 'gain_pct_net': 56.03}]

## Bootstrap

```
{
  "n_iter": 2000,
  "n_trades": 374,
  "pct_negative_sum": 0.3,
  "pct_negative_mean": 0.3,
  "pct_pf_below_1": 0.3,
  "sum_p05": 373.209,
  "sum_p50": 936.185,
  "sum_p95": 1747.828,
  "mean_p05": 0.9979,
  "mean_p50": 2.5032,
  "mean_p95": 4.6733
}
```

## Exposure / capacity

- Concurrent open: {'max': 29, 'p95': 21, 'median': 11, 'mean': 10.74}

| capacity | n | E% | PF |
|---|---|---|---|
| 5 | 108 | 4.0289 | 2.5873 |
| 10 | 210 | 3.0735 | 2.1633 |
| 15 | 301 | 2.104 | 1.773 |
| 20 | 350 | 2.875 | 2.0477 |

## Half-splits: {'n_splits': 20, 'both_halves_positive_E': 20, 'pct_both_positive': 100.0}

## Year splits

| bucket | n | E% | PF |
|---|---|---|---|
| year_2023 | 21 | -0.2224 | 0.9145 |
| year_2024 | 125 | 2.0582 | 1.7687 |
| year_2025 | 132 | 3.1684 | 2.0419 |
| year_2026 | 96 | 3.3569 | 2.2959 |
| odd_years | 153 | 2.703 | 1.9069 |
| even_years | 221 | 2.6223 | 1.9934 |

Artifacts: `D:/WORK/Projs/IB/backTraderTest/reports/ascending_channels/channel_touch_robustness_20260825_213405.json`, `D:/WORK/Projs/IB/backTraderTest/reports/ascending_channels/channel_touch_robustness_scenarios_20260825_213405.csv`
