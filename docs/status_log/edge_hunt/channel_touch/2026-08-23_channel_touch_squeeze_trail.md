# TTM Squeeze adaptive trail (2026-08-23)

## Hypothesis
Fixed 10% trail stopped BETR (Touch 3, 2025-08-21) at +75% on 2025-09-18 right as TTM Squeeze momentum went vertical. Widen trail when LazyBear mom is strong.

## Rule
- Base trail: 10%
- Wide trail: 18% when TTM Squeeze momentum is:
  - `mom > 0`
  - `mom >= mom[1]` (rising / non-fading)
  - `mom >= 75th percentile` of last 100 bars
- Hard stop unchanged at 3%

## BETR case
| Mode | Exit | Gain | Peak | Reason |
|------|------|------|------|--------|
| Fixed 10% | 2025-09-18 @ 29.39 | +75.4% | 32.66 | trail_stop |
| Squeeze-adaptive | 2025-09-22 @ 76.86 | **+358.6%** | 93.73 | trail_stop_wide |

## Full universe (ALPACA 1d, warm cache)
| Metric | Fixed 10% trail | Squeeze-adaptive (10%/18%) |
|--------|-----------------|----------------------------|
| Trades | 2259 | 2259 |
| Expectancy | +1.48% | **+1.72%** |
| Profit factor | 1.70 | **1.82** |
| Avg win | 13.42% | **14.39%** |
| Win rate | 26.7% | 26.6% |
| Wide-trail exits | 0 | 7 |

Wide trail is rare but high-impact (BETR case). Overall edge lifts modestly without blowing up loss side (avg loss still ~-2.86%).

Flags:
```bat
python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --trail-pct-wide 0.18 --workers 4
```

Trades: `reports/ascending_channels/channel_touch_trades_20260823_212235.csv`
