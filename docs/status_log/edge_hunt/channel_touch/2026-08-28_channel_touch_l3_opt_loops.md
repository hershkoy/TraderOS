# Channel-touch l3_touch optimization loops (2026-08-28 evening)

Five loops on the bounce-from-above `l3_touch` stack (in-channel, span<=365, beyond-width 0.25, RS top1, ATR k=2, squeeze 10/18, 0.25% friction, IB prefix, 2018-11-01 to 2026-08-27). Gate: PF + expectancy, year splits, n>~300. Trade fills were spot-checked against OHLCV.

Baseline bounce-only keeper: **n=1164 E +1.32% PF 1.44** (median -3.44%). Pivot same filters: n=700 E +1.14 PF 1.40.

## Loop 1 — Stale wait / age (post-filter A/B on raw 18215)

Hypothesis: long waits after H2 (WTFC-style) are junk; old L1-to-buy age is junk.

| Filter | n | E% | PF | Verdict |
|--------|---|-----|-----|---------|
| max wait 15-60d | 693-1116 | -0.46 to +1.07 | <1.36 | **Reject** (hurts) |
| max wait 180d | 1156 | +1.35 | 1.45 | noise vs base |
| max age 90-300d | 171-1047 | +0.10 to +0.89 | <1.30 | **Reject** |

Mean wait after H2 is 18d (median 11). Capping it removes good slower L3s.

## Loop 2 — RSI / %B (filter then RS)

Hypothesis from entry-feature quintiles: high RSI at the rail is a failed bounce.

| Filter | n | E% | PF | Years |
|--------|---|-----|-----|-------|
| **max RSI 50** | **1053** | **+1.61** | **1.55** | all buckets >=+0.33; 2022-23 weakest |
| max RSI 60 | 1133 | +1.42 | 1.48 | mild |
| max bb%B 0.2 | 931 | +1.61 | 1.54 | **Reject**: 2020-21 E +0.17 PF 1.05 |

`--max-rsi 50` is a clone of "not overbought at support". Soft-promote. Do not use %B 0.2.

## Loop 3 — Min wait after H2 (real swing L3)

Hypothesis: a 1-5 bar dip after H2 is not L3.

Post-filter min calendar wait 8d: n=1069 E +1.74 PF 1.60. Year split: 2018-19 slightly worse than base, **2022-23 lifts** (E +1.20 vs +0.51).

Sim v1 (skip then retarget later dip): diluted to E +1.43. Wrong. **Abort the setup if support is tagged before min wait** (do not buy the next dip as fake L3).

`--min-l3-wait-bars 6` (~8 calendar days) + abort-on-early-tag.

Also cancel if close > resistance after H2 (breakout, not pullback).

## Loop 4 — Combined full backtest

```bat
python scripts\research\backtest_channel_touch_trades.py --all-symbols --entry-mode l3_touch --min-l3-wait-bars 6 --max-rsi 50 --require-in-channel --max-channel-span-days 365 --max-beyond-width 0.25 --max-entries-per-day 1 --squeeze-adaptive --atr-stop-mult 2.0 --friction-pct 0.25 --fallback-provider IB --merge-mode prefix --start 2018-11-01 --end 2026-08-27 --workers 4 --load-workers 8
```

**n=1024 E +1.77% PF 1.62** median -3.21% WR 36.8% hold 23.3d. Raw 12572.

Year net E: 2018-19 +2.33, 2020-21 +1.19, 2022-23 +0.94, 2024-26 +2.47. All positive.

OHLCV verify 8/8 random keeper fills: rail_ok, wait_bars>=6, rsi<=50, channel_pos median **0.014**. WTFC 2019-07-16 absent. 1 trade with pos>0.2.

## Loop 5 — Squeeze rising overlay

n=203 E +3.30 PF 2.29 on bounce-raw, years all positive, drop-top-3 still PF 1.92. **n=203 is below the ~300 sample comfort line.** Park as optional overlay, not a hard gate. `squeeze_mom>0` (not rising) **rejects** (E -0.47).

## Rejected this session

- Max wait after H2, tighter max age
- bb%B 0.2 hard cap (regime fail 2020-21)
- squeeze_mom>0
- Min-wait retarget (buy a later dip after an early tag)

## Soft promote (research l3_touch)

`--entry-mode l3_touch --min-l3-wait-bars 6 --max-rsi 50` on top of in-channel + span365 + beyond 0.25. Nightly scanner stays **pivot** until explicitly switched.

## Artifacts

- `reports/ascending_channels/channel_touch_trades_20260828_194314.csv`
- `reports/ascending_channels/channel_touch_l3_opt_filter_ab_20260828.csv`
- `reports/ascending_channels/channel_touch_tv_report_interactive_rs_top1_default_fric0.25_l3_touch_minwait6_rsi50_beyond025_inchannel_span365_20260828_194359.html`
