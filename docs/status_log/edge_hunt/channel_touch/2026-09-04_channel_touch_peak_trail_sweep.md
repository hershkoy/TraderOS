# Peak-trail exit sweep (5 variants vs Keeper)

Date: 2026-09-04

## Setup

`--preset 15m --n-symbols 300 --entry-mode l3_touch --min-l3-wait-bars 12 --realistic-fill --realistic-fill-mode signal-close --touch-error-pct 0 --peak-trail-sweep`

One OHLCV load; six exit passes. Friction 0.10; in-channel; span<=10d; RS top1/day.

| ID | Rule |
|----|------|
| Keeper | ATR k=2 + scaled ~2% squeeze trail (preset) |
| F125 | Fixed peak trail 1.25% |
| F200 | Fixed peak trail 2.0% |
| F300 | Fixed peak trail 3.0% |
| T4d | Start 4%; −0.02pp/bar; floor 1% |
| G4t | Start 4%; −0.33pp per full +1% from entry; floor 1% |

## Results

| scenario | n | E | PF | med | WR% | eod | drop10 E | drop10 PF |
|----------|---|---|----|-----|-----|-----|----------|-----------|
| **Keeper** | 1744 | **+0.13** | **1.220** | −0.66 | 30.1 | 1 | +0.04 | 1.067 |
| F125 | 1743 | −0.04 | 0.916 | −0.44 | 35.7 | 2 | −0.11 | 0.796 |
| F200 | 1742 | +0.13 | 1.165 | −0.58 | 37.8 | 2 | +0.02 | 1.031 |
| F300 | 1728 | +0.11 | 1.096 | −0.93 | 38.3 | 3 | +0.01 | 1.004 |
| T4d | 1734 | +0.15 | 1.124 | −0.51 | 43.0 | 4 | +0.04 | 1.034 |
| **G4t** | 1729 | **+0.22** | 1.162 | −0.72 | 45.1 | 4 | **+0.11** | **1.085** |

Year splits: Keeper and G4t all buckets +; F125 fails most years; F300 soft in 2024-26.

Wall-clock: ~28 min (6×~250–280s scans, cache hit).

Artifacts:

- `reports/ascending_channels/2026-09-04/channel_touch_peak_trail_sweep_20260904_195111.csv`
- `..._years_...csv` / `..._summary_...txt`

## Implementation

- `_peak_trail_width` + `peak_trail_mode` in `backtest_channel_touch_trades._simulate_trade`
- CLI: `--peak-trail-mode`, `--trail-floor`, `--trail-decay-per-bar`, `--trail-tighten-per-pct`, `--peak-trail-sweep`
- `--resist-arm-trail` aliases `fixed`

## Verdict

**Do not promote** any peak-trail variant over Keeper.

- Gate needs **E and PF** both >= Keeper without fat-tail collapse.
- **G4t** is best experimental: highest E (+0.22) and best drop-top-10, all years +, but **PF 1.16 < Keeper 1.22**.
- T4d edges E slightly (+0.15) but PF weaker.
- Fixed 1.25% dies; 2% ties E and loses PF; 3% worse.

Keep ATR + scaled squeeze trail as the 15m research/live exit reference. G4t remains optional research only.
