# Channel-touch shakeout then second resist-break — 2026-09-05

## Why SXI stopped, and why the later rally was skipped

SXI 2020-08-11 @ 60.30 is a daily **H2 resist-break** (L1 2020-03-19 / H2 2020-07-16). 2020-08-25 @ 56.68 is the **6% ATR ceiling** (`gain_pct_net -6.25` = 6% stop + 0.25% friction). That first loss is not preventable with the current keeper stop.

The finder **stopped after the first close above resistance**, so the later re-break never emitted. Occupancy would have allowed a new trade after Aug 25.

This is **not** L3 `--shakeout-rebuy-bars` (close through support then reclaim; that rule **cancels** on a close above resistance and failed 2020-21).

## Rule

After a first H2 resist-break fill on the same frozen rails:

1. Require N closes back **inside** the channel (not still above resistance).
2. Support must not close through.
3. Fill the next close above resistance at the rail + slip.
4. Rebuy, not hold-through (`busy_until`).
5. At most one extra fill per H2 setup.
6. Window: remaining `max_l3_wait_bars=252` from H2.

No in-channel / RSI 50 (breakouts skip those). Span<=365, causal H2, unique-symbol/day, ATR k=2 clamp 1.5%–6%, 0.25% friction.

## Setup

Full ALPACA 1d universe (2216 + SPY), IB prefix, 2018-11-01 → 2026-08-27. One load, two scans (`min_inside` 1 and 5). Extras emitted with `parent_exit_reason`; hard-stop vs any-closed is a post-filter. Wall-clock **123s** (parquet cache hit).

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_shakeout_breakout.py --all-symbols --workers 4 --load-workers 8
```

## Results (0.25% friction)

Frozen causal unique-symbol H2 span365 (2026-09-03): **n=3364 E +2.40 PF 2.02**. Parent rows below are from the same scan as the extras (busy_until can skip a later parent if an extra is still open), so n is a bit lower.

### min_inside=1

| book | n | E | PF | 2020-21 | notes |
|------|---|---|----|---------|-------|
| Parent unique H2 span365 | 3119 | +2.37 | 1.996 | E +1.91 PF 1.79 | same-scan parent |
| **Sleeve any-closed** | **1439** | **+3.16** | **2.316** | **E +3.24 PF 2.39** | all year buckets + |
| Sleeve hard-stop | 511 | +2.31 | 1.964 | E +1.33 PF 1.51 | below parent; 2022-23 E +0.72 PF 1.24 |
| **Combined any-closed** | **4558** | **+2.62** | **2.098** | E +2.33 PF 1.97 | lifts parent and frozen 3364 / +2.40 / 2.02 |
| Combined hard-stop | 3630 | +2.36 | 1.992 | E +1.84 PF 1.75 | slight dilute vs parent |

Median still negative (fat tail). Sleeve any-closed med −1.54 vs parent −2.22.

### min_inside=5

| book | n | E | PF | 2020-21 |
|------|---|---|----|---------|
| Parent unique H2 | 3048 | +2.42 | 2.026 | E +1.91 PF 1.79 |
| Sleeve any-closed | 1786 | +2.70 | 2.093 | E +2.46 PF 2.02 |
| Sleeve hard-stop | 620 | +2.09 | 1.857 | E +1.77 PF 1.76; **2022-23 E +0.46 PF 1.15** |
| Combined any-closed | 4834 | +2.52 | 2.051 | E +2.11 PF 1.87 |
| Combined hard-stop | 3668 | +2.37 | 1.996 | E +1.89 PF 1.78 |

min_inside=5 any-closed still lifts, but the sleeve is weaker than min_inside=1. Hard-stop-only does not beat the parent.

## SXI sanity (loaded daily bars)

Parent 2020-08-11 @ 60.30, hard-stop 2020-08-25 @ 56.68, −6.25% net.

Extra (min_inside=1 and 5): **2020-10-05 @ 62.6564**, trail 2021-03-22 @ 97.353, **+55.13% net**, 36 inside bars, `parent_exit_reason=hard_stop`. Same L1/H2 rails as the Aug fill.

## Gate

Sleeve any-closed **min_inside=1** beats the frozen unique H2 book on E and PF, combined unique-symbol **lifts** rather than dilutes, and **2020-21 is healthy** (the hole that killed L3 support-reclaim rebuy).

**Soft-promote as a research flag** (`--shakeout-breakout`, default off). Prefer **any-closed + min_inside=1** over hard-stop-only and over min_inside=5.

**Nightly scanner not rewired.** Re-arming a filled H2 after exit is a live-path change (not just a backtest flag). 15m unchanged. Do not reuse `--shakeout-rebuy-bars`.

## Code

- [`scripts/research/backtest_channel_touch_trades.py`](../../../scripts/research/backtest_channel_touch_trades.py): `_shakeout_breakout_fill`, `--shakeout-breakout`
- [`scripts/research/backtest_channel_touch_shakeout_breakout.py`](../../../scripts/research/backtest_channel_touch_shakeout_breakout.py)
- Tests: `tests/unit/test_l3_touch_entry.py`

## Artifacts

- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_any_min1_20260905_114330.csv` (n=1439)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_hs_min1_20260905_114330.csv` (n=511)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_any_min5_20260905_114330.csv` (n=1786)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_hs_min5_20260905_114330.csv` (n=620)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_summary_20260905_114330.csv`
