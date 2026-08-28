# Channel-touch entry features + beyond-width sweep (2026-08-28)

## Goal

1. Record a point-in-time entry snapshot (RSI, MA distance, RS, SPY regime, vol, volume, squeeze, geometry, calendar, `max_beyond_width`) so filters can be mined after the fact.
2. Sweep `--max-beyond-width` as a pattern-validity gate: max `(high - resist) / width` from first support touch through the entry bar. Not an exit.

Detector v1 unchanged.

## Setup

```bat
python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --atr-stop-mult 2.0 --require-in-channel --max-channel-span-days 365 --max-entries-per-day 1 --friction-pct 0.25 --workers 4 --load-workers 8 --start 2018-11-01 --end 2026-08-27 --fallback-provider IB --merge-mode prefix
python scripts\research\analyze_channel_touch_entry_features.py --trades reports\ascending_channels\channel_touch_trades_raw_20260828_164937.csv --friction-pct 0.25
```

Windowed 504/252, ALPACA 1d + IB prefix. Wall-clock **119s** (universe list 58s, cache-hit load 16s, scan 43s, RS 3s).

## Keeper report (beyond-width off)

Same stack as the 2026-08-28 IB-windowed ATR run: **n=1001 E +0.97% PF 1.315** median -3.56% WR 29.5%. Fat-tail unchanged (BETR still the monster winner, `max_beyond_width=2.83`).

Raw (pre in-channel / span / RS-top1): **6485** trades.

## Beyond-width A/B (filter then RS top1)

Same in-channel + span365 + ATR k=2 + squeeze + 0.25% friction.

| max_beyond_width | n | E% | PF | median% | vs off |
|------------------|---|-----|-----|---------|--------|
| off | 1001 | +0.97 | 1.315 | -3.56 | baseline |
| 0.0 | 54 | +0.82 | 1.342 | -2.51 | too few |
| **0.25** | **700** | **+1.14** | **1.397** | -3.35 | **lift** |
| 0.5 | 835 | +0.74 | 1.246 | -3.47 | worse |
| 1.0 | 930 | +0.66 | 1.215 | -3.54 | worse |

`0.0` is stricter than the detector touch band (~0.15 widths) and starves the sample.

**0.25 is the only cap that lifts both E and PF with usable n.** It also **drops BETR** (2.83 widths), so the lift is not the fat-tail winner. 0.5 / 1.0 drop BETR too but keep more mediocre pierces and lose the RS-slot shuffle vs 0.25.

SRCE 2026-06-08 (the TV chart): `max_beyond_width=0.189`, gain +13.1% net, still **kept** at 0.25. The ~$90 spike was after entry.

**Soft promote:** `--max-beyond-width 0.25` on top of in-channel + span365. Do not use 0.0. Do not use this as `--resist-exit`.

## Feature mining (raw 6485, univariate only)

Spearman vs net gain is weak for almost everything (`atr_pct` -0.22 is the strongest; `max_beyond_width` -0.03). Do **not** promote from Spearman alone.

Clones (|rho|>=0.7): span vs age; `channel_pos` vs `room_to_resist`; RSI vs %B / dist_sma50 / channel_pos. Keep one from each pair.

Quintile notes (hypothesis only until a filter-then-RS A/B):

- `max_beyond_width` lowest bin (0-0.11): E +1.70 PF 1.61 — agrees with the 0.25 sweep
- `channel_pos` lower bins better on **raw**; this is **not** a reason to revive H3 lower-40% on the RS-top1 keeper (that already lost)
- Lowest ADV quintile looks strong on raw — same lottery/small-name effect already **rejected** as a hard ADV floor
- Thursday / June / November look hot — calendar multiple-testing, do not promote
- RSI high quintile (>75) is the **worst** RSI bucket (E +0.61 PF 1.22) vs RSI<=54 (E +1.56 PF 1.54) — possible follow-up A/B: `--max-rsi 60` after RS, not before mining confirmation
- `spy_above_sma50` still only a small raw lift (E 1.30 vs 0.97); H4 hard gate stays rejected

Next A/Bs worth doing (one at a time, filter then RS): `--max-rsi` ~60; maybe `bb_pctb` upper-cap (clone of RSI). Skip calendar and ADV floors.

## Code

- [`utils/research/channel_touch_entry_features.py`](../../../utils/research/channel_touch_entry_features.py)
- [`scripts/research/backtest_channel_touch_trades.py`](../../../scripts/research/backtest_channel_touch_trades.py): `--entry-features` (default on), `--max-beyond-width`, `--beyond-width-sweep`, raw CSV
- [`scripts/research/analyze_channel_touch_entry_features.py`](../../../scripts/research/analyze_channel_touch_entry_features.py)
- Tests: `tests/unit/test_channel_touch_entry_features.py`, `tests/unit/test_backtest_channel_touch_trades.py`

## Artifacts

- `reports/ascending_channels/channel_touch_trades_raw_20260828_164937.csv`
- `reports/ascending_channels/channel_touch_trades_20260828_164937.csv`
- `reports/ascending_channels/channel_touch_beyond_width_ab_20260828_164937.csv`
- `reports/ascending_channels/channel_touch_entry_features_buckets_20260828_165109.csv`
- `reports/ascending_channels/channel_touch_entry_features_spearman_20260828_165109.csv`
- `reports/ascending_channels/channel_touch_entry_features_corr_20260828_165109.csv`
