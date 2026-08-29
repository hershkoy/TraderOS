# Channel-touch 15m optimization loops (2026-08-29)

Five loops on IB 15m, 300 ADV-ranked names, 2018-11-01 to 2025-12-02. Detector v1 unchanged. Gate: PF + expectancy, year splits, n usable, OHLCV fill verify. Features at the entry bar only; model fits use buy_date strictly before the test cutoff.

Prior pivot 15m n=300 (2026-08-26): **n=1764 E +0.15% PF 1.24** median -0.70%. Not a promote.

## Loop 1 — 15m `l3_touch` L3 (entry_touch=3)

Hypothesis: bounce-from-above L3 plus daily keepers (min-wait 6, beyond 0.25) beats 15m pivot.

```bat
python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --entry-mode l3_touch --entry-touch 3 --min-l3-wait-bars 6 --max-beyond-width 0.25 --workers 4 --load-workers 8
```

Wall-clock **425s** (pick ~3 min, cache-hit load 10s, scan 215s, raw 73701).

| stack | n | E% | PF | median% | 2020-21 |
|-------|---|-----|-----|---------|---------|
| pivot 15m (prior) | 1764 | +0.15 | 1.24 | -0.70 | — |
| **L3 + beyond 0.25** | **1696** | **+0.17** | **1.30** | -0.63 | **E -0.05 PF 0.93** |
| L3 beyond off | 1766 | +0.29 | 1.50 | -0.61 | E +0.30 |

All 1696 keeper fills have `touch_num=3`, median `channel_pos` 0.04, `wait_bars` min 6 / med 13.5. Replay-verify 8/8 (fill in bar, rail tag, `trades_for_symbol` reproduces buy_time).

**Do not copy daily `--max-beyond-width 0.25` onto 15m L3.** Off is better and fixes 2020-21.

## Loop 2 — 4th touch (skip L3, fill L4)

Hypothesis: 15m channels continue, so L4 after a real leave-rail would have more (or better) signals.

`--entry-touch 4`: first from-above tag is L3 (confirm, no fill); require close to leave the rail (~0.2 width); min_wait again; then fill L4. After L3, a high above H2 is H3 (allowed). Close through support or close above resist still cancels. Early L3 aborts the setup (no retarget).

Wall-clock **342s**. Raw 22911. Keeper **n=1312 E +0.20% PF 1.36** (beyond 0.25). All `touch_num=4`, wait min 12 / med 26. Replay-verify 8/8.

Fewer signals than L3, not more. 2022-23 E -0.02. L4 does not beat L3 + beyond-off on this window.

## Loop 3 — entry-feature filter sweep (no rescan)

Filter then RS-top1 on Loop 1 raw. Friction 0.10, in-channel, span<=10d.

| Filter | n | E% | PF | years |
|--------|---|-----|-----|-------|
| beyond off | 1766 | +0.29 | 1.50 | all + |
| min_wait 12 | 1574 (w/ 0.25) / 1740 (off) | +0.23 / +0.33 | 1.40 / 1.59 | all + when beyond off |
| max RSI 50 | mild | — | — | 2020-21 only barely + with 0.25 |
| min_close_loc 0.6 | 1717 (off) | +0.48 | 2.03 | all + **but same-bar leak** |
| squeeze_rising | 1430 (off) | +0.43 | 1.84 | all +; optional overlay |

**Reject `min_close_loc` for wick fills.** The tag-bar close is not known when the low hits support. Repricing those trades at the bar close (honesty) flips L3 to **E -0.17 PF 0.81** (all years negative).

## Loop 4 — time-split logistic + tiny MLP

Train on buy_date < cutoff only. Threshold chosen on the last 20% of train (by time). Walk-forward folds 2022 / 2023 / 2024-25.

With all entry features (including close_loc): holdout 2023+ raw kept E/PF lifts, and test+RS looks strong. That lift is largely the same-bar close leak.

Without same-bar close features (`rsi` / `%B` / `close_loc` / `range_pct` / SMA distances dropped):

| fold | test E | test kept E | test PF | kept PF |
|------|--------|-------------|---------|---------|
| 2022 | +0.12 | +0.20 | 1.19 | 1.33 |
| 2023 | +0.21 | +0.20 | 1.38 | 1.37 |
| 2024-25 | +0.22 | +0.25 | 1.40 | 1.48 |

**Do not promote the neural net / logistic as a hard gate.** Walk-forward lift is small and vanishes in 2023.

## Loop 5 — honest combined rescan

Hypothesis: L3 + min_wait 12 (abort early tags) and **no** daily beyond-width is the 15m research stack.

```bat
python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --entry-mode l3_touch --entry-touch 3 --min-l3-wait-bars 12 --workers 4 --load-workers 8
```

Wall-clock **355s**. Raw 44346. Keeper **n=1751 E +0.32% PF 1.58** median -0.58% WR 36.6% hold 23.8 bars. Beyond A/B on this scan: off still the best usable cap (0.25 worse).

Year net E: 2018-19 +0.08, 2020-21 +0.27, 2022-23 +0.25, 2024-26 +0.57. All positive. 2018-19 is the weak bucket.

OHLCV verify 8 sample fills: 8/8 in-range + rail_ok; 7/8 replay_hit (RIG 2023-02-27 tagged support; isolated replay missed the windowed setup — not a mid-channel fill). `wait_bars` min 12 / med 17. `touch_num` all 3. pos>0.35: 0.

## Soft promote (15m research only)

`--preset 15m --entry-mode l3_touch --min-l3-wait-bars 12` on the 300-name IB panel. **Do not** add `--max-beyond-width 0.25` (daily keeper, 15m drag). **Do not** switch nightly. **Do not** stream IB 15m. **Do not** promote L4, close_loc wick filter, or the entry MLP/logistic.

Fat-tail unchanged (median still negative). Next optional: drop-top-N/bootstrap on `channel_touch_15m_trades_20260829_015527.csv` before expanding past 300 names.

## Code

- `scripts/research/backtest_channel_touch_trades.py`: `_h2_rail_tag_fills`, `--entry-touch` 4 on `l3_touch`, `--min-close-loc`
- `utils/research/channel_touch_entry_model.py`: time-split logistic + tiny MLP
- `scripts/research/sweep_15m_touch_filters.py`, `sweep_15m_combos.py`, `apply_channel_touch_entry_model.py`, `verify_15m_touch_sample.py`, `train_channel_touch_entry_model.py`

## Artifacts

- Loop 1: `channel_touch_15m_trades_20260829_013533.csv` / raw `_013533`
- Loop 2: `channel_touch_15m_trades_20260829_014306.csv` / raw `_014306`
- Loop 5: `channel_touch_15m_trades_20260829_015527.csv` / raw `_015527`
- Combos: `channel_touch_15m_loop5_combos.csv`
