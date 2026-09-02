# Walk-replay vs batch channel-touch (one symbol)

Causal detector: bars are revealed `0..t` only. Compare to original `trades_for_symbol` on the full series.

## How to run

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\replay_channel_touch_walk.py --symbol AAPL
python scripts\research\replay_channel_touch_walk.py --symbol AAPL --preset 15m
python scripts\research\replay_channel_touch_walk.py --symbol AAPL --freeze-h2
```

`--freeze-h2` keeps the first-seen L1-L2-H2 geometry (live watchlist). Default re-scans each prefix (Pine-like).

## Fast batch = replay (live H2 resist-break)

Do not re-run the detector on every prefix. Same O(windows x pairs) as the old batch:

1. `causal_h2` (default): freeze rails at the first completing H2; L1 from the last `max_low_pivots` lows *before L2*.
2. `h2_resist_break_only` **before occupancy**: drop L3 support-tag fills so they cannot occupy the book, then get stripped from the report. Walk never queued those tags. Post-filter after occupancy was a silent miss.

`--no-causal-h2` restores the old look-ahead batch. Detector v1 `find_channels` is unchanged.

## AAPL 1d (live H2 resist-break span365, 2018-11-01 -> 2026-08-27)

ALPACA_IB prefix, window 504/252, wait-6, unique occupancy, ATR k=2.

After occupancy fix (2026-09-02 evening): **exact match**, batch 0.1s, walk 2.1s.

| side | n | E% | PF | WR% | notes |
|------|---|----|----|-----|-------|
| batch | 3 | -2.22 | 0.087 | 33.3 | same three fills as walk |
| walk | 3 | -2.22 | 0.087 | 33.3 | |

| buy | price | H2 | gain% |
|-----|-------|----|-------|
| 2021-09-03 | 153.17 | 2021-07-15 | -3.23 |
| 2021-11-30 | 163.15 | 2021-09-07 | +0.88 |
| 2026-07-15 | 321.51 | 2026-06-08 | -3.55 |

The 2021-11-30 fill was always in the batch setup list (same rails). An earlier L3 support tag (~2021-09) occupied `busy_until`, then `h2_resist_break_only` stripped that L3 trade from the output.

Earlier leaky batch (later highs refit H2; L3 occupancy) was n=2, missing Nov-30.

## Code

- `utils/research/channel_touch_walk_replay.py`
- `scripts/research/replay_channel_touch_walk.py`
- `scripts/research/find_ascending_channels.py` (`causal_h2`)
- `scripts/research/backtest_channel_touch_trades.py` (`h2_resist_break_only` before occupancy)
- `tests/unit/test_channel_touch_walk_replay.py`
- `tests/unit/test_backtest_channel_touch_trades.py::test_h2_resist_break_only_skips_l3_occupancy`
