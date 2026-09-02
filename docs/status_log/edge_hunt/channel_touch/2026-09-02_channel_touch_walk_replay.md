# Walk-replay vs batch channel-touch (one symbol)

Causal detector: bars are revealed `0..t` only. Compare to original `trades_for_symbol` on the full series (later pivots can change rails / H2).

## How to run

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\replay_channel_touch_walk.py --symbol AAPL
python scripts\research\replay_channel_touch_walk.py --symbol AAPL --preset 15m
python scripts\research\replay_channel_touch_walk.py --symbol AAPL --freeze-h2
```

`--freeze-h2` keeps the first L1-L2-H2 geometry (live watchlist). Default re-scans each prefix (Pine-like).

## AAPL 1d (live H2 resist-break span365, 2018-11-01 -> 2026-08-27)

ALPACA_IB prefix, window 504/252, wait-6, unique occupancy, ATR k=2. Wall-clock: load 0.1s, batch 0.1s, walk 4.1s (1962 bars).

| side | n | E% | PF | WR% | notes |
|------|---|----|----|-----|-------|
| batch (original) | 2 | -3.64 | 0.00 | 0.0 | both hard/trail losers |
| walk (causal) | 3 | -2.22 | 0.087 | 33.3 | same 2 fills + one extra |

Matched **2/2** batch fills at the same price and H2:

| buy | price | H2 | gain% |
|-----|-------|----|-------|
| 2021-09-03 | 153.17 | 2021-07-15 | -3.23 |
| 2026-07-15 | 321.51 | 2026-06-08 | -3.55 |

Walk-only (not occupancy — Sep-3 trade was already stopped 2021-09-14):

| buy | price | H2 | gain% |
|-----|-------|----|-------|
| 2021-11-30 | 163.15 | 2021-09-07 | +0.88 |

Batch never armed that 2021-09-07 H2. Full-series `find_h2_l3_setups` sees later confirmed highs/lows, so width/H2 selection for the 2020-11-02 / 2021-09-07 pair is not the same as on the Nov-30 prefix. Detector v1 is unchanged; this is a scan-honesty gap, not a rewrite.

## Code

- `utils/research/channel_touch_walk_replay.py`
- `scripts/research/replay_channel_touch_walk.py`
- `tests/unit/test_channel_touch_walk_replay.py`
