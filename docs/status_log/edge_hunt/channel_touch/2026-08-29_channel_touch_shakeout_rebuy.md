# Channel-touch shakeout rebuy — 2026-08-29

## Idea

CSTM 2019-05-23 L3 hard-stopped, then reclaimed the rail ~9 bars later (2019-06-06) and ran. Test a **rebuy** (not hold-through): after a taken L3, if price **closes through support** and **tags from above** within N trading bars, fill a second trade. First position must already be closed (`busy_until`). Detector v1 unchanged. Nightly unchanged (flag default 0).

## Setup

Parents = live 1d keeper CSV `channel_touch_trades_20260828_194314.csv` (quality + RS top1). Reconstruct rails, `_shakeout_rebuy_fill`, same ATR k=2 / squeeze trail / 0.25% friction. Rebuys pass in-channel / span365 / beyond 0.25 / RSI<=50, then re-RS top1 on the combined set.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_shakeout.py --bars 10 --sweep 5,10,15
```

Wall-clock **64s** (parquet cache hit, 798 names).

## Results vs keeper n=1024 E **+1.77** PF **1.62**

| window | shakeout-only n | E | PF | 2020-21 | combined+RS n | E | PF |
|--------|-----------------|---|----|---------|---------------|---|-----|
| 5 bars | 33 | **-0.80** | **0.75** | E -3.77 PF 0.07 | 1041 | +1.73 | 1.60 |
| **10 bars** | **73** | **+0.89** | **1.29** | **E -3.19 PF 0.17** | **1064** | **+1.66** | **1.57** |
| 15 bars | 94 | +1.08 | 1.35 | E -3.28 PF 0.16 | 1072 | +1.60 | 1.55 |

CSTM does fire: 2019-06-06 @ 9.1475, trail 2019-08-02, **+27.3%** (9 bars after L3, RSI 45.7). One working example; 2020-21 shakeouts are the hole.

## Gate

Promote required shakeout-only PF+E >= keeper on the full window and no wrecked year bucket. Combined+RS must not dilute the keeper.

**No promote.** 10-bar rebuys are weaker than the L3 they follow, and they drag 2020-21. Wider windows add more of the same. Nightly stays stop-and-done (no re-arm after close-through-support).

## Code

- [`scripts/research/backtest_channel_touch_trades.py`](../../../scripts/research/backtest_channel_touch_trades.py): `--shakeout-rebuy-bars` (default 0); `_shakeout_rebuy_fill`
- [`scripts/research/backtest_channel_touch_shakeout.py`](../../../scripts/research/backtest_channel_touch_shakeout.py)
- Tests: `tests/unit/test_l3_touch_entry.py`

## Artifacts

- `reports/ascending_channels/channel_touch_shakeout_rebuy_10bars_20260829_202837.csv`
