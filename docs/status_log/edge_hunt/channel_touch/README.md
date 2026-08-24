# Channel-touch — status log

Ascending-channel / channel-touch research and nightly productionization.

| Date | Doc |
|------|-----|
| 2026-08-22 | [Edge improve (RS top1)](2026-08-22_channel_touch_edge_improve.md) |
| 2026-08-23 | [TTM squeeze adaptive trail](2026-08-23_channel_touch_squeeze_trail.md) |
| 2026-08-23 | [Refreshed-universe scan + TV watchlist](2026-08-23_channel_touch_refreshed_universe_scan.md) |
| 2026-08-24 | [Edge-v2 long-history sweep](2026-08-24_channel_touch_edge_v2_long_history.md) |
| 2026-08-25 | [Nightly cron + Telegram](2026-08-25_channel_touch_nightly_cron.md) |

Related: [current status](../../current_status.md), [edge hunt](../README.md), [TV trendline alerts](../../../features/tv_channel_trendline_alert.md)

## Keepers (live / research)

- Classical bottom touch ≥3, pivot confirmation
- Same-day RS vs SPY top1 (126d)
- Squeeze-adaptive trail 10%/18%
- ATR hard stop k=2.0 clamped 1.5%–6%
- Do **not** use `entry_mode=reclaim` until look-ahead fixed

## Harness

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --edge-v2 --workers 4 --load-workers 8 --start 2018-11-01
python scripts\scanners\channel_touch_nightly.py --skip-update --dry-run
crons\channel_touch_nightly.bat
```

Outputs: `reports/ascending_channels/`, `logs/scanners/channel_touch_nightly_*.log`
