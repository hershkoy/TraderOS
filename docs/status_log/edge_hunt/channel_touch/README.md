# Channel-touch — status log

Ascending-channel / channel-touch research and nightly productionization.

| Date | Doc |
|------|-----|
| 2026-08-22 | [Edge improve (RS top1)](2026-08-22_channel_touch_edge_improve.md) |
| 2026-08-23 | [TTM squeeze adaptive trail](2026-08-23_channel_touch_squeeze_trail.md) |
| 2026-08-23 | [Refreshed-universe scan + TV watchlist](2026-08-23_channel_touch_refreshed_universe_scan.md) |
| 2026-08-24 | [Edge-v2 long-history sweep](2026-08-24_channel_touch_edge_v2_long_history.md) |
| 2026-08-25 | [Nightly cron + Telegram](2026-08-25_channel_touch_nightly_cron.md) |
| 2026-08-25 | [Geometry quality audit + in-channel filters](2026-08-25_channel_touch_quality_audit.md) |
| 2026-08-25 | [Robustness: outliers, bootstrap, capacity](2026-08-25_channel_touch_robustness.md) |
| 2026-08-26 | [15m scaled hunt](2026-08-26_channel_touch_15m.md) |
| 2026-08-28 | [Fixed 2% stop vs ATR k=2](2026-08-28_channel_touch_stop2pct.md) |
| 2026-08-28 | [Entry features + beyond-width sweep](2026-08-28_channel_touch_entry_features.md) |
| 2026-08-28 | [L3 rail-touch entry](2026-08-28_channel_touch_l3_touch_entry.md) |
| 2026-08-28 | [L3 opt loops](2026-08-28_channel_touch_l3_opt_loops.md) |
| 2026-08-29 | [15m l3_touch / L4 / entry-model loops](2026-08-29_channel_touch_15m_opt_loops.md) |

Related: [current status](../../current_status.md), [edge hunt](../README.md), [TV trendline alerts](../../../features/tv_channel_trendline_alert.md)

## Keepers (live / research)

- Classical bottom touch ≥3, pivot confirmation
- Same-day RS vs SPY top1 (126d)
- Squeeze-adaptive trail 10%/18%
- ATR hard stop k=2.0 clamped 1.5%–6%
- **Soft promote:** `--require-in-channel` + `--max-channel-span-days 365` (drops above-resist / multi-year channels; lifts E/PF)
- **Soft promote:** `--max-beyond-width 0.25` (pattern already pierced; lifts E/PF on IB-windowed ATR stack, n=700; `0.0` too strict, `0.5`/`1.0` worse)
- **15m research (not live):** `--preset 15m --entry-mode l3_touch --min-l3-wait-bars 12` on 300 IB names; do **not** copy daily beyond-0.25 onto 15m L3
- Do **not** use `entry_mode=reclaim` until look-ahead fixed
- Do **not** require lower-40% geometry (`--geometry-filter` / H3) — hurts edge

## Harness

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --atr-stop-mult 2.0 --require-in-channel --max-channel-span-days 365 --max-beyond-width 0.25 --max-entries-per-day 1 --friction-pct 0.25 --workers 4 --load-workers 8 --start 2018-11-01
python scripts\research\analyze_channel_touch_entry_features.py --trades reports\ascending_channels\channel_touch_trades_raw_<stamp>.csv
python scripts\research\audit_channel_touch_quality.py
python scripts\research\channel_touch_robustness.py --trades reports\ascending_channels\channel_touch_trades_20260825_014435.csv
python scripts\research\generate_channel_touch_tv_report.py --trades reports\ascending_channels\channel_touch_trades_20260825_014435.csv --friction-pct 0.25 --rs-top1 --tag atr_k2_inchannel_span365_robust
python scripts\scanners\channel_touch_nightly.py --skip-update --dry-run
crons\channel_touch_nightly.bat
python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --workers 4 --load-workers 8
```

Outputs: `reports/ascending_channels/`, `logs/scanners/channel_touch_nightly_*.log`
