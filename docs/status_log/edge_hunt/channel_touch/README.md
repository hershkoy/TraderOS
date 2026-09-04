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
| 2026-08-29 | [Nightly switched to l3_touch keeper](2026-08-29_channel_touch_nightly_l3.md) | |
| 2026-08-29 | [Same-bar close leak: prior-bar features + daily 15m hybrid](2026-08-29_channel_touch_samebar_leak.md) |
| 2026-08-29 | [Ridge P&L confidence sizing](2026-08-29_channel_touch_confidence_size.md) |
| 2026-08-29 | [Shakeout rebuy](2026-08-29_channel_touch_shakeout_rebuy.md) |
| 2026-08-29 | [H2 resistance-break](2026-08-29_channel_touch_h2_resist_break.md) |
| 2026-08-29 | [Nightly switched to H2 resist-break](2026-08-29_channel_touch_nightly_h2_break.md) |
| 2026-08-29 | [15m full-universe H2 resist-break](2026-08-29_channel_touch_15m_full_h2_break.md) |
| 2026-08-29 | [Drop RS top1 cap: unique-symbol/day](2026-08-29_channel_touch_unique_symbol_day.md) |
| 2026-08-29 | [15m unique-symbol filter hypotheses](2026-08-29_channel_touch_15m_unique_filters.md) |
| 2026-08-30 | [15m H5 stack: WR/PF lift](2026-08-30_channel_touch_15m_h5_refine.md) |
| 2026-08-30 | [15m live monitor + IB 15m backfill](2026-08-30_channel_touch_15m_live_monitor.md) |
| 2026-08-30 | [15m hot dashboard (no TV alerts)](2026-08-30_channel_touch_15m_hot_dashboard.md) |
| 2026-09-02 | [Walk-replay vs batch (causal_h2)](2026-09-02_channel_touch_walk_replay.md) |
| 2026-09-03 | [Causal full-universe 1d + 15m; current_best replaced](2026-09-03_channel_touch_causal_full_universe.md) |
| 2026-09-04 | [IB 5m universe backfill (yield to RTH)](2026-09-04_ib_5m_backfill.md) |
| 2026-09-04 | [Realistic-fill full-universe 1d + 15m; current_best replaced](2026-09-04_channel_touch_realistic_fill.md) |
| 2026-09-04 | [15m L3 first-tag wait-1 vs wait-12 (realistic fill)](2026-09-04_channel_touch_15m_l3_wait1.md) |
| 2026-09-04 | [No buy below channel (re-entry or resist-break)](2026-09-04_channel_touch_no_buy_below.md) |
| 2026-09-04 | [Realistic fill default: signal-bar close](2026-09-04_channel_touch_signal_close.md) |
| 2026-09-04 | [15m L3 real support touch (0%) vs 0.24% near-miss](2026-09-04_channel_touch_touch_error_0.md) |

Related: [current status](../../current_status.md), [edge hunt](../README.md), [TV trendline alerts](../../../features/tv_channel_trendline_alert.md)

## Keepers (live / research)

Stable HTML links: `reports/ascending_channels/current_best/` (`1d_channel_touch.html` live H2, `1d_l3_touch.html` retired L3, `15m_channel_touch.html` / `15m_h2_resist_break.html` 300-name research, `15m_full_h2_resist_break.html` full IB 15m, `15m_h5_overshoot_vol.html` H5 + overshoot p80 + vol>=2). Write-up: `reports/ascending_channels/current_best/README.md`.

- **Live / nightly:** `--h2-resist-break --h2-resist-break-only --min-l3-wait-bars 6` + span365; **no** RSI / in-channel / beyond-width; **unique-symbol/day** (not RS top1); ATR k=2.0 clamped 1.5%–6%; windowed 504/252. Optional `--max-entries-per-day 1` restores RS top1.
- **Retired L3 keeper:** `--entry-mode l3_touch --min-l3-wait-bars 6 --max-rsi 50` + in-channel + span365 + beyond 0.25 (`1d_l3_touch.html`)
- Squeeze-adaptive trail 10%/18% (research exits; not a scan gate)
- **Live / nightly sample (causal optimistic 2026-09-03):** unique-symbol H2 span365 **n=3364 E +2.40 PF 2.02** (optional RS top1 n=1126 E +2.94 PF 2.24). Leaky Aug-29 was n=2253 E +2.80 PF 2.21.
- **`current_best` (realistic fill 2026-09-04):** same live recipe + `--realistic-fill` unique-symbol H2 span365 **n=713 E +1.78 PF 1.72** (RS top1 n=419 E +2.65 PF 2.08). Nightly scanner is still close fills until rewired. See [realistic fill](2026-09-04_channel_touch_realistic_fill.md).
- **15m research (not nightly):** Unique-symbol H5 + overshoot train p80 + vol>=2 was **n=8212 WR 47.4% E +0.92 PF 3.57** on optimistic fills; **realistic next-bar mid is n=7209 E ~0 PF 1.01** — do not promote. **`current_best/15m_channel_touch.html`** is now wait-12 **signal-close + `--touch-error-pct 0`** (**n=1744 E +0.13 PF 1.22**; RTH clocks; IB SPY from 2018-11-06). Touch 0.24 signal-close was n=1755 E +0.05 PF 1.08; prior next-mid freeze n=1746 E +0.09 PF 1.16. Prefer `--touch-error-pct 0` for 15m L3; do not set `--error-pct 0`. Textbook first-tag wait-1 signal-close **n=1777 E −0.12 PF 0.83** — do not promote. See [touch 0](2026-09-04_channel_touch_touch_error_0.md), [wait-1](2026-09-04_channel_touch_15m_l3_wait1.md), [signal-close](2026-09-04_channel_touch_signal_close.md). Live **research** loop (armed watchlist in TimescaleDB + Alpaca minute proximity + bar-close H5 + `/hot` dashboard): `scripts/scanners/channel_touch_15m.py` after IB 15m backfill (`scripts/data/backfill_ib_15m_universe.py`). Playbook: [15m live](../../../features/channel_touch_15m_live.md). Do **not** copy daily beyond-0.25 or span 365 onto 15m. Nightly stays **daily** EOD.
- Do **not** use `entry_mode=reclaim` until look-ahead fixed
- Do **not** require lower-40% geometry (`--geometry-filter` / H3) — hurts edge
- Do **not** size from ridge P&L confidence (RS-top1 OOS: 1d E worse; 15m lift dies after RS). Equal-dollar stays the size model.

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
python scripts\data\backfill_ib_15m_universe.py --inventory
python scripts\data\backfill_ib_5m_universe.py --inventory
crons\after_rth_ib_backfill.bat
crons\stop_ib_5m_backfill.bat
python scripts\scanners\channel_touch_15m.py --mode run --dry-run --max-symbols 50
python scripts\scanners\channel_touch_15m.py --mode proximity --dry-run
crons\backfill_ib_15m_universe.bat
crons\channel_touch_15m.bat
crons\channel_touch_15m_proximity.bat
python charting_server.py
python scripts\research\backtest_channel_touch_h2_break.py --preset 15m
python scripts\research\backtest_channel_touch_h2_break.py --preset 15m --all-symbols --workers 4 --load-workers 8
python scripts\research\backtest_channel_touch_confidence_size.py --stack 1d
python scripts\research\backtest_channel_touch_confidence_size.py --stack 15m
```

Outputs: `reports/ascending_channels/`, `logs/scanners/channel_touch_nightly_*.log`
