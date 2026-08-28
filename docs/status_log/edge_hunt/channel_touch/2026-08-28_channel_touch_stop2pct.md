# Channel-touch 2% hard stop (full universe, 2026-08-28)

Same windowed IB-fallback stack as `channel_touch_trades_20260828_000726.csv`, with ATR k=2.0 replaced by a **fixed 2% hard stop**.

## Setup

```bat
python scripts\research\backtest_channel_touch_trades.py --all-symbols --workers 4 --load-workers 8 --squeeze-adaptive --stop-pct 0.02 --require-in-channel --max-channel-span-days 365 --max-entries-per-day 1 --friction-pct 0.25 --start 2018-11-01 --end 2026-08-27 --fallback-provider IB --merge-mode prefix
```

- Windowed detector default 504/252; no `--atr-stop-mult`
- ALPACA_IB parquet cache hit (2217/2217)
- Wall-clock: **56.2s**

## Result vs ATR k=2.0 (same universe / filters)

| | ATR k=2 clamp 1.5–6% | **Fixed 2% stop** |
|---|---|---|
| n | 1001 | **1014** |
| E (net 0.25%) | **+0.97%** | +0.84% |
| PF | 1.32 | **1.45** |
| Win rate | **29.5%** | 16.7% |
| Median | −3.56% | **−2.25%** |
| Avg win / avg loss | +13.7 / −4.36 | +16.1 / **−2.21** |
| Hold days | 26.8 | **16.3** |
| Hard-stop exits | 543 (54%) | **814 (80%)** |
| Pre-2020 buys | 106 | 106 |

Year counts (2%): 2019=106, 2020=83, 2021=152, 2022=92, 2023=164, 2024=160, 2025=157, 2026=100.

Slightly more trades because a tighter stop frees the per-symbol slot sooner. Same 106 pre-2020 entries — stop only changes exits.

## Read

Tighter stop **lifts PF** (losses capped near 2%) and **cuts hold time**, but **lowers expectancy** and collapses win rate: most names never get room to run. Still fat-tailed (BETR +358% trail-wide). **Do not replace ATR k=2 keepers** on this evidence — PF gain is not an E/MDD promote.

## Artifacts

- CSV: `reports/ascending_channels/channel_touch_trades_20260828_023127.csv`
- Summary: `reports/ascending_channels/channel_touch_trades_summary_20260828_023127.txt`
- HTML: `reports/ascending_channels/channel_touch_tv_report_interactive_rs_top1_default_fric0.25_ib_fallback_stop2pct_inchannel_span365_20260828_023147.html`
