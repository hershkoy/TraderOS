# Channel-touch nightly cron (2026-08-25)

## Goal

Productionize channel-touch monitoring: nightly Windows Task Scheduler job that
refreshes ALPACA `1d` bars, scans for **l3_touch** fills on the latest bar
(min-wait 6, RSI<=50, in-channel, span<=365, beyond-width 0.25, RS top1,
ATR hard-stop k=2.0), and Telegram-notifies.

## Components

| Piece | Path |
|-------|------|
| Telegram pinger | `utils/notify/telegram_pinger.py` |
| Live trigger scan | `utils/scanning/channel_touch.py` |
| Nightly orchestrator | `scripts/scanners/channel_touch_nightly.py` |
| Cron bat | `crons/channel_touch_nightly.bat` |
| Task install | `crons/install_channel_touch_nightly_task.bat` |

## Env

Add to `.env` (also documented in `.env.example`):

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Test:

```bat
venv\Scripts\activate && set PYTHONPATH=. && python -m utils.notify.telegram_pinger --ping
```

## Manual run

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\scanners\channel_touch_nightly.py --skip-update --dry-run
```

Full (update + scan + Telegram):

```bat
crons\channel_touch_nightly.bat
```

## Windows Task Scheduler

Default schedule in installer: **Mon-Fri 23:00 local**.

```bat
crons\install_channel_touch_nightly_task.bat
```

Useful:

```bat
schtasks /Query /TN "backTraderTest\ChannelTouchNightly" /V /FO LIST
schtasks /Run /TN "backTraderTest\ChannelTouchNightly"
```

## Signal rules (live)

Updated 2026-08-29 to the daily l3_touch keeper (was pivot-confirm):

- `entry_mode=l3_touch`: arm at H2, fill first from-above support tag + 0.1% slip on as-of bar
- Abort if support is tagged before 6 bars after H2; do not retarget a later dip
- Quality then RS: in-channel, span<=365d, max beyond-width 0.25, RSI 14 <= 50, then same-day RS vs SPY (126d) top 1
- ATR hard stop mult = 2.0 clamped to 1.5%-6% of fill price
- Windowed 504/252 so v1 `max_low_pivots=16` still sees older H2 setups
- Does **not** use `entry_mode=reclaim` (look-ahead)
- `--entry-mode pivot` remains available to revert the old confirm-bar path

## Notes

- Update uses last 14 calendar days via `--since` over the ALPACA 1d symbol list
  written to `reports/ascending_channels/alpaca_1d_symbols.txt`
- Daily refresh uses Alpaca **multi-symbol** batches (`--multi-symbol`, batch-size 100,
  near-zero delays) instead of one HTTP call per ticker
- Artifacts: `reports/ascending_channels/channel_touch_nightly_*.csv`
- Logs: `logs/scanners/channel_touch_nightly_*.log`
