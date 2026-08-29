# Channel-touch nightly switched to l3_touch (2026-08-29)

## Change

Nightly EOD detector now matches the daily l3_touch opt keeper in
`channel_touch_tv_report_interactive_rs_top1_default_fric0.25_l3_touch_minwait6_rsi50_beyond025_inchannel_span365_20260829_105855.html`
(trades `channel_touch_trades_20260828_194314.csv`: n=1024 E +1.77% PF 1.62).

Previously live scanned **pivot-confirm** (no quality flags). That path remains
available via `--entry-mode pivot`.

## Live defaults

| Param | Value |
|-------|-------|
| `entry_mode` | `l3_touch` |
| `min_l3_wait_bars` | 6 (abort if tagged earlier; do not retarget) |
| `max_rsi` | 50 |
| `require_in_channel` | on |
| `max_channel_span_days` | 365 |
| `max_beyond_width` | 0.25 |
| RS vs SPY 126d | top 1 / day |
| ATR hard-stop | k=2.0 clamped 1.5%–6% |
| Window | 504 / 252 |
| Fill | first from-above support tag + 0.1% slip |

Detector v1 `find_channels` is unchanged. Live setups use `find_h2_l3_setups`.
Squeeze-adaptive trail and 0.25% friction stay research-exit/cost settings, not
scan gates. Data path is still Alpaca 1d refresh (no IB 15m stream).

## Code

- `utils/scanning/channel_touch.py` — `LIVE_DEFAULTS`, l3 fill + quality then RS
- `scripts/scanners/channel_touch_nightly.py` — CLI defaults + Telegram payload
- Tests: `tests/unit/test_channel_touch.py`, `tests/unit/test_channel_touch_nightly.py`
