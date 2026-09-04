# Realistic fill default: signal-bar close

`--realistic-fill` now buys at the **close of the rail-tag bar** (`signal-close`). The old next-bar mid purchaser is kept as `--realistic-fill-mode next-mid`.

## Why (GOOGL 2025-11-18)

CTF JSON fill `2025-11-18T15:15:00Z` `enp=281.625` was **not** a rail touch. The L3 tag was the prior 15m:

| Clock | Bar start | Low | Close | Mid | Role |
|-------|-----------|-----|-------|-----|------|
| 10:00 ET | 15:00 UTC | **279.46** | **280.79** | 280.43 | Detector L3 tag (touch_price 279.46 vs support ~279) |
| 10:15 ET | 15:15 UTC | 280.23 | 281.61 | **281.625** | Old next-mid BUY (mid-channel, never tagged) |
| 11:00 ET | 16:00 UTC | **278.20** | 282.09 | 280.27 | Visible wick through support ~278.98 |
| 11:15 ET | 16:15 UTC | 281.73 | 286.27 | **284.00** | Next bar after the 11:00 wick |

Wait-1 fires on the **first** near-rail tag (15:00), so it never waits for the 11:00 wick. Signal-close puts BUY on the 15:00 candle at **280.79**. Same setup in the new raw book: buy_time 15:00, +3.02% trail (not in RS-top1).

## Wait-1 A/B (300 IB 15m, RS top1, friction 0.10)

| Book | n | WR% | E% | PF |
|------|---|-----|-----|-----|
| Next-mid + no-buy-below | 1771 | 30.4 | **+0.04** | **1.06** |
| **Signal-close (new default)** | **1777** | 27.4 | **−0.12** | **0.83** |

Year E signal-close: 2018-19 −0.23, 2020-21 −0.17, 2022-23 +0.01, 2024-26 −0.15. `channel_pos<0` still 0.

**Do not promote.** Wait-1 was already worse than wait-12; signal-close makes it net-negative. Frozen `current_best/15m_channel_touch.html` is still **next-mid wait-12**. Restore that fill with `--realistic-fill-mode next-mid`. Wait-12 has not been rescanned at signal-close.

Trades: `reports/ascending_channels/2026-09-04/channel_touch_15m_trades_20260904_160021.csv`
HTML: `reports/ascending_channels/2026-09-04/channel_touch_tv_report_interactive_rs_top1_default_fric0.10_15m_l3_wait1_signal_close_20260904_160316.html`
