# /hot Bought tab + SELL NOW (2026-09-05)

Manual fill tracking on the dashboard. Click **Bought** on a candidate to persist an open position; the **Bought** tab shows last vs the live keeper stop; last through the stop fires **SELL NOW** (browser + Telegram).

## Stop (same keeper as nightly occupancy)

- ATR hard-stop k=2.0 clamped 1.5%–6% (3% fallback if ATR cannot be loaded)
- 10% trail from peak; `current_stop = max(hard_stop, peak * 0.90)`
- Live peak/stop use **Alpaca last** (not completed-bar high/low). Actionable while the bar is still forming. Not a bar-low backtest fill.

Entry price is `fill_px`, else last, else last close. One open/sell_now row per symbol+timeframe. **Sold** closes the row (no extra Telegram).

## Where it runs

- Always-on `hot_price_server.py` (~5s Alpaca) even if `/hot` is closed
- `channel_touch_15m --mode proximity` every RTH minute (backup if the price hub is down)
- Dedup via `sell_notified_at`

Schema: `init-scripts/17-channel-touch-bought-trades.sql`. Helpers: `utils/scanning/channel_touch_bought.py`.
