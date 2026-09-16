# Alpaca snapshot timeout (15m FAILED Telegram)

**Date:** 2026-09-15

RTH minute `channel_touch_15m_proximity` was Telegram-ing `Channel-touch 15m FAILED` with:

- `HTTPSConnectionPool(host='data.alpaca.markets', port=443): Read timed out. (read timeout=None)`
- `Connection aborted / ConnectionResetError 10054`

`alpaca-py` 0.13 `StockHistoricalDataClient` issues snapshot GETs with **no HTTP timeout**. A hung or reset `data.alpaca.markets` socket took down the whole job (and could stall HotPriceHub the same way). Concurrent hub (~5s) + minute proximity on the same snapshot endpoint made resets more likely.

**Fix:** `fetch_alpaca_last_prices` binds 5s connect / 15s read, retries transient errors per batch, and does not wipe stored last prices on failure. Minute proximity skips Alpaca when `last_price_ts` is already &lt;30s (hub is the primary poll). A flaky Alpaca poll is a warning, not a FAILED Telegram, and does not abort 15m bar-close IB fills.

Playbook: [channel_touch_15m_live](../../../features/channel_touch_15m_live.md).
