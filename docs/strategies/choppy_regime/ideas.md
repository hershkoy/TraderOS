Implemented 2026-09-13 as causal research (not a promote): `utils/research/choppy_regime.py`, runner `scripts/research/run_choppy_regime.py`, scan `scripts/scanners/choppy_regime_scan.py`. First book: [choppy regime](../../status_log/edge_hunt/2026-09-13_choppy_regime.md). Refine (SMA50, no VIX sit-out / size-scale, 0.10% friction, pullback time-stop 6, IB 15m developing close): [choppy regime refine](../../status_log/edge_hunt/2026-09-13_choppy_regime_refine.md). Confirm overlays (RSI cross-back, pullback close-near-high): [choppy regime confirm](../../status_log/edge_hunt/2026-09-13_choppy_regime_confirm.md).

Based on the execution styles that fit your schedule, here are two specific, mathematically testable rule sets to generate your buy signals.

Strategy 1: The RSI-2 System (Best for Market-On-Close Execution)
This strategy is designed to find severe, multi-day panic selling in stocks that are otherwise structurally healthy.

The Trend Filter: At the end of the day, run a scan for stocks that are currently trading above their 200-day Simple Moving Average (SMA).  

The Buy Signal: Within that list, look for stocks where the 2-period RSI (Relative Strength Index) has plunged below 5.  

Safety Filters: To adapt this to a choppy market, you can skip taking signals on days when the broader market VIX spikes above 30, or right before major news events like CPI releases, as these often cause unpredictable overnight gaps.  

The Execution: Because you have a 15-minute window at the end of the day, you can run this scan at 3:45 PM. If a stock meets the criteria, submit a Market-On-Close order.  

The Exit: You hold the position until the stock's price closes above its 5-day SMA, or until the 2-period RSI crosses back above 60 to 70. If the stock fails to bounce at all, you exit automatically after 5 to 10 days (a time stop) to prevent your capital from being trapped.  

Strategy 2: The 3-Day Pullback (Best for Set-and-Forget Limit Orders)
This strategy looks for a specific sequence of consecutive drops, allowing you to calculate your buy price the night before so you don't have to watch the market during the day at all.

The Trend Filter: The stock must close the day above its 100-day moving average.  

The Buy Setup: The stock must have just registered exactly three consecutive lower lows, and its closing price must be below its 5-day moving average.  

The Execution: You do not buy right away. Instead, that evening, calculate a limit buy order for the following day. A standard formula is to take the current price and subtract 0.5 times the 10-day Average True Range (ATR). You enter this limit order with your broker to run automatically the next day.  

The Exit: Once the order is triggered and you own the stock, you wait for a day where the stock closes higher than the previous day's close. When that happens, you instruct your broker to sell the position on the next morning's open.