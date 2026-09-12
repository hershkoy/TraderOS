# Jargon

Short notes. Add as needed.

**MAE** — Maximum Adverse Excursion: the worst dip against you after the fill, as a % of entry (here: how far price went against the trade *before* the trail’s peak). Not “mean absolute error.”

**RMSE** — Root Mean Squared Error: typical size of prediction mistakes, in the same units as the target. If we predict MAE and RMSE is 2.7, guesses are off by about 2.7 percentage points on average (large errors count more than small ones).

**AAA %b / ATR Anchored Range** — Session overlay on Charts: at each new ATR timeframe (usually 1D), mid = session open or prior close and half-range = ATR/2. Bands stay flat until the next session (TradeSeekers AAA %b overlay). Intraday uses the last completed HTF ATR (causal). Daily midnight-UTC stamps keep that calendar date.

**Session volume-delta** — Buy vs sell volume estimated from candle geometry (`volume * (close-low)/(high-low)`). `15m_sum` splits each 15m bar then sums (default). `session_ohlc` splits once on the day's high/low/close using summed 15m volume (closer to a daily Volume Delta pane). Neither is a POC/VAH/VAL volume profile.

**Morning Doji Star** — Bullish three-candle reversal: large bearish candle, gapped-down doji, strong bullish third candle that closes into the first body. Entry helper only (`utils/research/morning_doji_star.py`); skip-first last-15m occupancy-walk was not run (2022 still red).

**Shakeout-confirm-only** — Skip the first H2 close-above-resist fill; buy only after ≥1 close at/below the rail then a second close above (`--shakeout-confirm-only`). Not the same as `--shakeout-breakout` (buy first and optionally rebuy). Live last-15m N+1 n=1191 E +0.36 PF 1.12; 2022 fails harder. No promote.

**Evening Doji Star** — Bearish three-candle reversal: large bullish candle, gapped-up doji (buyer exhaustion), strong bearish third candle that closes into the first body. On the last-15m book the fill session may be candle 1; exit is the next 15m mid after candle 3's close.

**Failed-breakout exit** — After an H2 fill, the first later session whose close is at/below the projected resist rail; sell the next 15m mid if that is earlier than ATR/trail. Not the delayed 2nd-close *re-entry* (overlay C).

**N+1 mid (sell)** — Decision on completed bar N (15m low tagged the ATR/trail stop); fill is the next RTH 15m midpoint. Overnight to the next 09:30 ET bar is allowed because you already have a position. Not the same as a buy next-mid (last RTH 15:45 has no same-session next bar and cancels).

**Outside-area ratio** — For an ascending channel, the yellow region below support: `sum(max(0, support − low))` on daily bars from L1 through the buy day, divided by channel area `width × n_bars` (bar-index parallelogram). Peak analog is `max(support − low) / width`. Not the same as `max_beyond_width` (overshoot *above* resist).

**TARS-like cancel** — H2 resist-break setup that dies on a *shallow* support close (undershoot / width `<= 0.25`) and the next daily close is back at/above support. Not a fill. A later close above the same resist inside `max_wait` is a reclaim, not shakeout-breakout (shakeout is close back at/below *resist* with support still holding).

**lerp85** — Hot-cross fill heuristic: `rail + 0.85*(close − rail)` if that 15m close is at/above the resist rail, else the rail; then clamp to the bar `[low, high]`. “10% below close” is 15% of the way from close back toward the rail (along that span), not 10% of price. Gap open still lerps then clamps. Default `--hot-cross-fill`. Exact last-print is unknown (no universe 5m/tick tape); do not invent `(high−close)/2`.
