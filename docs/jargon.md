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

**VIX backwardation skip** — Prior-session VIX/VIX3M >= 1 (optionally VIX>=20) as a last-15m keep/skip. Not the Phase 6c SPY overlay. On n=2297 it makes 2022 worse (leftover E −2.69); 2023 skips none. No promote.

**Causal HMM skip** — Expanding 2/3-state Gaussian HMM on daily SPY log-return / 60d vol / 20d mean + VIX change + VIX/VIX3M (optional TLT/GLD/UUP). Fit through t only, Wasserstein-match states so bull = highest SPY-drift mean, join P(bull) at prior session. Last-15m P(bull)<0.5/0.65 fails like SPY SMA (2022 leftover harder; 3-state zeros 2022 and wrecks 2020-21). No promote.

**Universe SMA20 breadth** — Daily fraction of the ALPACA ~2,217-name panel with close > own SMA20 (names without SMA20 omitted). Last-15m uses prior session. B50 (<0.50) and below-expanding-median skips fail the 2022/winner-$ gate. Not SPY.

**RTH exhaustion (last-15m)** — (15:45 ET open − 09:30 ET open) / ATR. 15m ATR14 uses completed bars before 15:45; daily ATR is prior session. Skip if already >=2 15m-ATR or >=1 daily ATR. X_d1 lifts pooled E slightly and makes 2022 leftover worse. No promote.



