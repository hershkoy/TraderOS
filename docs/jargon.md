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

**H2+L3 occupancy blend** — Timestamp FCFS on last-15m H2 + 15m L3 wait-12. Unified max_open 8/15/20 vs dedicated L3 reserves (15+5 / 12+3 / 8+4). Unique-symbol. Cap 8 starves H2 (late 15:45 fill); cap 15 keeps 76% of L3. Dedicated 15+5 keeps L3. 2022 red on every walk. No promote.

**TLT global gate** — Skip last-15m H2 when prior-session TLT is below SMA100/200 or 63d ROC < 0 (yield-shock proxy). Tags 2022 (113/117) but is off 1502/2297 fills, not a calendar sit-out; leftover 2023 E −1.69; 2020-21 winner-$ blunt. No promote.

**VIX backwardation skip** — Prior-session VIX/VIX3M >= 1 (optionally VIX>=20) as a last-15m keep/skip. Not the Phase 6c SPY overlay. On n=2297 it makes 2022 worse (leftover E −2.69); 2023 skips none. No promote.

**Causal HMM skip** — Expanding 2/3-state Gaussian HMM on daily SPY log-return / 60d vol / 20d mean + VIX change + VIX/VIX3M (optional TLT/GLD/UUP). Fit through t only, Wasserstein-match states so bull = highest SPY-drift mean, join P(bull) at prior session. Last-15m P(bull)<0.5/0.65 fails like SPY SMA (2022 leftover harder; 3-state zeros 2022 and wrecks 2020-21). No promote.

**Universe SMA20 breadth** — Daily fraction of the ALPACA ~2,217-name panel with close > own SMA20 (names without SMA20 omitted). Last-15m uses prior session. B50 (<0.50) and below-expanding-median skips fail the 2022/winner-$ gate. Not SPY.

**RTH exhaustion (last-15m)** — (15:45 ET open − 09:30 ET open) / ATR. 15m ATR14 uses completed bars before 15:45; daily ATR is prior session. Skip if already >=2 15m-ATR or >=1 daily ATR. X_d1 lifts pooled E slightly and makes 2022 leftover worse. No promote.

**Deepvue take-4** — Causal weekend scan (`utils/research/weekly_deepvue.py`): QQQ Friday close above a rising 21 EMA, RS 21d/63d >= 85, RMV tightness near 10/21 EMA, sector cap 2, keep 4 names. Buy next 09:30 ET 15m mid. Not Deepvue the product. AND with last-15m H2 is n=1 (MRNA); no promote.

**MOO** — Market On Open: buy (or sell) at the session's **opening print**. Here that is the next RTH **09:30 ET 15m open** (IB bar start), known when that bar begins — not the 09:30 bar's mid or close, and not a broker auction-order type. Opposite of waiting for a later 15m confirm. Chase-all MOO on last-15m n=2297 E −0.01; no promote.

**Open no-chase (last-15m)** — After a last-15m H2, buy next 09:30 ET open only if that open did not gap up vs the signal session's last 15m close and still holds the rail. Gap-up (or open already below the rail): same-session wait for a pullback then a green/hammer/engulfing 15m whose low and close hold at/above the rail; fill that close. Tick gap-up, not 15m `gap_15m`. Chase-all MOO n=2297 E −0.01; pullback n=1236 E +0.16 PF 1.05. No promote.

**RSI-2 MOC** — Connors-style daily. Research fill is the same-session close (15:45 IB 15m developing close is the live scan proxy; not Alpaca 17:00 EOD). First book: close > SMA200, RSI(2)<5, skip prior VIX>30; unique n=5451 E +0.46 PF 1.36; occ-15 fric0.25 PF 0.96; 2022 n=10. Refine (2026-09-13): SMA50, no VIX sit-out (size = clip(20/VIX, 0.25, 1)), fric 0.10; SMA50 occ-15 n=2082 E +0.37 PF 1.30; 2022 n=194 E +1.08. Extreme RSI<2 / no-MA occ-15 2022 PF 0.95 — do not drop SMA50. RSI cross-back (arm <5, buy reclaim 5 or 10) cuts max concurrent 185→110 but occ-15 fric0.10 PF 1.21 vs threshold 1.30 — no promote. Exit close > SMA5 or RSI(2)>=70, else 8-session time stop. Not Phase 5 `rsi2_stock`.

**3-day pullback (limit)** — Evening setup: close > SMA100, exactly three consecutive lower lows, close < SMA5. Next-session GTC limit = close − 0.5 × ATR(10); gap-through fills at the open. First up-close vs the prior close sells the next morning open. Refine default time stop 6 sessions. Unique n=4564 E +0.27 PF 1.24; 2023 n=1193 E −0.40 PF 0.70. First book (no time stop) n=4536 E +0.26 PF 1.23; 2022-23 PF 0.77. Close-loc 0.7 (setup close in top 30% of the bar) n=1104 E +0.01 PF 1.01; 2023 PF 0.49. No promote.



