# Jargon

Short notes. Add as needed.

**MAE** — Maximum Adverse Excursion: the worst dip against you after the fill, as a % of entry (here: how far price went against the trade *before* the trail’s peak). Not “mean absolute error.”

**RMSE** — Root Mean Squared Error: typical size of prediction mistakes, in the same units as the target. If we predict MAE and RMSE is 2.7, guesses are off by about 2.7 percentage points on average (large errors count more than small ones).

**AAA %b / ATR Anchored Range** — Session overlay on Charts: at each new ATR timeframe (usually 1D), mid = session open or prior close and half-range = ATR/2. Bands stay flat until the next session (TradeSeekers AAA %b overlay). Intraday uses the last completed HTF ATR (causal). Daily midnight-UTC stamps keep that calendar date.
