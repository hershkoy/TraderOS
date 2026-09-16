# SMB LOD / 9 EMA x VWAP reclaim (2026-09-13)

Causal 15m translation of the SMB Capital short [The Trade I Wait For Every Day](https://www.youtube.com/shorts/S1UnwVDgKoY). **No promote.** Do not occupancy-walk onto last-15m. Do not treat the advertised 3:1 as a live payoff.

Runner: `scripts/research/backtest_vwap_ema_reclaim.py`. Logic: `utils/research/vwap_ema_reclaim.py`.

## Clock (realistic purchasing)

Session VWAP and running LOD on RTH 15m. 9 EMA is **continuous** across sessions (chart EMA, not a session reset). Signal: first completed 15m where 9 EMA crosses up through VWAP after the session low is in (LOD on a prior bar, still below VWAP, signal bar does not make a new low). Last entry 15:30 ET (15:45 has no same-session next bar).

- **Buy (primary):** next same-session 15m **mid** after the cross bar (`next-mid`). Wild-bar 0.5% gate. Not an intra-bar fill at the line cross (video "do not wait for the candle to close" needs tick/5m; close-based 9 EMA is only known when the 15m completes).
- **Buy (analog):** `signal-close` of the cross bar — earliest causal fill for a close-based EMA.
- **Sell:** measured-move from LOD to VWAP at the cross. Target = cross + 1 unit. Stop = cross − unit/3. Decision on bar N (high/low tag; same-bar both → stop). Fill the **next same-session** 15m mid. Overnight is not allowed (day trade). Last-bar tag or no tag flattens that close (MOC). Not a same-bar stop/target clip.

Friction **0.10%** round-trip (short hold). Unique-symbol occupancy (one name cannot stack). Uncapped concurrent.

## Sample

Top 300 IB 15m names by ALPACA 1d ADV. 2018-11-01 -> 2026-08-21. Wall **250s** (list 121s, 15m 10s, index 26s, fills 92s).

## Results

| Book | n | E | PF | geo | WR | median | max conc |
|------|---|---|----|-----|----|--------|----------|
| next-mid gross | 227915 | +0.005 | **1.02** | 1.02 | 52.4 | +0.04 | **170** (med 38) |
| next-mid **fric 0.10** | 227915 | **−0.10** | **0.67** | 0.66 | 44.9 | −0.06 | 170 |
| signal-close gross | 245796 | +0.009 | 1.03 | 1.03 | 52.7 | +0.05 | 177 (med 42) |
| signal-close **fric 0.10** | 245796 | **−0.09** | **0.72** | 0.71 | 46.3 | −0.05 | 177 |

Year buckets (next-mid **fric 0.10**): every regime PF **0.62–0.68**. Calendar **2022** n=28217 E −0.10 PF 0.69. Drop-top-3 does not move E. Median unit **0.90%** of price (stop theoretical 0.30%). Median hold **75m**.

Exits (next-mid): EOD **85873** / target 82573 / stop 59469. Target mean **+0.44%** (not +0.90%). Stop mean **−0.59%** (not −0.30%). Realized R: target **+0.69**, stop **−1.19**, EOD ~0. The 3:1 geometry does not survive next-mid slippage.

## Verdict

**No promote.** Gross PF 1.02 is noise around scratch. 0.1% friction is larger than the edge: E ≈ −friction. This is not a rare "wait for" setup — it is almost every session's first 9/VWAP cross on liquid names (n≈228k on 300). Avg win ≈ avg loss (~0.46 / −0.49 gross). Max concurrent 170 is not live equal-dollar. 15m is coarser than a day-trader 1m/5m chart; a faster clock would make units even smaller versus the same 0.1% friction. Do not filter-fish this book. Do not grade against clip 1d H2.

Selectivity follow-up ([filters](2026-09-13_vwap_ema_reclaim_filters.md)): daily top1 is the ~100x cut (n=1959 E −0.11 PF 0.85); unit>=5 is 263x pooled scratch that fails geo/2022/drop-top-3. **Still no promote.**

```bat
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\backtest_vwap_ema_reclaim.py --n-symbols 300
```
