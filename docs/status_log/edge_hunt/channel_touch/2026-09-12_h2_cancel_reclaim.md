# H2 support-cancel then later breakout (TARS Jan-2024) — 2026-09-12

Last-15m book symbols (940), windowed H2 504/252, span365, error 1.2%, min_wait 6, max_wait 252. Occupancy not re-walked. **No promote.**

TARS Nov-15 2023 H2 cancelled Dec-1 (close 15.85 vs support 16.22, **0.11×width**), next session recovered, then close-above-resist Dec-27. Last-15m that day opened above the rail @ **20.40**. Keeper ATR k=2 still **hard-stopped 2024-01-05 @ 19.28 (−5.5%)** before the melt-up to 39. Ignoring the poke does not harvest that rally under current_best exits.

## Pre-registered

| Term | Rule |
|------|------|
| Cancel | First close `< support * (1 − 1.2%)` after H2, before a resist-break fill |
| Shallow | Cancel-bar close undershoot / width `<= 0.25` (TARS was 0.11) |
| Recovered | Next daily close back at/above that bar's support |
| TARS-like | Shallow **and** recovered |
| Later breakout | First close `> resist * (1 + 1.2%)` after cancel, still inside max_wait from H2 |
| Last-15m sleeve | That session's last RTH 15m mid if open > rail, then 15m N+1 ATR k=2 + 10% trail |

Not a skip overlay on the 2297 fills (those H2s already filled). This walks setups that **never** printed in the book.

## Count (8909 span365 H2s)

| | n | Later breakout | Rate |
|--|---|----------------|------|
| Setups | 8909 | — | — |
| Filled first (normal H2) | 2470 | — | — |
| Expired (no fill, no cancel) | 109 | — | — |
| **Cancelled before fill** | **6330** | **1812** | **28.6%** |
| Shallow cancel | 4379 | 1231 | 28.1% |
| Recovered next | 742 | 265 | 35.7% |
| **TARS-like (shallow + recovered)** | **679** | **239** | **35.2%** |

A yellow one-bar poke is common (**4379 / 6330**). Shallowness does **not** raise the later-breakout rate vs all cancels (28.1% vs 28.6%). Next-bar recovery does a bit (35.7%). TARS is one of 239 TARS-like later-breakouts, not a one-off.

## Last-15m sleeve (occupancy not re-walked)

| Sleeve | n | E | PF | geo year PF | WR | 2020-21 | 2022-23 | overlap open |
|--------|---|---|----|-------------|----|---------|---------|--------------|
| Last-15m book (gate) | 2297 | +0.27 | 1.09 | 1.095 | 31.4 | — | −0.90 / 0.73 | — |
| All-cancel later-breakout | 1707 | +0.25 | 1.08 | **0.89** | 29.6 | −0.05 / 0.99 | −0.21 / 0.94 | 264 |
| **TARS-like later-breakout** | **232** | **+1.15** | **1.38** | **1.21** | 31.5 | **−0.20 / 0.94** | +0.47 / 1.13 (n=41) | 32 |

Fric 0.25 on TARS-like: E **+0.90** PF 1.28. Median **−3.3%**. Hard-stop 110 / 232. Skip reasons on 239 daily reclaim: last-15m filled 232, end_of_session 4, no_open_cross 2, no_15m 1.

Drop-top-3: n=229 E **+0.41** PF 1.13. Winner $ 979 vs loser $ 712. Fat-tailed; 2020-21 fails the equal-weight year gate; n=232 is thin. **No promote.** Do not occupancy-walk.

All-cancel reclaim is the same book with extra noise (geo 0.89). Do not re-arm after every support break.

## TARS

| | |
|--|--|
| H2 | 2023-11-15 (L1 Apr-14 / L2 Sep-14, span 215, width 28%) |
| Cancel | 2023-12-01 undershoot 0.112×width, recovered next |
| Reclaim last-15m | 2023-12-27 15:45 ET mid **20.40** (rail 19.99) |
| Exit | 2024-01-05 09:45 ET **19.28** hard-stop **−5.49%** |
| Book overlap | none (July throwovers already closed) |

The Jan-2024 chart rally is still after the keeper stop (Jan-5 low 19.00 vs ~6% ceiling).

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\analyze_h2_cancel_reclaim.py --symbols TARS
python scripts\research\analyze_h2_cancel_reclaim.py --workers 8
```

Wall-clock: daily cache 5.5s + walk 13.1s; 15m cache 3.2s + sleeve 370s; wall **391.7s**. First 15m TimescaleDB load (killed full-history index) was 638s / 775 symbols.

Artifacts: `reports/ascending_channels/2026-09-12/h2_cancel_reclaim_setups.csv`, `h2_cancel_reclaim_last15m_all.csv`, `h2_cancel_reclaim_last15m_tars_like.csv`, `h2_cancel_reclaim_summary.json`.
