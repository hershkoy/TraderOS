# 1d close-cross + trail MAE (RDWR microscope, 2026-09-06)

Diagnostic, not a promote. `--intraday-trigger close-cross`: 1d H2 arms on prior completed daily bars; first RTH 15m **close > daily rail**; fill is the **next** 15m mid (`purchase_after_close_signal`). Does not wait for that session's daily close. `--trail-mae` walks 15m from the fill with squeeze 10/18 and **no hard stop** (`stop_pct=1.0`); `mae_pct` / `mae_atr_15m` / `mae_atr_1d` is the stop that would have survived that trail path.

Confirm-bar features (causal, that 15m close is known): volume vs 20-bar and vs same TOD, range, wicks, close/open vs rail in % and ATR, session failed wick-closes, rail slope × wait (`rail_rise_since_h2_pct`), width in daily ATR. Skip rows (`wild_low_to_mid`, occupancy) stay in the CSV.

## RDWR-only run

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --symbols RDWR --shakeout-breakout --intraday-trigger close-cross --realistic-fill --realistic-fill-mode next-mid --trail-mae --workers 1
```

CSV: `reports/ascending_channels/2026-09-07/channel_touch_h2_resist_break_*.csv` (and span365 / unique). **36 rows / 6 fills / 30 skips** (17 occupancy, 13 `wild_low_to_mid`, 0 `end_of_session`).

| confirm ET | fill UTC | fill px | vol_tod | trail-only | MAE % | MAE ATR 1d | keeper exit |
|------------|----------|---------|---------|------------|-------|------------|-------------|
| 2020-02-11 10:00 | 15:15 | 26.62 | 0.44 | −9.59 trail | 10.29 | 6.56 | hard_stop |
| 2021-08-18 14:15 | 18:30 | 33.12 | 7.14 | +8.10 trail | 2.63 | 1.21 | trail_stop |
| 2024-10-14 10:00 | 14:15 | 24.05 | 2.10 | −8.90 trail | 10.04 | 5.09 | hard_stop |
| 2025-06-05 11:15 | 15:30 | 24.41 | 0.65 | +16.44 trail | 1.93 | 0.93 | trail_stop |
| 2025-09-18 11:00 | 15:15 | 26.93 | 1.45 | −6.41 trail | 6.63 | 3.36 | hard_stop |
| 2026-05-15 13:30 | 17:45 | 28.03 | 0.36 | +2.49 trail | 1.89 | 0.59 | trail_stop |

Occupancy-honest span365 unique n=3 (2024-10-14, 2025-06-05, 2025-09-18). n=6 raw is not a book.

**RDWR 2025-06-13 never prints.** Close-cross filled **2025-06-05 11:15 ET** (next mid 11:30 ET / 15:30 UTC 24.41), trail-only +16% with MAE 1.9%. Jun-9/10 shakeout extras are occupancy skips. Jun-13 is not even a close-cross candidate on this setup (hot-cross had a Jun-5 wick poke; open-cross bought Jun-13 09:45 close 25.71). A wick through the rail with close still below is not a signal.

Wild cancels are mostly 09:30 bars that close well through the rail (`close_over_rail` 2–12%) — the 0.5% low-to-mid gate on the *next* bar, not the confirm. Keep them as skip rows; do not loosen the gate from n=6.

No volume→stop formula. On this name the three trail-only winners had MAE ~2% (~0.6–1.2 daily ATR); the three trail-only losers needed ~6–10% (~3–6 daily ATR) to survive. That is a hint, not a rule.

Do not wire `/hot` to close-then-wait-15m. Do not replace hot-cross `current_best` (n=4079 E −0.19 PF 0.94). Full-universe only if RDWR still looks like a microscope, not a book.
