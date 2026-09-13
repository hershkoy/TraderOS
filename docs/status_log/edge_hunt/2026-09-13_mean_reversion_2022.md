# Daily Bollinger mean reversion in 2022 (2026-09-13)

Test of `strategies/mean_reversion_strategy.py` on calendar **2022** (not a last-15m overlay, not a promote).

Buy: completed daily close at/below the lower Bollinger (period 20, k=2). Exit: close back at/above SMA20 (the upper-band exit in code is redundant). **No stop.** Signal on close, fill **next open**. Commission 0.1% per side (20 bps RT). Universe: ALPACA 1d, PIT top-500 ADV, min price $10, min ADV $1M. Panel 2021-08-02 -> 2022-12-31 (2063 names x 358 dates) for SMA/ADV warmup. Wall **236s**.

Also re-simmed Phase 5 `dump3_stock` on the same isolated 2022 window (3-day dump, SMA20 / 8d / 10% stop, 10 bps RT, max 15).

## Results

SPY 2022 (IB): **−19.95%**, Sharpe −0.83, MDD −25.4%.

| Book | n | E | PF | WR | median | vs SPY |
|------|---|---|----|----|--------|--------|
| BB unconstrained (1 pos/symbol) | 1041 | **−0.16%** | **0.95** | 56% | +0.85 | trade book, not a portfolio |
| BB occupancy max 15 (most oversold) | 141 | +1.24% | 1.40 | 64% | — | CAGR **+11.4%** Sharpe **0.59** MDD **−10.2%** |
| dump3_stock occupancy 15 | 364 | ~0 | 1.00 | 49% | — | CAGR **−2.3%** Sharpe −0.09 MDD −17.6% |

Unconstrained extras: SMA exits n=840 E +0.46 PF 1.15; year-end leftovers n=201 E **−2.75**. Avg win +5.14 vs avg loss **−6.92** (no stop). Max concurrent **410** (2022-09-28), median 6, 531 symbols. Drop-top-3 E −0.24. Worst: GNRC −49%, LCID/COIN eod −43/−42, AMC −39.

Monthly unconstrained (buy month):

| Month | n | E | PF |
|-------|---|---|----|
| Jan–Jul | 45 | mixed, small n | — |
| **Aug** | 193 | **−4.68** | **0.25** |
| Sep | 364 | −0.05 | 0.98 |
| **Oct** | 99 | **+5.93** | **8.55** |
| **Nov** | 83 | **+5.28** | **3.66** |
| **Dec** | 257 | **−1.15** | **0.47** |

Occupancy +11% is the Oct–Nov bounce on the 15 deepest band-breaks, not the raw strategy. Unconstrained taking every liquid poke is flat-to-red. dump3 isolated 2022 is **−2.3%** (Phase 5 full-path year return was **−6.0%** on the running book).

## Verdict

**No promote. Not a 2022 last-15m H2 rescue.**

Unconstrained PF < 1. Occupancy beats SPY in 2022 on return and MDD but **Sharpe 0.59 fails Sharpe > 1.0**; n=141 is a one-year bounce sample. No-stop left tail is the 2022 tell (knives that never tag SMA). Phase 5 already closed swing MR (best full-sample Sharpe 0.28). Do not occupancy-walk this onto last-15m.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\run_mean_reversion_2022.py
```

Artifacts: `reports/mean_reversion/2026-09-13/`.
