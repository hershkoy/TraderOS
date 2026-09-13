# Choppy-regime RSI-2 MOC + 3-day pullback (2026-09-13)

Implementation of [choppy_regime/ideas.md](../../strategies/choppy_regime/ideas.md). **No promote.**

Not Phase 5 `rsi2_stock` / `dump3_stock` (those used SMA20, RSI<=10, next-open, SMA20 exit / 3-day % dump). This book is Connors-style **MOC** RSI(2)<5 above SMA200, and an **evening limit** on exactly three lower lows.

Runner: `scripts/research/run_choppy_regime.py`. Logic: `utils/research/choppy_regime.py`. Scan: `scripts/scanners/choppy_regime_scan.py`.

## Clock

**RSI-2 MOC.** Signal on the session close: stock > SMA200, Wilder RSI(2) < 5. Fill is that **same close** (3:45 last vs official close can differ; this is the Connors proxy, not next-open). Skip if **prior-session** VIX > 30 (cash VIX is not closed at 15:45 ET) or if the session is the last trading day before a BLS CPI 08:30 ET print. Exit: close > SMA5 or RSI(2) >= 70, else time stop 8 sessions. No same-bar re-entry.

**3-day pullback.** Setup on a **completed** close: close > SMA100, exactly three consecutive lower lows, close < SMA5. Next session GTC limit = close − 0.5 × ATR(10). Fill at the limit, or at the open if the open gaps through; a miss cancels. First later close higher than the prior close sells the **next morning open**. Same-session VIX > 30 skip (VIX is known after the setup close). No time stop.

PIT top-500 ADV lagged one session, min price $10, min ADV $1M. ALPACA 1d only (no IB prefix). Panel 2017-01-03 → 2026-08-21 is **1527 dates × 2217 names** (Alpaca coverage, not 2017). Wall **352s**.

## Results

SPY (IB, eval 2018-01-01 → 2026-08-21): **+183.7%**, Sharpe 0.67, MDD −34.1%.

| Book | n | E | PF | geo | WR | median | notes |
|------|---|---|----|-----|----|--------|-------|
| RSI-2 unconstrained | 5451 | **+0.46** | **1.36** | **1.30** | 66% | +1.07 | max concurrent **172**; SMA5 4963 / time 455 / eod 33 |
| RSI-2 fric 0.25 | 5451 | +0.21 | 1.15 | 1.09 | — | — | |
| RSI-2 occ 15 | 2198 | +0.18 | 1.12 | 1.09 | 65% | +1.03 | |
| RSI-2 occ 15 fric 0.25 | 2198 | **−0.07** | **0.96** | 0.93 | — | — | friction kills the cap-15 book |
| Pullback unconstrained | 4536 | +0.26 | 1.23 | 1.06 | 58% | +0.42 | max concurrent **104**; next-open 4509 |
| Pullback fric 0.25 | 4536 | +0.01 | 1.01 | 0.85 | — | — | |
| Pullback occ 15 | 2736 | +0.31 | 1.27 | 1.13 | 59% | +0.52 | |
| Pullback occ 15 fric 0.25 | 2736 | +0.06 | 1.05 | 0.92 | — | — | |

Year buckets (gross; geo skips n<30 so **2018-19 and 2020-21 do not enter**):

| Book | 2018-19 | 2020-21 | 2022-23 | 2024-26 |
|------|---------|---------|---------|---------|
| RSI-2 | n=0 | n=20 E −0.20 PF 0.87 | n=986 E +0.25 PF 1.22 | n=4445 E +0.51 PF 1.38 |
| Pullback | n=0 | n=15 E −0.77 PF 0.37 | n=1303 E **−0.28 PF 0.77** | n=3218 E +0.48 PF 1.45 |

Calendar **2022** (not the 2022-23 bucket): RSI-2 **n=10** E +1.49 (VIX>30 sits out almost all of 2022 — not a sample). Pullback n=110 E +0.87 PF 3.22; the 2022-23 red is **2023** (~1193 trades). Drop-top-3 barely moves E (not a fat-tail book). Hold median 3d (RSI-2) / 2d (pullback).

## Verdict

**No promote.** Not a 2022 last-15m H2 rescue.

Unconstrained RSI-2 PF 1.36 is still in the 1.0–1.5 fragile band. Equal-dollar occ-15 with 0.25% friction is **below 1**. Pullback **2022-23 PF 0.77** fails the year-bucket gate; geo 1.06 is 2024-26 carrying a red regime. Empty 2018-19 is Alpaca history + SMA200 warmup, not a tested early window (no IB prefix). VIX>30 does what it says for RSI-2 (almost no 2022 fills) and so cannot be graded as a 2022 chop edge. Max concurrent 172 / 104 is not a live equal-dollar book. Do not occupancy-walk this onto last-15m.

The stored-daily scanner is a post-close checklist. Alpaca 1d is typically ready after 17:00 ET, which is **too late for a same-day MOC**; live 15:45 would need a developing close (Alpaca last), not this TimescaleDB snapshot.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\run_choppy_regime.py
python scripts\scanners\choppy_regime_scan.py
```

Artifacts: `reports/choppy_regime/2026-09-13/`.
