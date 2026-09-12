# Universe breadth + last-15m RTH exhaustion skip (2026-09-12)

Same-list skip on last-15m-open-mid + 15m N+1 mid **n=2297 E +0.27 PF 1.09** (geo 1.095). Occupancy not re-walked. **No promote.**

Pre-registered:

- **B50** — skip if prior-session ALPACA-universe % above own SMA20 < 0.50
- **B_med** — skip if that breadth < expanding median through t-1
- **X_m2** — skip if (15:45 open − 09:30 open) / 15m ATR14 (bars before 15:45) >= 2
- **X_d1** — skip if same run / prior-session daily ATR >= 1

Breadth is 2,217 ALPACA daily names with a valid SMA20 that day (not SPY). Fill-day daily closes unused. 15:45 open is known at the last-15m decision; 15:45 close is not. Runner: `scripts/research/overlay_last_15m_breadth_exhaustion.py`.

## Results

| Overlay | n | E | PF | geo | skipped | 2022 n / E / PF | 2023 E / PF | 2022-23 E / PF |
|---------|---|---|----|-----|---------|-----------------|-------------|----------------|
| Baseline | 2297 | +0.27 | 1.09 | 1.095 | — | 117 / **−2.37 / 0.41** | −0.33 / 0.90 | −0.90 / 0.73 |
| **B50** | 1525 | **−0.24** | 0.93 | 0.96 | 772 | 56 / −1.82 / 0.56 | **−0.90 / 0.72** | −1.09 / 0.68 |
| **B_med** | 1609 | +0.07 | 1.02 | 1.05 | 688 | 63 / −2.11 / 0.49 | −0.68 / 0.78 | −1.01 / 0.70 |
| **X_m2** | 373 | +0.01 | 1.00 | 0.86 | 1924 | 14 / **−3.86 / 0.13** | −0.82 / 0.74 | −1.58 / 0.54 |
| **X_d1** | 1372 | +0.35 | 1.11 | 1.103 | 925 | 75 / **−2.71 / 0.36** | −0.49 / 0.85 | −1.14 / 0.68 |

Mean prior-session breadth on joined fills was **0.55**. Last-15m H2 prints when internals are already majority-above-SMA20 — including 2022 bounce days. B50 makes 2022 less red (−1.82) by dropping 61 names and **turns 2020-21 and 2024-26 red** (blunt winner-$ both windows).

X_d1 is the only pooled E/geo bump (+0.27→+0.35 / 1.095→1.103). 2020-21 E rises (loser $ saved). **2022 leftover is worse (−2.71)** and 2024-26 E 0.48→0.25 (win$ 1752 vs lose$ 1393). Same bear-rally pattern as SPY SMA / VIX / HMM: skip the obvious grind, keep the chase.

X_m2 is too tight (n=373, 2018-19 n=19 < 30 for geo). Not a gate.

## Winner-$ (2020-21 / 2024-26)

| Overlay | 2020-21 win$ / lose$ | blunt | 2024-26 win$ / lose$ | blunt |
|---------|----------------------|-------|----------------------|-------|
| B50 | 1301 / −816 | **yes** | 1494 / −911 | **yes** |
| B_med | 617 / −613 | no | 1521 / −933 | **yes** |
| X_m2 | 2008 / −1804 | no | 3244 / −2675 | no |
| X_d1 | 653 / −872 | no | 1752 / −1393 | no |

## 15m L3 wait-12 is not a last-15m overlay

The 15m L3 wait-12 signal-close + touch 0% book is already `current_best/15m_channel_touch.html` (**n=1744 E +0.13 PF 1.22**; 2022 +0.05 / 1.09, 2023 +0.30 / 1.56). Oracle calendar splice of that book onto last-15m 2022 did not make 2023 last-15m work. Avg win is ~2.5% vs last-15m ~9% — different scale.

Live 15m today is **H2 armed + H5 resist-break**, not L3. Nightly is **daily H2 rail-clip**. Productionizing L3 means changing `channel_touch_15m.py` / `/hot` FILLED to a wait-12 support-tag, not another skip on the 2297 1d book. Do that only as an explicit live-path change.

Do not threshold-fish breadth 40/60 or run-ATR 0.8/1.2. No L1 / OFI until a skip clears **2022 itself** without the winner-$ hit.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\overlay_last_15m_breadth_exhaustion.py --workers 8
```

Artifacts: `reports/ascending_channels/2026-09-12/last_15m_breadth_exhaustion_summary.csv`.
