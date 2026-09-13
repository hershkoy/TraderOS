# Sit 2022 out vs TLT as a global gate (2026-09-13)

Same last-15m-open-mid + 15m N+1 mid book **n=2297 E +0.27 PF 1.09 geo 1.10**. Prior-session TLT only. Occupancy not re-walked. **No promote.**

The yield-shock story is right about calendar 2022. TLT below a medium MA is **not** a 2022 sit-out: it is off for most of the sample, including the 2020-21 winners, and leftover 2023 is worse.

## 1. Oracle cash sit-out (peeking)

Drop trades by calendar year. No live sensor.

| Book | n | E | PF | geo | 2022 | 2023 | 2022-23 |
|------|---|---|----|-----|------|------|---------|
| Baseline | 2297 | +0.27 | 1.09 | 1.10 | n=117 E **−2.37** PF 0.41 | n=300 E **−0.33** PF 0.90 | −0.90 / 0.73 |
| **Sit cash 2022** | 2180 | +0.41 | 1.13 | 1.15 | 0 | n=300 E **−0.33** PF 0.90 | −0.33 / 0.90 |
| Sit cash 2022-23 | 1880 | +0.53 | 1.17 | 1.25 | 0 | 0 | — |

2022 losers are real (skip win$ +195 vs lose$ −472, not blunt). 2023 last-15m stays red. Peeking the year does not make resist-chase work in chop. The [L3 splice](2026-09-12_synthetic_2022_l3_splice.md) is the same peek with 2022 replaced by wait-12 (n=2423 E +0.37) — still not a switch.

## 2. TLT global gate (live, prior session)

Skip last-15m H2 when prior-session TLT close is below SMA100 / SMA200, or 63d ROC < 0. Missing TLT keeps.

| Gate | n | E | PF | geo | 2022 leftover | 2023 leftover | 2020-21 |
|------|---|---|----|-----|---------------|---------------|---------|
| TLT < SMA100 | 795 | +0.37 | 1.12 | **1.02** | 4 / **−3.90** | 99 / **−1.69 / 0.51** | **E −0.65 PF 0.81** blunt |
| TLT < SMA200 | 760 | +0.12 | 1.04 | **0.91** | 1 | 44 / **−1.98 / 0.52** | E −0.55 PF 0.82 blunt |
| TLT 63d ROC < 0 | 878 | +0.19 | 1.06 | **0.97** | 13 / −2.14 | 110 / **−1.57 / 0.54** | E −0.73 PF 0.79 blunt |

SMA100 **does** skip 113 / 117 of 2022. It also skips **1502 / 2297** fills (first off 2019-10-30, last 2026-08-24): 509 of 2020-21, 201 of 2023, 621 of 2024-26. That is a multi-year bond-bear flag, not a 2022 kill switch.

Leftover 2023 is worse than the full 2023 sleeve (−0.33 / 0.90): TLT had already stabilized, so the gate keeps the failed resist-chases. Geo-PF of year buckets falls vs baseline (1.02 vs 1.10) because 2020-21 turns red. Winner-$ on 2020-21: drop win$ 1977 vs lose$ 1542 — **blunt**.

[Regime switch](2026-09-13_last_15m_regime_switch.md) already had the H2-skip + L3 splice (`SWITCH_TLT100_L3` n=1722 E +0.31 PF 1.18 geo 1.19; 2022-23 E −0.22). Mixing ~2.5% L3 winners with last-15m does not clear the gate either.

## Why

2022 last-15m H2 is a rate-hike failed-breakout book. TLT below SMA100 tags that year. Bonds also spent 2020-21 and chunks of 2024-26 below that MA, when this book made money. When TLT recovered in 2023, H2 win rate was still 27%. Same failure mode as SPY SMA / VIX / HMM / breadth: skip the obvious bear, keep bounce chases, clip other-regime winners.

15m L3 wait-12 remains the 2022-23 book that is already green. Do not wire TLT as a last-15m off switch.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_regime_switch.py
```

One-shot sit-out vs TLT year buckets was printed 2026-09-13 (not a new runner).
