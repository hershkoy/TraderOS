# Unconstrained last-15m H2 + 15m L3 concurrent overlay (2026-09-13)

Raw overlay of the two `current_best` signal logs on one timeline. **No occupancy re-walk, no pooled P&L, no promote.** Question: max concurrent opens if both books stay always-on.

Last-15m H2 unique n=2297 (15m N+1 mid sells, median hold **15.75d**). 15m L3 wait-12 n=1744 (signal-close, median hold **0.81d**).

## Max concurrent

| Book | Event-time max names (slots) | Peak | Calendar B-day max / med / p95 |
|------|------------------------------|------|--------------------------------|
| Last-15m H2 | **97** (97) | 2021-03-12 20:45 UTC | **100** / 24 / 71 |
| 15m L3 wait-12 | **7** (7) | 2019-02-20 19:00 UTC | **8** / 2 / 4 |
| **Overlay H2+L3** | **98** (98) | 2021-02-25 14:30 UTC | **102** / 24 / 72 |

Event-time: +1 at `buy_time`, −1 at `sell_time`; sells before buys on a tie. Calendar: existing `concurrent_open_stats` (`buy_date` ≤ d ≤ `sell_date` on business days). Unique names vs trade-slots match at the blend peak (one same-symbol double at 2020 max: slots 91 / names 90). **80** H2 trades overlap an L3 on the same ticker at some point.

Event-time median along the path is 30 names (p95 74). That samples at trade events, not idle days; calendar median 24 is the crowding-note analog (1d unique next-mid was max 85 / med 18).

## Year peaks (event-time names)

| Year | Overlay | H2 | L3 |
|------|---------|----|----|
| 2019 | 78 | 74 | 7 |
| 2020 | 90 | 89 | 4 |
| **2021** | **98** | **97** | 5 |
| **2022** | **29** | **27** | **4** |
| 2023 | 65 | 62 | 4 |
| 2024 | 81 | 79 | 5 |
| 2025 | 42 | 40 | 5 |
| 2026 | 60 | 60 | 0 |

L3 never exceeds **7** concurrent in any year. 2022 overlay max **29** is still almost all H2 (27). L3 does not “find more setups” as capital occupancy in the rate-shock year; 15m holds are too short. H2 concurrent falls in 2022 because there are fewer fills (n=117) and more hard-stops, not because L3 takes the book.

## Gate

Answer: unconstrained overlay **max concurrent = 98 unique names** (calendar **102**). That is the H2 melt-up book plus one L3 slot. A live equal-dollar pool still needs a `max_open` cap; this is not a 2022 switch and not a scored blend. Do not occupancy-walk the blend as an edge until someone asks for pooled P&L vs last-15m alone. Follow-up occupancy (unified FCFS 8/15/20 vs dedicated 15+5): L3 is not locked out at cap 15 (76% kept); cap 8 starves H2; 2022 stays red; no promote.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\channel_touch_robustness.py --trades reports\ascending_channels\2026-09-11\channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv
```
