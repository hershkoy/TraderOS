# Last-15m earnings / biotech event filters (2026-10-26)

Same-list overlays on current_best last-15m-open-mid + 15m N+1 mid **n=2297 E +0.27 PF 1.09 geo 1.10**. Occupancy not re-walked. Yahoo earnings cache (AMC if hour≥16). Name ATM IV from the 2026-10-18 Theta marks. **No promote.**

Smoke context: AMPL 2024-02-09 gap was an earnings print; VST/TARS/MRNA were the chart losers. Filters 1–4 were pre-registered (no cutoff search).

## Book

| Overlay | n | E | PF | geo | 2022 E / PF | 2023 E / PF | 2022-23 |
|---------|---|---|----|-----|-------------|-------------|---------|
| Baseline | 2297 | +0.27 | 1.09 | 1.10 | −2.37 / 0.41 | −0.33 / 0.90 | −0.90 / 0.73 |
| **2** entry skip AMC/BMO | 2256 | +0.25 | 1.08 | **1.07** | −2.70 / 0.34 | −0.44 / 0.86 | −1.07 / 0.69 |
| **3** EM>ATR-stop skip | 2285 | +0.28 | 1.09 | 1.10 | −2.37 / 0.41 | −0.34 / 0.89 | −0.91 / 0.73 |
| **4** biotech sleeve off | 2125 | +0.27 | 1.09 | 1.10 | −2.47 / 0.38 | −0.20 / 0.93 | −0.80 / 0.76 |
| **1** flatten ≤1 sess | 2256 | +0.25 | 1.08 | 1.08 | −2.65 / 0.34 | −0.39 / 0.87 | −1.01 / 0.70 |
| **1** flatten ≤5 sess | 2256 | +0.34 | 1.11 | 1.10 | −2.44 / 0.37 | −0.33 / 0.89 | −0.92 / 0.72 |
| **1** flatten ≤10 sess | 2256 | **+0.38** | **1.13** | 1.11 | −2.24 / 0.40 | −0.31 / 0.90 | −0.85 / 0.73 |

Wall **344 s** (earnings refetch of empty Yahoo rows + cached IB 15m). Earnings coverage after refresh: **922 / 940** symbols.

## Winner-$ / early-exit cuts

| Overlay | n_drop / n_flat | win$ / lose$ | dollar ratio | blunt? |
|---------|-----------------|--------------|--------------|--------|
| 2 entry skip | 41 | +209 / −150 | **1.39** | yes (clips more winner $ than it saves) |
| 3 EM>stop | 12 | +11 / −45 | 0.26 | no — too thin to matter |
| 4 biotech | 172 | +652 / −615 | **1.06** | borderline — winners ≈ losers |
| 1 flatten≤10 (skip only) | 41 | same as #2 | 1.39 | skip sleeve blunt |
| 1 flatten≤10 (early exits) | **229** flattened | winners Δ **−429** / losers Δ **+731** | — | early-exit sleeve is helpful |

Flatten≤10 early exits raise mean on those 229 rows from +0.71 to +2.02. That is real event-gap avoidance, not a 2022 fix.

## Smoke names

| Name | Fill | What happened under flatten≤10 |
|------|------|--------------------------------|
| **AMPL** 2024-02-09 | −21.3% → gap Feb-21 | **Caught** (6 sessions to AMC 2024-02-20). Exit last 15m mid → **−0.71%**. |
| **VST** 2021-02-18 | −19.6% | **Caught** (5 sessions to 2021-02-25 print). → **−1.70%**. Freeze-week + earnings on the calendar — not a pure biotech readout. |
| **TARS** 2023-07-20 | −19.4% | **Miss** — next print 2023-08-10 is **15** sessions out (`outside_window`). |
| **MRNA** 2021-06-22 | −6.3% | **Miss** — Aug-4 print is 30 sessions out. |

## Why no promote

1. **Tight entry skip (#2) hurts.** Only 41 fills (AMC tonight / BMO next). Pooled E/geo fall; 2022 leftover **worse** (−2.70). Dollar ratio 1.39 clips winners.
2. **EM > stop (#3) is almost a no-op.** Name ATM IV marked on 524/2297; only **12** trips. Overnight `IV*sqrt(1/252)` rarely beats the 1.5–6% ATR stop. 2022 unchanged.
3. **Biotech ban (#4) is a wash.** 172 drops; win$ ≈ lose$ (ratio 1.06). AMPL (software) and VST (power) stay. 2022 still −2.47.
4. **Flatten-before-print (#1) is the only useful mechanism**, and only at 5–10 sessions. AMPL-type gaps are fixed. Pooled E lifts to +0.38 / PF 1.13 — still fragile vs fric0.25 baseline ≈ flat, and **2022 stays −2.24**. 2022-23 PF still &lt; 1. Same-session AMC skips are the blunt #2 sleeve.

This is event-risk hygiene for live `/hot` (do not hold a known print), not a last-15m edge or a 2022 regime fix. Do not wire into nightly. Do not fish max_sessions past 10.

## Code / artifacts

- `utils/research/last_15m_earnings_event_filters.py`
- `scripts/research/overlay_last_15m_earnings_event_filters.py`
- `tests/unit/test_last_15m_earnings_event_filters.py` (8 passed)
- `reports/ascending_channels/2026-10-26/last_15m_earnings_event_filters_summary.csv`
- Flatten detail: `last_15m_flatten_sess{1,5,10}_detail.csv`
- Earnings cache: `reports/ascending_channels/earnings_event_filters/earnings_cache.json`

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_earnings_event_filters.py
```
