# Last-15m AND Deepvue weekend take-4 (2026-09-13)

Same-list join of the last-15m-open-mid + 15m N+1 mid book **n=2297 E +0.27 PF 1.09** (geo 1.095) with the causal Deepvue weekend take-4 (**204** focus rows, occupancy **n=179 E +2.25 PF 1.85 geo 0.985**). Occupancy not re-walked. **No promote.**

Pre-registered:

- **L15_and_week** — last-15m fill in the week after a same-name Deepvue take-4 (`signal_date < fill <= signal_date+7`)
- **L15_and_signal_day** — last-15m fill on the Friday signal session itself
- **L15_ever_focused** — last-15m name ever appeared on the Deepvue take-4 (universe, not a week AND)
- **DV_and_l15_week** — Deepvue occupancy trade whose name also printed last-15m that week

Trade-week is causal: Deepvue's list is known after Friday cash close; last-15m during the following Mon-Fri does not use that Friday's last 15m. Runner: `scripts/research/overlay_last_15m_deepvue_and.py`. Join helpers: `utils/research/deepvue_l15_and.py`.

## Overlap

| Count | n |
|-------|---|
| Last-15m fills / stocks | 2297 / 940 |
| Deepvue signals / stocks | 204 / 94 |
| Deepvue occupancy trades | 179 / 93 |
| Stocks that ever appear in both | **59** |
| **L15_and_week** | **1** |
| **L15_and_signal_day** | **0** |
| **DV_and_l15_week** | **1** |
| L15_ever_focused fills | 161 |

The one week AND is **MRNA**: Deepvue signal 2021-06-18, last-15m fill 2021-06-22 hard-stop **−6.29%**. Deepvue's own occupancy row that week is **+86.2%** (Monday 09:30 15m mid + 3% stop / two closes below 21 EMA). Same name, same week, opposite books because the clocks and exits differ.

## Results

| Book | n | E | PF | geo | 2022 n / E / PF | 2020-21 E / PF | 2024-26 E / PF |
|------|---|---|----|-----|-----------------|----------------|----------------|
| Last-15m baseline | 2297 | +0.27 | 1.09 | 1.095 | 117 / **−2.37 / 0.41** | +0.46 / 1.15 | +0.48 / 1.16 |
| Deepvue baseline | 179 | +2.25 | 1.85 | 0.985 | 11 / **−1.57 / 0.30** | +4.57 / 2.76 | +3.02 / 2.09 |
| **L15_and_week** | **1** | −6.29 | 0 | n/a | 0 | −6.29 / 0 | — |
| **L15_and_signal_day** | **0** | — | — | — | — | — | — |
| **L15_ever_focused** | 161 | +1.47 | 1.45 | 1.113 | 12 / **−1.94 / 0.51** | **−1.31 / 0.66** | +4.19 / 2.46 |
| **DV_and_l15_week** | **1** | +86.2 | — | n/a | 0 | +86.2 | — |

Geo on L15_ever_focused skips 2018-19 (n=7). The 1.113 vs 1.095 bump is 2024-26 PF 2.46 against a red 2020-21. That is not a robust overlay.

L15_ever_focused 2020-21 winner-cut drops win$ 2372 vs lose$ 2002 (rest of the book). Kept Deepvue-name last-15m H2s in 2020-21 are the worse chases, not a quality sleeve. 2022 leftover stays red (−1.94 / 0.51).

## Why the buys almost never meet

Deepvue's take-4 is tightness near the 10/21 EMA in high-RS leaders after a QQQ-risk-on Friday (coiled). Last-15m H2 is the last RTH 15m **open already above** the daily resist rail (already expanding). Those are opposite locations on the swing. Deepvue also takes at most 4 names a week from a 300 ADV slice; last-15m is 940 names, unique-symbol/day.

An OR of the two occupancy books is not a combined buy: both already fail 2022-23 on their own (last-15m E −0.90 PF 0.73; Deepvue E −2.16 PF 0.17). A two-sleeve portfolio of failed-breakout books does not clear the gate. Do not occupancy-walk Deepvue's RS/RMV/QQQ screen onto last-15m to manufacture overlap — QQQ-above-rising-21 is the same failure mode as prior-session SPY SMA, and ever-focused already shows Deepvue names as a last-15m universe wreck 2020-21.

Do not AND the buy signals. Do not treat the MRNA +86 Deepvue print as confirmation of last-15m.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_deepvue_and.py
```

Artifacts: `reports/ascending_channels/2026-09-13/last_15m_deepvue_and_summary.csv`.
