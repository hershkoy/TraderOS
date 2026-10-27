# Prior-session option filters on last-15m (2026-10-18)

Same-list skip on the last-15m-open-mid + 15m N+1 mid book **n=2297 E +0.27 PF 1.09** (geo 1.095). 2022 is **n=117 E −2.37 PF 0.41**. The stock book stays the book. These marks only decide whether the stock buy is taken. Occupancy not re-walked. **No promote.**

As-of is the **prior completed session** (the fill is 15:45 ET; that day's close IV is a leak). Source is Theta `v3/option/history/greeks/eod`. Around 30 DTE: ATM IV is the strike nearest 50 delta, skew is 25-delta put IV minus ATM IV. Each series is a trailing 252-session percentile (min 126). A missing mark **keeps** the trade. One skip per layer, walked in a fixed order, locked only when the pre-registered tail drops at least 30 trades, is not a winner-dollar clip, beats the kept mean, and does not make 2022 worse. No cutoff search.

Cash VIX/VIX3M backwardation and the VX futures slope already failed on this list. SPY 30-day ATM IV is that same level, so it was not tested again.

Runner: `scripts/research/overlay_last_15m_option_filters.py`. Features: `utils/research/last_15m_option_filters.py`. Sector map is a non-Custom/Clean GICS label, else the latest SIC (`1910/2297` fills). Yahoo closes are split-scaled onto listed strikes. Wall clock **17028 s (~4.7 h)**, almost all of it the liquid-name chain pull.

## Quintiles (what the tails actually are)

| Feature | Covered | Q5 (top) E / PF | Q5 n | 2022 in Q5 E | Locked? |
|---------|---------|-----------------|------|--------------|---------|
| SPY skew percentile | 816 | **+1.47 / 1.55** | 414 | **+0.23** | no — top is the right tail (blunt, 2022 worse) |
| SPY skew 5-session change | 651 | −0.13 / 0.96 | 66 | −3.30 | **yes, top** |
| Dispersion (SPY IV − sector mean) | 1030 | **+1.70 / 1.61** | 129 | −2.21 | no — top is the right tail |
| Sector ATM IV percentile | 1826 | **+0.96 / 1.33** | 532 | −1.92 | no — top is the right tail (Q1 is the weak one, E −0.84) |
| Sector skew percentile | 1586 | −0.18 / 0.95 | 428 | −3.09 | **yes, top** (Q4 is the fat winners, E +1.81) |
| Name ATM IV percentile | 251 / 761 liquid | −0.64 / 0.82 | 51 | +0.86 | top no (2022 leftover worse); **bottom yes** (Q1 E −0.18) |
| Name skew percentile | 157 | **+1.42 / 1.50** | 29 | −1.89 | no — fewer than 30 drops, and the top is green |
| Name IV − sector IV | 216 | **+0.83 / 1.29** | 38 | +1.37 | no — top is green and 2022 leftover worse |

Name marks are ADV ≥ $25M only (378 symbols, 761 trades). A trailing percentile needs 126 sessions, and about 58% of the snapped strikes came back empty, so most liquid fills stay unmarked and are kept.

## Locked skips

| Overlay | n | E | PF | geo | 2022 n / E / PF | dropped win$ / lose$ | dollar ratio |
|---------|---|---|----|-----|-----------------|----------------------|--------------|
| Baseline | 2297 | +0.27 | 1.09 | 1.095 | 117 / **−2.37 / 0.41** | — | — |
| SPY skew-change top | 2231 | +0.28 | 1.09 | 1.103 | 104 / −2.25 / 0.44 | 199 / −208 | 0.96 |
| Sector skew top | 1869 | +0.37 | 1.12 | 1.182 | 82 / −2.06 / 0.49 | 1324 / −1399 | 0.95 |
| Both ETF skips | 1815 | +0.39 | 1.13 | 1.197 | 71 / **−1.70 / 0.56** | — | — |
| Name IV bottom | 2218 | +0.28 | 1.09 | 1.099 | 115 / −2.36 / 0.42 | 228 / −242 | 0.94 |

Pooled winner-fraction of the drops is about 0.27–0.29, under the 0.35 blunt cutoff, which is why the locks fired. The dollars do not. Each locked skip gives back about as much winner P&L as it saves in loser P&L.

## Winner-$ on the fat-winner windows

| Overlay | 2020-21 kept E / PF | 2020-21 drop win$ / lose$ | ratio | 2024-26 kept E / PF | 2024-26 drop win$ / lose$ | ratio |
|---------|---------------------|---------------------------|-------|---------------------|---------------------------|-------|
| Baseline | +0.46 / 1.15 | — | — | +0.48 / 1.16 | — | — |
| SPY skew-change top | +0.39 / 1.12 | 191 / −126 | **1.51** | +0.48 / 1.16 (nothing dropped) | 0 / 0 | — |
| Sector skew top | +0.67 / 1.23 | 595 / −625 | 0.95 | +0.45 / 1.15 | 516 / −422 | **1.22** |
| Name IV bottom | +0.51 / 1.16 | 58 / −82 | 0.72 | +0.46 / 1.15 | 117 / −78 | **1.51** |

Sector skew is the only skip with a visible geo lift, and 2024-26 pays for it: the dropped trades in that window are winner-heavy (2025 drops alone are E +3.03). SPY skew-change hurts 2020-21 and leaves 2024-26 untouched. Name IV bottom does not move 2022 (2 of 117 dropped) and clips 2024-26.

## Why

A rich SPY put-wing, a rich sector IV, and a rich name-versus-sector spread are where the right tail lives. Skipping those tails is the same failure as the VIX curve gate: the filter sits out the prints that pay for the book. The tails that do look weak (a five-day jump in SPY skew, a steep sector put-wing, a name whose own IV is in its bottom quintile) separate losers from winners by a few tenths of a percent of expectancy, with dollar ratios near 1. 2022 stays red on every lock (best leftover is the two ETF skips together, E −1.70). That is why the name pull ran, and the name pull did not close the hole.

Do not wire these skips into the nightly book or `/hot`. Do not search a tighter percentile. Do not occupancy-walk them.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_option_filters.py
```

Artifacts: `reports/ascending_channels/2026-10-18/option_filter_summary.json`, `last15m_option_features.csv`, `quintile_*.csv`, `etf_option_panel.csv`. Chain cache: `reports/ascending_channels/option_filters/cache/`.
