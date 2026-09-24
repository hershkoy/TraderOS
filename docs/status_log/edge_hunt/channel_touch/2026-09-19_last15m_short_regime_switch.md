# Last-15m H2 vs short H3+ regime switch (2026-09-19)

Same pre-registered sensors as the [L3 switch](2026-09-13_last_15m_regime_switch.md), but when long H2 is OFF take **15m short H3+** instead of L3 / cash. Not an occupancy blend ([blend no-promote](2026-09-19_last15m_resist_short_blend.md)).

Inputs: last-15m unique + 15m N+1 mid **n=2297**; short unique span10 fric0.10 **n=7004**. Occupancy **not** re-walked. Mixed-scale pooled E is not a gate. **No promote.**

Runner: `scripts/research/overlay_last15m_short_regime_switch.py`. Logic: `utils/research/last_15m_regime_switch.py` (`combine_h2_alt`, `keep_when_*`).

## Baseline

| Book | n | E | PF | geo | 2022 E / PF | 2023 E / PF | 2022-23 E / PF |
|------|---|---|----|-----|-------------|-------------|----------------|
| Last-15m H2 | 2297 | +0.27 | 1.09 | 1.10 | −2.37 / 0.41 | −0.33 / 0.90 | −0.90 / 0.73 |
| Short H3+ all | 7004 | +0.21 | 1.47 | 1.44 | +0.26 / 1.57 | +0.20 / 1.45 | +0.23 / 1.52 |

## Rolling paper WR / PF

| Book | n | long/short | E | PF | geo | 2022 E / PF | 2023 E / PF | 2022-23 |
|------|---|------------|---|----|-----|-------------|-------------|---------|
| SWITCH WR20 | 4125 | 1131 / 2994 | +0.23 | 1.20 | 1.13 | **−0.48 / 0.64** | +0.07 / 1.12 | −0.08 / 0.89 |
| SWITCH WR30 | 4203 | 1254 / 2949 | +0.20 | 1.16 | 1.23 | **−0.16 / 0.84** | +0.01 / 1.02 | −0.05 / 0.93 |
| SWITCH PF20 | 4955 | 916 / 4039 | +0.20 | 1.21 | 1.18 | +0.17 / 1.29 | +0.29 / 1.59 | +0.22 / 1.42 |

WR still fails 2022 (sensor on into early bear-rally; leftover long worse than full sleeve). PF20 greens 2022/23 only because the book is **~82% short** (4039/4955) and **2024-26 year bucket is flat** (E −0.01 PF 0.99). That is "mostly trade the short sleeve," not a long↔short switch that preserves last-15m.

## Macro (TLT / UUP / SPY)

| Book | n | long/short | E | PF | geo | 2022 E / PF | 2023 E / PF | Fail |
|------|---|------------|---|----|-----|-------------|-------------|------|
| SWITCH TLT100 | 4531 | 795 / 3736 | +0.23 | 1.25 | 1.32 | +0.24 / 1.50 | **−0.06 / 0.94** | 2023 red; TLT skip still blunt 2020-21 |
| SWITCH TLT200 | 4683 | 760 / 3923 | +0.19 | 1.22 | 1.26 | +0.26 / 1.55 | +0.10 / 1.16 | ~84% short; geo << short-alone 1.44 |
| SWITCH UUP50 | 5026 | 1132 / 3894 | +0.18 | 1.17 | 1.18 | +0.15 / 1.24 | **−0.05 / 0.95** | 2023 |
| SWITCH UUP100 | 4964 | 1114 / 3850 | +0.31 | 1.30 | 1.25 | +0.23 / 1.46 | **−0.13 / 0.88** | 2023 |
| SWITCH SPY200 | 3781 | 2112 / 1669 | +0.24 | 1.12 | 1.15 | +0.05 / 1.08 | **−0.51 / 0.80** | 2023 |
| SWITCH SPY100 | 3776 | 2032 / 1744 | +0.19 | 1.10 | 1.13 | +0.08 / 1.13 | **−0.34 / 0.84** | 2023 |

## Verdict

**No promote.** Same failure mode as the L3 switch and the occupancy blend:

- Rolling WR/PF is not a 2022 sensor; leftover long 2022 is worse; PF20 "works" only by living in short.
- TLT/UUP/SPY tag 2022 then leave **2023 last-15m red** (or blunt 2020-21 when used as a long skip).
- No sensor is a clean long-only ↔ short-only flip that clears both 2022 and 2023 while keeping last-15m as the default sleeve.

Trade short H3+ as its own research book if desired ([resist short](2026-09-16_resist_short_h3.md)); do not wire it as a regime alternate to last-15m via these gates. Occupancy not re-walked.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last15m_short_regime_switch.py
```

CSV: `reports/ascending_channels/2026-09-19/last15m_short_regime_switch_summary.csv`.
