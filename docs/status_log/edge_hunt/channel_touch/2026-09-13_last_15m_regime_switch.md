# Last-15m H2 vs L3 causal switch (2026-09-13)

Can a **live** sensor replace the [oracle 2022=L3 splice](2026-09-12_synthetic_2022_l3_splice.md)? Pre-registered: rolling paper H2 WR/PF, TLT below SMA, UUP above SMA. Last-15m unique + 15m N+1 mid **n=2297**; L3 wait-12 signal-close **n=1744**. Occupancy **not** re-walked. **No promote.**

Runner: `scripts/research/overlay_last_15m_regime_switch.py`. Logic: `utils/research/last_15m_regime_switch.py`.

Paper H2 rolling stats include skipped names (sell_time strictly before the next fill). Macro joins are prior-session daily. Missing TLT/UUP keeps.

## Baseline

| Book | n | E | PF | geo | 2022 E / PF | 2022-23 E / PF |
|------|---|---|----|-----|-------------|----------------|
| Last-15m H2 | 2297 | +0.27 | 1.09 | 1.10 | −2.37 / 0.41 | −0.90 / 0.73 |
| 15m L3 all years | 1744 | +0.23 | 1.43 | 1.44 | +0.05 / 1.09 | +0.18 / 1.32 |

Oracle calendar 2022=L3 was n=2423 E +0.37 PF 1.13 geo 1.16 with **2022-23 still −0.16 / 0.92** (2023 last-15m stays). That is peeking.

## 1. Rolling 20/30-trade WR (and PF20)

Off if paper WR < 0.28; back on if WR >= 0.32. PF20 off < 1.0 / on >= 1.2.

| Book | n | E | PF | geo | 2022 taken / skip | leftover 2022 E | 2022-23 E / PF |
|------|---|---|----|-----|-------------------|-----------------|----------------|
| H2 WR20 skip | 1131 | +0.29 | 1.09 | **0.86** | 77 / 40 | **−2.82** | −2.89 / 0.29 |
| SWITCH WR20+L3 | 1899 | +0.25 | 1.12 | 1.03 | — | −1.50 | −0.60 / 0.59 |
| H2 WR30 skip | 1254 | +0.13 | 1.04 | 0.93 | 68 / 49 | −2.77 | −3.00 / 0.30 |
| SWITCH WR30+L3 | 1998 | +0.17 | 1.08 | 1.12 | — | −1.00 | −0.54 / 0.65 |
| H2 PF20 skip | 916 | +0.13 | 1.04 | 1.03 | 23 / 94 | −4.32 | −0.19 / 0.95 |
| SWITCH PF20+L3 | 1927 | +0.13 | 1.07 | 1.09 | — | −0.35 | +0.16 / 1.19 |

WR20 is **off 50% of H2 closes** from 2019-09-25 through 2026-06-16. It is not a 2022 sensor; 2021's 33% WR keeps H2 **on into 2022**, so the leftover 2022 prints are the early bear-rally chases (worse than the full 2022 sleeve). 2023 leftover n=31 E −3.06. PF20 covers more of 2022 then **turns 2024-26 H2 red** (E −0.37 PF 0.89). Switch pooled E mixes ~9% last-15m winners with ~2.5% L3 winners — do not grade that vs last-15m +0.27.

## 2. TLT below SMA100 / SMA200 / 63d ROC < 0

TLT < SMA100 **does** skip 113 / 117 of 2022.

| Book | n | E | PF | geo | 2022 leftover | 2020-21 | blunt? |
|------|---|---|----|-----|---------------|---------|--------|
| H2 TLT<SMA100 skip | 795 | +0.37 | 1.12 | 1.02 | n=4 E −3.90 | **E −0.65 PF 0.81** | **Yes** (win$ 1977 / lose$ 1542) |
| SWITCH TLT100+L3 | 1722 | +0.31 | 1.18 | 1.19 | E −0.02 PF 0.97 | E −0.04 PF 0.98 | — |
| H2 TLT<SMA200 skip | 760 | +0.12 | 1.04 | 0.91 | n=1 | E −0.55 PF 0.82 | **Yes** |
| H2 TLT 63d < 0 | 878 | +0.19 | 1.06 | 0.97 | n=13 E −2.14 | E −0.73 PF 0.79 | **Yes** |

2023 last-15m stays: TLT had already stabilized while H2 WR was still 27%. Switch 2022-23 **E −0.22 PF 0.82**. Same winner-$ failure as SPY SMA200.

## 3. UUP above SMA50 / SMA100 / 63d ROC > 0

| Book | n | E | PF | geo | 2022 taken / skip | leftover 2022 | 2022-23 E / PF | 2020-21 blunt |
|------|---|---|----|-----|-------------------|---------------|----------------|---------------|
| H2 UUP>SMA50 skip | 1132 | **−0.05** | **0.98** | 0.95 | 28 / 89 | −3.81 | −1.41 / 0.58 | **Yes** |
| SWITCH UUP50+L3 | 2113 | +0.12 | 1.06 | 1.07 | — | −0.45 | −0.39 / 0.76 | — |
| H2 UUP>SMA100 skip | 1114 | +0.61 | 1.20 | 1.02 | 10 / 107 | −3.78 | **−1.35 / 0.59** | No |
| H2 UUP 63d > 0 | 1124 | +0.35 | 1.11 | 1.00 | 13 / 104 | −2.58 | −1.59 / 0.55 | No |

UUP>SMA100 lifts pooled E by skipping a lot of 2022 **and** keeping 2020-21, but **2022-23 stays red** (2023 n=139 E −1.18). Dollar-uptrend is not the 2023 failed-breakout regime.

## Verdict

**No promote.** None of the three sensors is a switch:

- Rolling WR/PF is lagged and fires in every year (off ~half the book). Leftover 2022 is worse.
- TLT below SMA is a 2022 tag and a **2020-21 winner-$** gate fail; 2023 H2 remains.
- UUP uptrend skips most of 2022 then leaves 2023 red; SMA50 skip loses pooled PF.

This is the same failure mode as SPY SMA / VIX / HMM / breadth. The 2022-23 problem on last-15m H2 is failed resist-chases, not a missing bond/dollar bit. 15m L3 wait-12 already works in that regime as its **own** book. Do not splice it onto last-15m via a macro or rolling-WR gate. Occupancy not re-walked. Number to beat remains last-15m **n=2297 E +0.27 PF 1.09**.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_regime_switch.py
```

CSV: `reports/ascending_channels/2026-09-13/last_15m_regime_switch_summary.csv`.
