# Synthetic splice: last-15m 1d H2, 2022 = 15m L3 wait-12 (2026-09-12)

Oracle calendar splice. **No switch signal.** Occupancy not re-walked. **Not a strategy. No promote.**

Replace calendar-2022 trades on the last-15m 1d H2 book (unique last-RTH 15m mid if open above rail, 15m N+1 mid sells, n=2297) with the 15m L3 wait-12 signal-close book for that year (n=243, E +0.05 PF 1.09). Other years stay last-15m H2. Same splice also run on `current_best` hot-cross for comparison.

Gross `gain_pct` (HTML default friction 0.25). L3 keeper was scored at 0.10.

## Last-15m 1d H2 splice

| Book | n | E | PF | geo year PF | 2022 n / E / PF | 2022-23 E / PF |
|------|---|---|----|-------------|-----------------|----------------|
| Last-15m 1d H2 | 2297 | +0.27 | 1.086 | 1.095 | 117 / **−2.37 / 0.41** | −0.90 / 0.73 |
| 15m L3 wait-12 (all years) | 1744 | +0.23 | 1.434 | 1.436 | 243 / **+0.05 / 1.09** | +0.18 / 1.32 |
| **Splice 2022=L3** | **2423** | **+0.37** | **1.132** | **1.158** | 243 / +0.05 / 1.09 | **−0.16 / 0.92** |

2018-19 / 2020-21 / 2024-26 unchanged. The 2022-23 YEAR_BUCKET still mixes **2022 L3** with **2023 last-15m H2** (n=300 E −0.33), so that bucket stays slightly red. Replacing 2022 removes the disaster year; it does not make 1d resist-chase work in chop.

n: 2297 − 117 + 243 = 2423.

## Hot-cross 1d splice (same oracle)

| Book | n | E | PF | 2022 E / PF | 2022-23 E / PF |
|------|---|---|----|-------------|----------------|
| Hot-cross lerp85 | 4079 | +0.06 | 1.018 | 261 / −2.47 / 0.37 | −0.65 / 0.81 |
| Splice 2022=L3 | 4061 | +0.22 | 1.073 | 243 / +0.05 / 1.09 | **+0.18 / 1.08** |

Gross. Hot-cross HTML is friction 0.25 → E −0.19. Do not treat the spliced +0.22 as beating the live 1d gate.

## Caveats

- No live rule for “now use L3.” Calendar year is peeking.
- Occupancy not re-walked across 1d vs 15m clocks.
- 15m L3 avg win ~2.5% vs last-15m ~9%; equal-trade E mixes scales.
- 2023 last-15m H2 is still red on the last-15m splice.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\splice_2022_l3_into_1d.py
python scripts\research\generate_channel_touch_tv_report.py --trades reports\ascending_channels\2026-09-12\synthetic_1d_last15m_2022_l3.csv --friction-pct 0.25 --tag synthetic_2022_l3 --summary reports\ascending_channels\2026-09-12\synthetic_1d_last15m_2022_l3_summary.txt --comparison-json reports\ascending_channels\2026-09-12\synthetic_1d_last15m_2022_l3_compare.json --title "Synthetic last-15m 1d H2, 2022 = 15m L3 wait-12"
```

HTML: `reports/ascending_channels/2026-09-12/channel_touch_tv_report_interactive_fric0.25_synthetic_2022_l3_20260912_020936.html`
CSV: `synthetic_1d_last15m_2022_l3.csv` (and `synthetic_1d_hot_cross_2022_l3.csv`).
