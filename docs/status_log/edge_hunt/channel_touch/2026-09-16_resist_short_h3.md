# Channel-touch short H3+ resistance fade (2026-09-16)

Hypothesis: after a causal H2 arm, **short** subsequent from-below resistance tags (H3, H4, ...) and cover at support / ATR trail — classical channel fade, opposite of the long L3 / H2-resist-break books.

Harness: `utils/research/channel_touch_resist_short.py`, CLI `scripts/research/backtest_channel_touch_resist_short.py`. Detector rails freeze at H2. Fill = resist − slip (never above rail). Cancel on close through support or close above resist before the tag. Require leave-rail (~0.2 width) after H2 / between tags. ATR k=2 (1.5%–6%) hard stop above + 10% trough trail; default also covers at support.

## Daily 300 ADV (ALPACA+IB prefix, windowed 504/252, fric 0.25)

Wall-clock ~9s (cache hit).

| book | n | E% | PF | geo-PF | notes |
|------|---|-----|-----|--------|-------|
| raw | 434 | −0.54 | 0.81 | 0.83 | loses |
| span≤365 unique | **250** | **+0.06** | **1.02** | **1.06** | flat |
| H3 only | 190 | −0.35 | 0.88 | 0.92 | loses |
| H4+ (post-filter) | 60 | +1.34 | 1.51 | 1.51 | thin; 2024-26 weak |
| entry_touch=4 unique | 69 | +0.60 | 1.21 | 1.21 | thin; not robust |
| ATR+trail only | 250 | −0.04 | 0.99 | 1.02 | support cover does not save |

**No promote on daily.** Fighting an ascending channel on 1d tags is flat after friction; H3 is negative; H4+ lift is small-n.

## 15m 300 ADV (IB, wait-12, span≤10, fric 0.10)

Wall-clock ~249s (load 142s + scan 101s + rescan 100s).

| book | n | E% | PF | geo-PF |
|------|---|-----|-----|--------|
| unique span≤10 | **7004** | **+0.21** | **1.47** | **1.45** |
| H3 only | 6399 | +0.20 | 1.45 | 1.42 |
| H4+ | 605 | +0.30 | 1.68 | 1.70 |
| RS top1/day | 1536 | +0.32 | 1.75 | 1.67 |
| ATR+trail only | 6952 | +0.21 | 1.44 | 1.42 |

All YEAR_BUCKETS positive on unique span10. Median still **negative** (−0.38 gross / −0.48 net) — fat-tailed MR. Hard-stops ~43%. Max concurrent (calendar) **12**.

### Calendar 2022 vs 2023 (not the pooled 2022-23 bucket)

| book | 2022 n / E / PF | 2023 n / E / PF |
|------|-----------------|-----------------|
| 1d unique (fric 0.25) | 16 / **+1.05** / 1.33 | 33 / **+0.90** / 1.34 |
| 1d entry_touch=4 | 5 / **−5.41** / 0.00 (0 wins) | 13 / +3.53 / 2.75 |
| **15m unique (fric 0.10)** | **1031 / +0.26 / 1.57** | **900 / +0.20 / 1.45** |

15m shorts are green in both years (contrast last-15m long H2 2022 n=117 E −2.37 PF 0.41). Daily 2022 is too thin to trust; H4-only entry is wiped in 2022.

Stress on unique span10:

| stress | E% | PF |
|--------|-----|-----|
| fric 0.10 | +0.21 | 1.47 |
| fric 0.25 | +0.06 | 1.11 |
| fric 0.50 | −0.19 | 0.73 |
| drop-top-10 @ fric0.10 | +0.20 | 1.44 |
| drop-top-50 @ fric0.10 | +0.16 | 1.37 |

Survives drop-top-50 (not a single fat-winner book). **Fragile to round-trip friction ≥0.25%.**

Vs 15m long L3 wait-12 signal-close keeper (**n=1744 E +0.13 PF 1.22**): short fade has higher pooled PF/E and more trades on the same 300 panel, but different fill/exit model and fric0.10 default — do not treat as a head-to-head promote.

## Verdict

- **Daily: no promote.**
- **15m: research-interesting, not a hard promote.** Edge is small per trade, median negative, dies under 0.25% friction. Worth a full-IB-universe pass and realistic cover fills before any live short sleeve. Do not wire `/hot` or nightly.

Artifacts: `reports/ascending_channels/2026-09-16/channel_touch_resist_short_*_20260916_211013.csv` (daily), `*_211434.csv` (15m).
