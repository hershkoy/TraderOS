# 15m full-universe H2 resist-break — 2026-08-29

Full IB 15m panel (not the 300 ADV-ranked sample). Same stack as the 300-name A/B:
`--preset 15m`, wait-12, span<=10d on breakouts, friction 0.10, prior-bar features.
Detector v1 unchanged. **Research only** — do not stream IB 15m; nightly stays daily EOD.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py --preset 15m --all-symbols --workers 4 --load-workers 8
```

## Timing

Wall-clock **3064s** (~51 min). Cold TimescaleDB load is the bottleneck.

| Step | Time |
|------|------|
| `list_symbols_fast(IB, 15m)` | 1478 names + SPY |
| Load | **1857s**, 1177/1177 from DB (cache hits=0, chunk=10, workers=8) |
| Scan | **1066s**, n=456328 raw fills, workers=4 |
| Write CSVs | remainder |

~301 names were already parquet-cached from the 300-name A/B; the other ~1177 were a cold DB pull.

## Books (friction 0.10)

| Book | n | E | PF | WR | notes |
|------|---|---|----|----|--------|
| L3 quality (no RS) | 129115 | +0.15 | 1.27 | 32.1 | wait-12 in-channel |
| Resist-break, no span, no RS | 185048 | +0.32 | 1.58 | 36.7 | all year buckets + |
| Resist-break, span<=10d, no RS | 67416 | +0.30 | 1.55 | 36.5 | all year buckets + |
| **Unique symbol/day** | **65561** | **+0.30** | **1.55** | 36.5 | live-style cap (1855 same-symbol extras) |
| **Resist-break span10 + RS top1** | **1766** | **+0.30** | **1.48** | 38.6 | 1-name/day cap, not a quality filter |
| L3 quality + RS top1 | 1775 | +0.34 | 1.61 | 40.6 | **beats H2 after RS** |
| L3 + span10 break + RS top1 | 1776 | +0.35 | 1.58 | 39.4 | combo does not beat L3 RS on PF |

Year buckets for span10 + RS: 2018-19 E +0.18 PF 1.33; 2020-21 +0.45 / 1.77; 2022-23 +0.18 / 1.28; 2024-26 +0.31 / 1.47.

## vs 300-name sample

| Book | 300 ADV IB | Full IB 15m (~1177 loaded) |
|------|------------|----------------------------|
| L3 wait-12 + RS | n=1751 E +0.31 PF **1.57** | n=1775 E +0.34 PF **1.61** |
| H2 span10 + RS | n=1664 E +0.46 PF **1.87** | n=1766 E +0.30 PF **1.48** |

RS top1/day keeps n ~1760 either way — extra names change **which** ticker wins the rank, not how many trades you take. The 300 ADV sample overstated H2 after RS. On the full IB 15m set, **wait-12 L3 + RS still wins** E/PF.

Unranked H2 still has a raw lift (span10 n=67416 E +0.30 PF 1.55 vs L3 n=129115 E +0.15 PF 1.27), but that is not a live book.

## Gate

**Do not promote 15m H2 resist-break over 15m L3 wait-12** on the full IB 15m universe after RS top1. Unique-symbol/day (the live rule) is **n=65561 E +0.30 PF 1.55** — almost the unranked book. Do not copy daily span 365 or beyond-width 0.25 onto 15m. Nightly stays **daily** H2 resist-break.

## Artifacts

- Raw breakouts: `reports/ascending_channels/channel_touch_15m_full_h2_resist_break_20260829_224432.csv`
- Span<=10 sleeve: `reports/ascending_channels/channel_touch_15m_full_h2_break_span10.csv` (same bytes as `..._20260829_224432.csv`)
- HTML (default max/day=1, IB daily SPY overlay 2018-11-05): `reports/ascending_channels/current_best/15m_full_h2_resist_break.html`
- 300-name A/B remains `current_best/15m_h2_resist_break.html`
