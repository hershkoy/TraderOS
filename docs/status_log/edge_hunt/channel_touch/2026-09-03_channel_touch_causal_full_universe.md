# Causal full-universe rescan (1d + 15m)

Replaced `reports/ascending_channels/current_best/` after running the **fixed** detector (`causal_h2` freeze-at-first-H2 + `--h2-resist-break-only` before occupancy) on the same date windows as the leaky Aug 29–31 books.

`find_channels` v1 is unchanged. Nightly trigger is still daily H2 resist-break (unique-symbol/day). Do **not** promote 15m H2 over wait-12 L3 on the full IB 15m panel.

## Why the n/E/PF moved

Leaky batch let later confirmed highs refit width/H2, and H2-only occupancy treated L3 support-tag fills as live then stripped them. Walk never queued those tags (AAPL 1d **2021-11-30 @ 163.15** was the occupancy miss). Causal freeze + drop L3 before occupancy **adds fills** the walk would take. E/PF on H2 books is a bit lower; 15m L3 barely moved; H5 tight stack is the same shape with more n.

## Daily (`1d`) — 2018-11-01 → 2026-08-27, 2216 ALPACA 1d, IB prefix, window 504/252

Wall-clock: cache load 16.5s + H2 scan **49.8s**. Friction 0.25%.

| Book | Causal | Leaky Aug-29 |
|------|--------|--------------|
| **Live H2 span365 unique-symbol** | **n=3364 E +2.40 PF 2.02** WR 41.0 (all years +) | n=2253 E +2.80 PF 2.21 |
| Live H2 optional RS top1 | n=1126 E +2.94 PF 2.24 | n=981 E +3.11 PF 2.34 |
| Retired L3 quality (in-channel, span365, beyond 0.25, RSI 50) unique | n=3407 E +1.68 PF 1.61 | (HTML was RS top1) |
| Retired L3 RS top1 | n=1134 E +2.42 PF 1.92 | n=1024 E +1.77 PF 1.62 |

AAPL 1d live book now includes 2021-11-30 (was missing from leaky `1d_channel_touch.html`).

HTML: `current_best/1d_channel_touch.html` (same bytes as `1d_h2_resist_break.html`). Trades: `reports/ascending_channels/channel_touch_h2_break_span365.csv`.

## 15-minute — 2018-11-01 → 2025-12-02, friction 0.10%

Full IB 15m: 1478 names, cache load 46s, H2 scan **1155s**, elapsed **1282s**.

| Book | Causal | Leaky Aug-29 |
|------|--------|--------------|
| Full H2 span≤10 RS top1 | **n=1770 E +0.21 PF 1.33** | n=1766 E +0.30 PF 1.48 |
| Full H2 span≤10 unique | n=87073 E +0.25 PF 1.44 | n=65561 E +0.30 PF 1.55 |
| 300-name H2 span≤10 RS top1 | n=1699 E +0.32 PF 1.57 | n=1664 E +0.46 PF 1.87 |
| 300-name L3 wait-12 | n=1756 E +0.30 PF 1.55 | n=1751 E +0.31 PF 1.57 |
| **H5 + overshoot p80 + vol≥2** (unique) | **n=8212 WR 47.4 E +0.92 PF 3.57** (drop-top-1% PF 3.17) | n=6306 WR 48.8 E +0.93 PF 3.61 |

H5 remains **research only — not nightly**. Full-panel 15m H2 still loses to wait-12 L3 on the 300-name analog; do not promote 15m H2.

`15m_full_h2_resist_break.html` SPY overlay still starts **2020-07-27** (Alpaca SPY; first trade 2018-11-05). IB-prefix SPY if regenerating so equity and SPY share the first-trade date.

## `current_best/` replace (2026-09-03)

Deleted the old hard-link names first, then copied timestamped causal HTML in (do not `copy /Y` onto a hard link — that overwrites the dated file).

| `current_best` name | Causal HTML stamp |
|---------------------|-------------------|
| `1d_channel_touch.html` / `1d_h2_resist_break.html` | `...h2_break_span365_causal_20260903_015652.html` |
| `1d_l3_touch.html` | `...l3_touch_causal_20260903_015653.html` |
| `1d_keeper_plus_h2_resist_break.html` | `...keeper_plus_h2_causal_20260903_015714.html` |
| `15m_channel_touch.html` | `...15m_l3_wait12_causal_20260903_023054.html` |
| `15m_h2_resist_break.html` | `...15m_h2_span10_causal_20260903_023029.html` |
| `15m_full_h2_resist_break.html` | `...15m_full_h2_causal_20260903_021803.html` |
| `15m_h5_overshoot_vol.html` | `...15m_h5_over_p80_vol2_causal_20260903_021944.html` |

Write-up: `reports/ascending_channels/current_best/README.md`.
