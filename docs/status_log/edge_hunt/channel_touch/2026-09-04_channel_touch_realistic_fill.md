# Realistic-fill full-universe rescan (1d + 15m)

Replaced `reports/ascending_channels/current_best/` after hooking `utils/research/realistic_purchaser.py` (`--realistic-fill`) into `backtest_channel_touch_trades.py` / `backtest_channel_touch_h2_break.py` and rerunning the **causal** books on the same date windows as 2026-09-03.

`find_channels` v1 is unchanged. Nightly trigger is still daily H2 resist-break with **close** fills (`scripts/scanners/channel_touch_nightly.py`) — not this purchaser. Do **not** promote 15m H2 or H5 on these fills.

## Fill rules

- **15m:** signal = completed bar T close; buy the next same-session 15m **mid**. Last RTH start 15:45 ET: confirm at 16:00 has **no** window (do not roll overnight). Cancel if `(mid−low)/mid > 0.5%` (`--max-low-to-mid-pct`).
- **1d:** daily signal price X; first RTH 15m whose range contains X; next 15m mid; fill `(X + next_mid) / 2`. Same wild/EOD rules. Skip names with no IB 15m (738 of 2216).

## Daily (`1d`) — 2018-11-01 → 2026-08-27, 2216 ALPACA 1d, IB prefix, window 504/252

Wall-clock: 1d cache 24.7s + IB 15m load **1139s** (1478/2216) + H2 scan **1988s**, elapsed **3152s**. Friction 0.25%.

| Book | Realistic fill | Causal optimistic 2026-09-03 |
|------|----------------|------------------------------|
| **Live H2 span365 unique-symbol** | **n=713 E +1.78 PF 1.72** WR 35.3 (all years +) | n=3364 E +2.40 PF 2.02 |
| Live H2 optional RS top1 | n=419 E +2.65 PF 2.08 | n=1126 E +2.94 PF 2.24 |
| Retired L3 quality (in-channel, span365, beyond 0.25, RSI 50) unique | n=28 E +1.68 PF 1.66 (raw pre-filter n=126) | n=3407 E +1.68 PF 1.61 |
| Retired L3 RS top1 | n=27 E +1.83 PF 1.71 | n=1134 E +2.42 PF 1.92 |

HTML: `current_best/1d_channel_touch.html` (same bytes as `1d_h2_resist_break.html`). Trades: `reports/ascending_channels/channel_touch_h2_break_span365.csv`.

## 15-minute — 2018-11-01 → 2025-12-02, friction 0.10%

Full IB 15m: 1478 names, cache load 44s, H2 scan **1025s**, elapsed **1122s**.

| Book | Realistic | Causal optimistic 2026-09-03 |
|------|-----------|------------------------------|
| Full H2 span≤10 RS top1 | **n=1755 E −0.07 PF 0.90** | n=1770 E +0.21 PF 1.33 |
| Full H2 span≤10 unique | n=67020 E −0.02 PF 0.96 | n=87073 E +0.25 PF 1.44 |
| 300-name H2 span≤10 RS top1 | n=1658 E −0.05 PF 0.92 | n=1699 E +0.32 PF 1.57 |
| 300-name L3 wait-12 | n=1746 E +0.09 PF 1.16 | n=1756 E +0.30 PF 1.55 |
| **H5 + overshoot p80 + vol≥2** (unique) | **n=7209 WR 24.8 E ~0 PF 1.01** | n=8212 WR 47.4 E +0.92 PF 3.57 |

H5 remains **research only — not nightly**. The tight H5 stack was fill-bar clip, not a live edge. Full-panel 15m H2 still loses to wait-12 L3; after realistic fill H2 is below 1.0 PF. Do not promote 15m H2.

## `current_best/` replace (2026-09-04)

Deleted the old names first, then copied timestamped realistic HTML in (do not `copy /Y` onto a hard link).

| `current_best` name | Realistic HTML stamp |
|---------------------|----------------------|
| `1d_channel_touch.html` / `1d_h2_resist_break.html` | `...h2_break_span365_realistic_20260904_030346.html` |
| `1d_l3_touch.html` | `...l3_touch_realistic_20260904_045029.html` |
| `1d_keeper_plus_h2_resist_break.html` | `...keeper_plus_h2_realistic_20260904_045030.html` |
| `15m_channel_touch.html` | `...15m_l3_wait12_realistic_20260904_033433.html` |
| `15m_h2_resist_break.html` | `...15m_h2_span10_realistic_20260904_033432.html` |
| `15m_full_h2_resist_break.html` | `...15m_full_h2_realistic_20260904_032018.html` |
| `15m_h5_overshoot_vol.html` | `...15m_h5_over_p80_vol2_realistic_20260904_032039.html` |

Write-up: `reports/ascending_channels/current_best/README.md`.
