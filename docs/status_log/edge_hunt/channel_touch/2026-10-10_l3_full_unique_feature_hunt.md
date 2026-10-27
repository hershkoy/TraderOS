# Full-IB unique L3 wait-12 — feature hunt (2026-10-10)

## Why

Full IB unique-symbol/day L3 wait-12 is flat (**n=46885 E −0.04 PF 0.93**). Mine prior-bar entry features already on the CSV, then join SEC fundamentals asof `filed`, with year geo-PF + winner-$ + ADV-overlap gates.

## Book

- Unique CSV: `reports/ascending_channels/2026-10-10/channel_touch_15m_l3_wait12_full_unique.csv`
- Keeper compare: `reports/ascending_channels/channel_touch_15m_l3_wait12.csv` (268 symbols in trade list / 300 ADV panel)
- Baseline: n=46885 E −0.0417 PF 0.933 WR 26.5% geo-PF 0.941; keeper_trade% ≈20.7%

## Phase 1 — Univariate

```bat
python scripts\research\analyze_channel_touch_entry_features.py --trades reports\ascending_channels\2026-10-10\channel_touch_15m_l3_wait12_full_unique.csv --friction-pct 0.10
```

Wall ~4s. Strongest Spearman vs net: `atr_pct` −0.28, `channel_width_pct` −0.23, `room_to_resist_pct` −0.21, `range_pct` −0.20; `adv_20` +0.06.

Quintile notes (hypothesis only):
- Highest ATR / range / width / room quintiles are the worst PF (~0.80–0.86)
- Top ADV quintile only positive bucket (E +0.025 PF 1.04)
- Skip calendar; treat ADV floor as liquidity diagnostic

Artifacts: `channel_touch_entry_features_{buckets,spearman,corr}_20261010_183455.csv`

## Phase 2 — Price overlays

Harness: `scripts/research/overlay_l3_full_unique_feature_filters.py`  
Summary: `reports/ascending_channels/2026-10-10/l3_full_unique_feat_price_summary_20261010_183708.csv`  
Wall ~0.6s (CSV-only).

| variant | n | E | PF | geo_PF | win$keep | keeper_trade% | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 46885 | −0.04 | 0.93 | 0.94 | 1.00 | 20.7 | — |
| atr≤0.733 | 37535 | −0.02 | 0.97 | 0.98 | 0.79 | 22.6 | no_promote |
| range≤0.489 | 28124 | −0.00 | 0.99 | 0.99 | 0.58 | 23.2 | no_promote |
| atr+range | 33907 | −0.01 | 0.98 | 0.99 | 0.71 | 23.0 | no_promote |
| adv≥4.2e6 | 9377 | **+0.02** | **1.04** | **1.04** | 0.21 | **73.8** | **adv_recovery** |
| adv4.2e6+atr0.733 | 8097 | **+0.05** | **1.09** | **1.09** | 0.18 | **76.4** | **adv_recovery** |

All pure price caps fail `win$keep ≥ 0.90`. Best geo lift is ADV top-quintile (and ADV+ATR) — collapses toward keeper names, not a new full-univ edge. Still far below 300 ADV keeper (E +0.13 PF 1.22).

## Phase 3 — Fundamentals asof filed

Helper: `utils/research/channel_touch_fundamentals.py` (`filed <= buy_date`, sane `period_end`).  
Tests: `tests/unit/test_channel_touch_fundamentals.py` (2 passed).

Join coverage on trades: **has_fund=27553/46885 (58.8%)**, four-filter screen pass **n=795**.  
Summary: `l3_full_unique_feat_fund_summary_20261010_183727.csv`  
Wall ~14s (DB load + join).

| variant | n | E | PF | geo_PF | win$keep | keeper_trade% | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| has_fund | 27553 | −0.04 | 0.94 | 0.95 | 0.60 | 23.3 | no_promote |
| fund_screen (4/4) | 795 | −0.01 | 0.99 | 1.00 | 0.02 | 53.6 | no_promote |
| fund_npass≥2 | 10800 | +0.03 | 1.05 | 1.07 | 0.24 | 31.8 | no_promote |
| fund_npass≥3 | 4978 | +0.03 | 1.05 | 1.10 | 0.11 | 37.2 | no_promote |
| fcf_roa≥0.12 | 8627 | +0.01 | 1.02 | 1.04 | 0.19 | 33.3 | no_promote |
| adv4.2e6+fund_screen | 331 | +0.06 | 1.12 | 1.13 | 0.01 | **92.5** | adv_recovery |

Fund filters can lift pooled E/PF/geo but fail winner-$ (and year-all-positive). Full four-filter screen is too thin (n=795) and still red/flat.

## Phase 4 — Walk-forward ML

**Skipped.** No simple filter cleared research_pass (geo lift + win$keep≥0.90 + years+ without ADV collapse). Prior 15m/1d logistic/ridge already no-promote.

## Verdict

**No promote. ADV recovery only.**

- Full-univ unique stays a dead book under price and SEC fund overlays.
- Liquidity (top ADV quintile / ADV+ATR) recovers toward the 300 ADV keeper but clips winner-$ and is not a new detector edge.
- Keep `current_best/15m_channel_touch.html` on the 300 ADV L3 wait-12 book.

## Code

- `scripts/research/overlay_l3_full_unique_feature_filters.py`
- `utils/research/channel_touch_fundamentals.py`
- `tests/unit/test_channel_touch_fundamentals.py`
