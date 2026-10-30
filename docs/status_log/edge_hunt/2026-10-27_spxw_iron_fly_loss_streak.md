# SPXW 0DTE iron fly — loss-streak / VIX overlays on full window (2026-10-27)

**Verdict: no promote / no soft-promote.** Locked consecutive-loss sit-out and tighter VIX do **not** clear Phase-5 MDD vs SPY on the expand book. Same failure class as last-15m loss-streak ([channel_touch note](channel_touch/2026-09-16_last_15m_loss_streak.md)). Do not L/S/P-fish; do not wire.

Parent expand: [2026-10-27_spxw_0dte_iron_fly_from_20220511.md](2026-10-27_spxw_0dte_iron_fly_from_20220511.md).

## Setup

- Book: natural settle `2026-10-27_from_20220511/trades.csv` **n=611** (2022-08-11→2026-10-26).
- Reuse `apply_loss_streak_mask` (`utils/research/last_15m_loss_streak.py`); skipped signals do not invent paper P&L.
- Runner: `scripts/research/overlay_spxw_iron_fly_loss_streak.py`
- Artifacts: `reports/research/spxw_0dte_iron_fly/2026-10-27_from_20220511/loss_streak/`
- HTML (L5/S3/P2): `.../loss_streak/spxw_iron_fly_L5_S3_P2_report.html` (`scripts/research/generate_spxw_iron_fly_loss_streak_report.py`)

## 1. Streak diagnostic

| bucket | n | max consec loss | mean run | n runs ≥5 | % trades in run≥5 |
|--------|--:|----------------:|---------:|----------:|------------------:|
| 2022-2023 | 160 | 6 | 2.06 | 3 | **10.0%** |
| 2024-2026 | 451 | **7** | 1.61 | 3 | 3.8% |
| FULL | 611 | 7 | 1.72 | 6 | 5.4% |

`longer_tailed` vs 2024-2026: **False** (hostile max 6 < 7). Hostile has a higher share of trades inside ≥5 runs, but the absolute streak sensor is not unique — same lesson as last-15m.

## 2. Locked rules (pre-registered)

| Rule | n | E | PF | 2022-23 E | Phase-5 | MDD |
|------|--:|--:|---:|----------:|---------|----:|
| Baseline | 611 | +88.8 | 1.43 | +47.4 | FAIL MDD | **26.7%** |
| count L5 S3 P2 | 593 | +91.3 | 1.45 | +49.1 | FAIL MDD | 20.9% |
| count L5 S3 P3 | 593 | +86.9 | 1.42 | +40.6 | FAIL MDD | 25.8% |
| dollar ≤−$1500 S3 P2 | 542 | +89.2 | 1.43 | +48.6 | FAIL MDD | 24.3% |
| VIX open &lt;18 | 483 | +100.0 | 1.50 | +32.7 | FAIL MDD | 20.9% |
| VIX open &lt;16 | 297 | +68.2 | 1.33 | +16.8 | FAIL Sharpe+MDD | 21.0% |

L5/S3/P2 trims MDD (26.7→20.9) and slightly lifts 2022-23 E, but still ≥ SPY MDD (~19%). Dollar and VIX tighten do not clear the gate; VIX&lt;16 blunts expectancy and fails Sharpe vs SPY too.

## 3. Exploratory L∈{3,4,5} (report-only — no promote)

| Rule | n | E | 2022-23 E | Phase-5 | MDD |
|------|--:|--:|----------:|---------|----:|
| L3 S3 P2 | 530 | +100.6 | +73.7 | FAIL MDD≈SPY | 19.0% |
| L4 S3 P2 | 563 | +94.9 | +62.5 | **PASS** | 15.9% |
| L5 S3 P2 | 593 | +91.3 | +49.1 | FAIL | 20.9% |

L4 is the only PASS, but it is **post-hoc L fishing** on the full sample — same reject as last-15m exploratory L∈{4,5,6}. Do not soft-promote L4; do not grid more L/S/P.

## Promote / wire

| Gate | Result |
|------|--------|
| Hostile streak longer-tail diagnostic | Fail (max vs 2024-26) |
| Locked L5 / dollar / VIX Phase-5 | All FAIL |
| Exploratory L4 Phase-5 | PASS — **no promote** (fished L) |
| Soft-promote overlay | **No** |
| Wire | **No** |

Full-window soft-promote remains off; ~2y sample soft-promote unchanged ([path_phase5](2026-10-27_spxw_0dte_iron_fly_path_phase5.md)).

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_spxw_iron_fly_loss_streak.py
pytest tests\unit\test_overlay_spxw_iron_fly_loss_streak.py -v
```
