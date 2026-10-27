# 15m L3 wait-12 occupancy vs SPY (2026-10-10)

## Why

Pooled L3 wait-12 signal-close is **n=1744 E +0.13 PF 1.22** (fric0.10) — research keeper, not a hard promote. Edge-hunt promote needs fixed-slot occupancy equity vs SPY (Sharpe>1, Sharpe≥SPY, MDD better than SPY; 2022 waives Sharpe>1).

## Book

`reports/ascending_channels/channel_touch_15m_l3_wait12.csv` (current_best 15m signal-close + touch-0).

Unconstrained concurrent (business-day open count): **max 8 / p95 4 / median 2 / mean 2.07**.

## Method

- Greedy FCFS `apply_max_open` (tie-break fifo/symbol)
- Equal-slot mark-at-cost equity (`wheel_slot_equity` on `gain_pct_net`)
- SPY buy-hold on ALPACA+IB prefix daily
- Caps: 1, 3, 5, 7, 8, 15
- Runner: `scripts/research/overlay_l3_wait12_occupancy.py`

Wall-clock ~24s (SPY load dominates).

## Results (FULL)

| max_open | n_kept | pooled E / PF | Sharpe | CAGR | MDD | inv% | vs SPY (0.88 / 14.9% / −25.4%) | Gate |
|----------|--------|---------------|--------|------|-----|------|--------------------------------|------|
| 1 | 850 | +0.12 / 1.19 | 0.21 | 5.2% | −32.4% | 49 | worse Sharpe + MDD | **FAIL** |
| **3** | **1655** | **+0.15 / 1.25** | **1.01** | **12.0%** | **−10.2%** | 33 | ≥SPY Sharpe, better MDD | **PASS** |
| 5 | 1738 | +0.13 / 1.22 | 0.96 | 6.8% | −6.2% | 21 | ≥SPY Sharpe, better MDD | **FAIL** (Sharpe≤1) |
| 7 | 1743 | +0.13 / 1.22 | 0.96 | 4.9% | −4.4% | 15 | ≥SPY Sharpe, better MDD | **FAIL** (Sharpe≤1) |
| 8 | 1744 | +0.13 / 1.22 | 0.96 | 4.2% | −3.9% | 13 | ≥SPY Sharpe, better MDD | **FAIL** (Sharpe≤1) |
| 15 | 1744 | +0.13 / 1.22 | 0.96 | 2.3% | −2.1% | 7 | ≥SPY Sharpe, better MDD | **FAIL** (Sharpe≤1) |

## Subwindows at occ3 (only FULL passer)

| window | n | Sharpe | CAGR | MDD | Gate |
|--------|---|--------|------|-----|------|
| 2022 | 238 | −0.17 | −1.6% | −8.5% | **PASS** (2022 waiver; beats SPY −0.83 / −25%) |
| 2023 | 237 | 1.37 | 18.7% | −9.4% | **FAIL** (Sharpe&lt;SPY 1.92) |
| 2022-23 | 475 | 0.66 | 7.7% | −9.4% | **FAIL** (Sharpe≤1) |
| 2024-26 | 453 | 1.03 | 12.3% | −10.2% | **FAIL** (Sharpe&lt;SPY 1.30) |

## Verdict

**No hard promote.** Occ3 FULL clears Phase-5 once, but:

1. Cap-sensitive — occ1 fails hard; occ5+ fall just under Sharpe>1 (~0.96) while staying better MDD than SPY.
2. Subwindows fail the same gate (2023 / 2024-26 lose to SPY Sharpe; 2022-23 Sharpe&lt;1).
3. Do not cap-fish 2/4/6 around the occ3 spike.

L3 wait-12 remains a **hostile-regime research sleeve** (2022 equity beats SPY under the waiver) with shallow concurrent demand (median 2, max 8) — not a cleared equal-dollar book for live capital.

## Artifacts

- `reports/ascending_channels/2026-10-10/l3_wait12_occupancy_summary_20261010_133915.csv`
- `reports/ascending_channels/2026-10-10/l3_wait12_occupancy_20261010_133915.json`
