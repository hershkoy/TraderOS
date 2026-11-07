# Listed buy-write mid±slip occupancy + MTM (2026-11-04)

**Verdict: still no promote.** Soft mid mark-at-cost can scrape a Phase-5 PASS at `max_open=15`, but **$0.05 / $0.10 credit slip kills every FULL gate**, and **call EOD MTM** on the intermediate slip book fails FULL (Sharpe 0.28 vs SPY 0.64).

Source: Theta VALUE listed ATM buy-write mid (`2026-11-03_cc_earnings_listed`, n=802 fills).

## Pooled (no occupancy)

| Slip | n | E | PF | 2022 E / PF |
|-----:|--:|--:|---:|------------:|
| $0.00 | 802 | +1.10 | 1.48 | +0.47 / 1.15 |
| $0.05 | 802 | +0.97 | 1.41 | +0.33 / 1.10 |
| $0.10 | 802 | +0.84 | 1.35 | +0.18 / 1.06 |

## Occupancy mark-at-cost vs SPY (FULL)

| Book | max_open | kept | Sharpe | SPY | MDD | Gate |
|------|---------:|-----:|-------:|----:|----:|------|
| mid | 1 | 53 | 0.17 | 0.64 | −49% | FAIL |
| mid | 4 | 195 | 0.76 | 0.64 | −16% | FAIL Sharpe≤1 |
| mid | 8 | 342 | 0.48 | 0.64 | −13% | FAIL |
| **mid** | **15** | **522** | **1.00** | **0.64** | **−7%** | **PASS** |
| mid_slip $0.05 | 1–15 | 53–522 | 0.13–0.86 | 0.64 | | **all FAIL** |
| mid_slip $0.10 | 1–15 | 53–522 | 0.10–0.73 | 0.64 | | **all FAIL** |

2022 under waiver: several slip occ books PASS vs SPY that year; not enough without FULL.

## MTM (binding)

`mid_slip $0.05`, FCFS `max_open=8` (kept 342), Theta EOD call marks (Value `history/eod`):

| Model | FULL Sharpe / CAGR / MDD | vs SPY | Gate |
|-------|-------------------------:|--------|------|
| mark-at-cost | 0.35 / 3.3% / −12.9% | FAIL | Sharpe≤1, <SPY |
| **call MTM** | **0.28 / 3.4% / −15.3%** | FAIL | Sharpe≤1, <SPY |

MTM **worsens Sharpe and MDD** vs mark-at-cost on the same kept set — same pattern as the wheel MTM fail.

## Read

1. Treating full ask/bid as too harsh is fair; **$0.05–$0.10 mid slip** is the intermediate stress you wanted.
2. That intermediate stress **removes** the only soft PASS (raw mid occ15).
3. Path MTM does not rescue it.
4. Do not promote / do not wire `/hot`. Value sub is optional now.

## Artifacts

- `reports/ascending_channels/2026-11-03_cc_earnings_listed/bw_listed_slip_pooled.csv`
- `bw_listed_occupancy_gates.csv` / `bw_listed_occupancy.json`
- Runner: `scripts/research/overlay_last_15m_buy_write_occupancy.py --with-mtm`
