# Last-15m wheel — liquid-name / tight-spread filter

**Date:** 2026-10-09  
**Source:** matched mid vs ask_bid fills n=340 (same contracts). No re-claimcheck; post-filter on `adv_20` + put half-spread at entry (causal: known before sell).

## Verdict

**Soft-interesting, still no promote.** Superseded on n by [2026-10-10 full-pool](2026-10-10_last15m_wheel_fullpool.md): liquid ADV≥$25M & half≤15% **n=236**, ask/bid **occ8 PASS** (Sharpe 1.16) — **2022 still never Sharpe>1**.

## Full-panel slices

| Filter | n | mid E / PF | ask/bid E / PF | stock E | mid−ab ΔE | med half% of mid |
|--------|--:|----------:|---------------:|--------:|----------:|-----------------:|
| ALL (ADV≥$2M) | 340 | +1.29 / 1.96 | +0.15 / 1.08 | −0.82 | 1.14 | 13.6 |
| ADV≥$10M | 200 | +0.73 / 1.49 | +0.20 / 1.12 | −0.17 | 0.52 | 8.2 |
| ADV≥$25M | 153 | +0.90 / 1.72 | +0.51 / 1.37 | −0.32 | 0.39 | 7.1 |
| ADV≥$50M | 102 | +0.38 / 1.27 | +0.06 / 1.04 | −0.68 | 0.32 | 6.2 |
| half≤15% of mid | 183 | +1.00 / 1.69 | +0.68 / 1.43 | −0.68 | 0.32 | 6.5 |
| **ADV≥$25M & half≤15%** | **117** | **+1.12 / 2.01** | **+0.91 / 1.76** | −0.00 | **0.22** | 5.2 |

## Year splits (binding)

| Filter | 2022 ask/bid E/PF (n) | 2022–23 ask/bid E/PF (n) |
|--------|----------------------:|-------------------------:|
| ALL | −0.79 / 0.75 (72) | −0.04 / 0.98 (235) |
| ADV≥$10M | −1.18 / 0.64 (49) | −0.07 / 0.96 (148) |
| ADV≥$25M | −0.66 / 0.75 (43) | +0.22 / 1.14 (116) |
| half≤15% | **+0.44 / 1.20 (38)** | **+0.46 / 1.28 (136)** |
| ADV≥$25M & half≤15% | **+0.04 / 1.02 (30)** | **+0.64 / 1.48 (91)** |

## Read

1. **ADV alone is the wrong lever.** Higher ADV shrinks half-spreads (med 13.6% → 7.1% at $25M) and ΔE (1.14 → 0.39), but mid E falls and 2022 ask/bid stays red. The mid edge was partly in thinner names.
2. **Half-spread at entry is the right lever.** Rejecting puts where `(mid−bid)/mid > 15%` is live-executable and flips ask/bid 2022 green without needing mega-cap ADV.
3. **Combo** keeps mid strong (+1.12 / 2.01) while ask/bid almost matches (+0.91 / 1.76, Δ only 0.22pp). 2022 ask/bid barely clears (n=30).
4. Still beat same-row stock on every liquid slice; that claim survives the filter.
5. **Do not promote** without occupancy vs SPY on the filtered book and a larger 2022 n.

## Artifacts

- `reports/ascending_channels/2026-10-09/last_15m_wheel_liquidity_slices.json`
- Script: `scripts/research/_wheel_liquidity_slice.py`
- Prior mark compare: [mid_vs_askbid](2026-10-09_last15m_wheel_mid_vs_askbid.md)

## Occupancy vs SPY (liquid presets) — 2026-10-09 evening

FCFS unique-symbol + equal-slot mark-at-cost. Runner: `overlay_last_15m_wheel_occupancy.py --liq-presets`. SPY FULL Sharpe **0.63** CAGR 10.9% MDD −24.3%.

| Book | max_open | kept | FULL Sharpe / CAGR / MDD | vs SPY | 2022 Sharpe |
|------|---------:|-----:|-------------------------:|--------|------------:|
| half15 mid | 1 | 32 | 0.69 / 9.0% / −15.0% | FAIL ≤1 | 0.32 |
| half15 mid | 8 | 132 | 0.73 / 3.0% / −7.3% | FAIL ≤1 | 0.21 |
| half15 mid | 15 | 166 | 0.93 / 2.3% / −4.3% | FAIL ≤1 | 0.40 |
| half15 ask/bid | 1–15 | 32–166 | 0.42–0.60 | FAIL | ≤0.25 |
| **adv25_half15 mid** | **1** | **26** | **1.23 / 16.8% / −5.2%** | **PASS** | 0.90 |
| adv25_half15 mid | 4 | 69 | 0.70 / 4.1% / −10.0% | FAIL | −0.38 |
| **adv25_half15 mid** | **8** | **98** | **1.08 / 3.5% / −5.0%** | **PASS** | 0.25 |
| adv25_half15 mid | 15 | 112 | 0.94 / 1.8% / −3.2% | FAIL ≤1 | 0.30 |
| **adv25_half15 ask/bid** | **1** | **26** | **1.10 / 14.9% / −6.2%** | **PASS** | 0.76 |
| adv25_half15 ask/bid | 4 | 69 | 0.54 / 3.1% / −10.4% | FAIL | −0.49 |
| adv25_half15 ask/bid | 8 | 98 | 0.90 / 2.9% / −5.2% | FAIL ≤1 | 0.08 |
| adv25_half15 ask/bid | 15 | 112 | 0.74 / 1.4% / −3.4% | FAIL | 0.11 |

Read:

- **half≤15% alone fails FULL occupancy** at every cap (mid best Sharpe 0.93) — pooled E did not survive FCFS slotting.
- **ADV≥$25M & half≤15%** is the only liquid book with Phase-5 FULL PASSes: mid occ1/occ8, and **ask/bid occ1** (first natural-fill PASS on this sleeve).
- Those PASSes are **thin** (occ1 kept n=26) and **2022 never clears Sharpe>1**. Multi-slot ask/bid fails (occ8 0.90).
- Still **no promote** — liquid filter improves natural marks but does not clear a robust occupancy gate.

Artifacts: `reports/ascending_channels/2026-10-09/last_15m_wheel_occupancy_liq_20261009_182908.json` / `.csv`.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_wheel_occupancy.py --liq-presets
```
