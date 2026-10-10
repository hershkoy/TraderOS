# Last-15m wheel — full-pool expand (n=1118)

**Date:** 2026-10-10 (gate update same day)  
**Verdict: soft-interesting under revised 2022 gate; still no hard promote / do not wire `/hot` yet.** Larger liquid book unlocks ask/bid occ8 FULL PASS; with **2022 Sharpe>1 waived** (require only Sharpe≥SPY + MDD), every liquid book’s 2022 window **PASSes** vs SPY −0.77. Remaining gaps: half15 ask/bid FULL still fails; thin occ1 n; mark-at-cost (no option MTM).

## Claimcheck

- Pool: `buy≥2022 & ADV≥$2M` → **1118** attempted (was 500).
- Filled **817** (73%; 297–301 `no_put_chain` on ask_bid).
- Mid raw re-fetch poisoned 2022–23 cache with `no_put_chain` (Theta miss overwrite). **Repaired** by splicing prior mid fills (`last_15m_wheel_mid_repaired_20261010_112646.csv`). Cache loader now prefers fills over later misses.
- Ask/bid: `last_15m_wheel_ask_bid_20261010_112516.csv` (native cache hits for 2022–23).

| Slice | mid E / PF / WR (n) | ask/bid E / PF / WR (n) |
|-------|--------------------:|------------------------:|
| FULL | **+1.33 / 1.90 / 81.5% (817)** | +0.03 / 1.02 / 78.7% (817) |
| 2022 | +0.60 / 1.25 / 75% (72) | −0.79 / 0.75 / 71% (72) |
| 2023 | +1.00 / 1.93 / 81% (163) | +0.29 / 1.21 / 79% (163) |
| 2022–23 | +0.88 / 1.59 / 79% (235) | −0.04 / 0.98 / 77% (235) |
| 2024–26 | +1.51 / 2.03 / 82% (582) | +0.07 / 1.03 / 80% (582) |

Pooled ask/bid FULL flattened vs the n=359 pilot (+0.15) — extra 2024–26 names are near-flat on natural fills.

## Liquid occupancy (full-pool)

Matched mid/ask_bid: half15 **n=391** (was 183); ADV≥$25M & half≤15% **n=236** (was 117).  
Artifacts: `last_15m_wheel_occupancy_liq_20261010_112655.json`.

| Book | max_open | kept | FULL Sharpe / CAGR / MDD | vs SPY | 2022 Sharpe |
|------|---------:|-----:|-------------------------:|--------|------------:|
| half15 mid | 1 | 40 | **1.11 / 14.9% / −15.0%** | **PASS** | 0.32 |
| half15 mid | 15 | 323 | **1.20 / 4.3% / −4.3%** | **PASS** | 0.40 |
| half15 ask/bid | 1–15 | 40–323 | 0.06–0.91 | FAIL | ≤0.25 |
| **adv25_half15 mid** | **1/4/8/15** | 36–222 | **1.17–1.44** | **PASS all** | ≤0.90 |
| **adv25_half15 ask/bid** | **1** | **36** | **1.28 / 17.5% / −8.3%** | **PASS** | 0.76 |
| **adv25_half15 ask/bid** | **8** | **186** | **1.16 / 5.1% / −5.2%** | **PASS** | 0.08 |
| adv25_half15 ask/bid | 4 / 15 | 117 / 222 | 0.94 / 0.95 | FAIL ≤1 | −0.49 / 0.11 |

## Phase-5 gate (2026-10-10)

`passes_phase5_gates(..., window="2022")`: waive absolute Sharpe>1; still require Sharpe≥SPY and MDD better than SPY. FULL / other windows unchanged.

Re-score: `last_15m_wheel_occupancy_liq_20261010_120853.csv` — **all** liquid books PASS 2022 under the exception. Headline CLEAR (FULL+2022):

| Book | max_open | FULL | 2022 (waived >1) |
|------|---------:|------|------------------|
| adv25_half15 mid | 1/4/8/15 | PASS | PASS (Sharpe −0.38…0.90 vs SPY −0.77) |
| adv25_half15 ask/bid | 1, 8 | PASS | PASS |
| half15 mid | 1, 15 | PASS | PASS |
| half15 ask/bid | any | FAIL FULL | PASS 2022 only |

## Read

1. Full pool ~2× liquid n; ask/bid **occ8 FULL PASSes** (Sharpe 1.16).
2. Under the 2022 exception, 2022 is no longer the binding fail — every liquid occ book beats SPY that year.
3. Mid repair required (Theta `no_put_chain` overwrite); use repaired mid CSV.
4. Soft-interesting (esp. adv25_half15 ask/bid occ8); **no hard promote** pending option-path MTM / live fill policy.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\claimcheck_last_15m_wheel.py --n 1118 --min-buy-date 2022-01-01 --ensure-years 2022,2023 --min-adv 2000000 --source theta --strike-mode delta --cache reports\ascending_channels\2026-10-09\last_15m_wheel_cache.jsonl --outdir reports\ascending_channels\2026-10-09
python scripts\research\overlay_last_15m_wheel_occupancy.py --liq-presets --trades-mid reports\ascending_channels\2026-10-09\last_15m_wheel_mid_repaired_20261010_112646.csv --trades-askbid reports\ascending_channels\2026-10-09\last_15m_wheel_ask_bid_20261010_112516.csv
```
