# Last-15m wheel — why mid vs ask/bid diverges (trade compare)

**Date:** 2026-10-09  
**Matched fills:** 340 of 359 (same stock / put entry / expiry / strike)

## Short answer

You are not "buying ask" on entry — `mark_mode=ask_bid` **sells at the bid**. Mid credits `(bid+ask)/2`; natural credits `bid`. On OTM expire (no exit debit), that half-spread is the *entire* P&L gap. It is small in cash-secured % when the name is liquid, but on this last-15m panel the EOD half-spread is often a large % of a thin ~30Δ credit — and assigned cycles get the haircut **twice** (CSP + CC).

| Book | mean gain CS% | mean put credit $ | mean half-spread $ | median haircut as % of mid credit |
|------|--------------:|------------------:|-------------------:|----------------------------------:|
| mid | +1.29 | 3.02 | — | — |
| ask/bid | +0.15 | 2.42 | 0.60 | 13.6% |
| **diff** | **+1.14pp** | **0.60** | | mean haircut 24.6% of mid |

OTM-only mean gain gap = **0.66pp CS** (= put half-spread / strike). Assigned gaps are larger (call_assign **3.33pp**, share_liquidate **1.73pp**) because the CC is also sold at bid.

## Mark rule (code)

`utils/research/spy_wheel.py` → `entry_credit`:

- `mid` → `(bid+ask)/2`
- `ask_bid` → `bid` (sell credit)

Hold-to-expiry OTM puts never buy to close, so exit mark never fires. Assigned cycles open a CC with the same rule.

## OTM examples (gap = put half-spread only)

| Stock | buy | strike / DTE | mid bid–ask (implied) | sell mid | sell bid | half $ | % of mid | gain mid | gain ask/bid | Δ CS pp |
|-------|-----|-------------:|----------------------:|---------:|---------:|-------:|---------:|---------:|-------------:|--------:|
| ACGL | 2022-01-10 | 45 / 39d | 0.95 – 1.20 | 1.075 | 0.95 | 0.125 | 11.6% | +2.39 | +2.11 | **0.28** |
| ICLR | 2023-08-11 | 250 / 35d | 4.80 – 5.30 | 5.05 | 4.80 | 0.25 | 5.0% | +2.02 | +1.92 | **0.10** |
| PNC | 2024-08-06 | 165 / 31d | 2.85 – 4.40 | 3.625 | 2.85 | 0.775 | 21.4% | +2.20 | +1.73 | **0.47** |
| VRRM | 2023-12-11 | 20 / 39d | 0.05 – 0.15 | 0.10 | 0.05 | 0.05 | 50% | +0.50 | +0.25 | **0.25** |

ICLR/ACGL match the intuition: ~30 DTE, half-spread does not wreck the trade. PNC/VRRM show the panel problem — EOD NBBO on mid-cap names is often $0.50–$1.50 wide on a few-dollar credit.

## Assigned examples (CSP + CC both haircut)

| Stock | exit | put mid→bid | call mid→bid | gain mid | gain ask/bid | Δ CS pp |
|-------|------|------------:|-------------:|---------:|-------------:|--------:|
| KEY | share_liq | 0.575→0.55 (−4%) | 0.45→0.40 | −13.98 | −14.28 | **0.30** (tight book) |
| VECO | call_assign | 0.925→0.80 | 0.725→0.35 | +9.46 | +7.68 | **1.79** |
| FULT | call_assign | 2.175→1.15 (−47%) | 2.475→0.95 | −1.75 | −14.50 | **12.75** |
| NMIH | share_liq | 2.40→0.90 (−62%) | 1.325→0.05 | +1.85 | −7.40 | **9.25** |

FULT/NMIH are the scary rows: Theta EOD quotes with multi-dollar spreads. Mid books those as ~$2 credits; natural books ~$1 / nickels. Same path, different fill assumption — not a different strike pick.

## Decomposition (matched n=340)

| exit_reason | n | mean put haircut CS pp | mean call credit $ haircut | mean gain Δ pp |
|-------------|--:|-----------------------:|---------------------------:|---------------:|
| put_expire_otm | 232 | 0.66 | — | **0.66** |
| share_liquidate | 79 | 0.99 | 0.53 | **1.73** |
| call_assign | 29 | 1.43 | 0.75 | **3.33** |

Half-spread ≥20% of mid put credit on **39%** of trades; ≥50% on **17%**.

## Why book E looks "huge" while one OTM trade looks fine

1. **Your intuition holds for liquid OTM cycles** — ICLR-style ~0.1pp CS is noise.
2. **This signal universe is not SPY.** Last-15m names often print wide EOD equity-option NBBOs; median haircut is already ~14% of the credit.
3. **~32% of cycles assign** and take a second sell-at-bid on the CC — that is where mean Δ jumps to 1.7–3.3pp.
4. Mean E mid→ask/bid (+1.29 → +0.15) is almost exactly the average of those haircuts (~1.14pp). PF collapses because winners shrink and fat left tails (FULT/NMIH-style) get worse.

## Artifacts

- Compare CSV: `reports/ascending_channels/2026-10-09/last_15m_wheel_mid_vs_askbid_compare.csv`
- Summary JSON: `…/last_15m_wheel_mid_vs_askbid_summary.json`
- Helpers: `scripts/research/_compare_wheel_mid_askbid.py`, `_wheel_trade_detail.py`

**Still no promote** — natural fills remain the binding gate; mid occupancy PASS is optimistic on these spreads.
