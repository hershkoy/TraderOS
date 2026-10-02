# Carl Allen 21 DTE SPX Put BWB — claim-check (2026-10-02)

**Source:** [Broken Wing Butterfly: How to Collect 10-13% in 2 Weeks (SPX Options)](https://youtu.be/8ea0DwpJtgg) — Theta Profits / Carl Allen (datadrivenoptions.com).

**Verdict: no promote (mechanical Monday mid).** Mid validates Carl's ~70% WR / ~13% median RoR headline, but pooled edge is thin (PF 1.17). **Leg-by-leg ask/bid is not a fair live proxy** (COB fills); **COB-style mid ± $0.15 package slip flips the baseline mid book to PF 0.94**. Filters Carl hinted at (down-day) and a simple SPX>SMA50 gate lift mid and keep a thin edge under $0.15 slip — still small-n / research-only, not a live promote vs Forward Factor calendars.

## Mechanical rules (video)

| Rule | Implementation |
|------|----------------|
| Structure | Put BWB 25x50: long K+25, short 2x K, long K-50 |
| DTE | ~21 DTE at open (SPXW weekly) |
| Short delta | ~0.28-0.32 (scroll for credit) |
| Credit | 12-15% of narrow wing on **pre-slip mid** |
| Take profit | Close when package debit <= $0.50 |
| Stop | Close when debit >= 3x **fill** credit |
| Time | Time-stop at 7 DTE left |
| Claim | ~80% WR no stop / ~70% with stop; 10-13% RoR in ~2 weeks |

## Data

| Piece | Source |
|-------|--------|
| Options | Theta Terminal `:25503` — SPXW EOD NBBO |
| Spot | Yahoo `^GSPC` (Theta index EOD 403 on FREE for long windows) |
| Cadence | Sequential **Mondays** 2022-01-03 -> 2025-08-29 (one trade at a time) |
| Friction | $0.65/leg x 8; optional **package slip** open credit −slip / close debit +slip |

Harness: `utils/research/spx_put_bwb.py`, `scripts/research/claimcheck_spx_put_bwb.py`  
Artifacts: `reports/research/spx_put_bwb/2026-10-02/` (baseline mid+ask_bid)  
Friction suite: `reports/research/spx_put_bwb/2026-10-02_friction/` (wall ~34.5m, shared Theta cache)

## A) Baseline (leg mid vs sum-of-legs ask/bid)

| Book | n | WR% | meanRoR% | medRoR% | PF | Sharpe~ | PnL | Exits |
|------|--:|----:|---------:|--------:|---:|--------:|----:|-------|
| **mid + stop** | 80 | **71.2** | +1.34 | **+12.95** | **1.17** | 0.39 | +$2,349 | tp 46 / time 20 / sl 14 |
| **ask/bid + stop** | 94 | 36.2 | **-31.32** | -33.88 | **0.12** | -4.17 | **-$63,879** | sl 59 / tp 21 / time 14 |

Ask/bid collapses from **4-leg natural NBBO + EOD wing blowouts** (e.g. 2024-12-16 debit $86 on a 25-wide) — **not** a CBOE complex-order fill. Treat ask/bid as a stress caricature, not live.

### Year buckets (baseline mid)

| Year | mid WR% | mid meanRoR% | mid PnL |
|------|--------:|-------------:|--------:|
| 2022 | 57.1 | -5.65 | -$2,559 |
| 2023 | 79.2 | +5.27 | +$2,728 |
| 2024 | 75.0 | +5.75 | +$3,020 |
| 2025 | 72.7 | -3.53 | -$840 |

## B) Friction / filter suite (mid only; shared cache)

| Book | n | WR% | meanRoR% | medRoR% | PF | Sharpe~ | PnL |
|------|--:|----:|---------:|--------:|---:|--------:|----:|
| mid + stop (replay) | 80 | 71.2 | +1.34 | +12.95 | 1.17 | 0.39 | +$2,349 |
| **mid + stop + down-day** | 44 | **77.3** | +3.81 | +13.63 | **1.60** | **1.18** | +$3,659 |
| mid **no stop** | 73 | 74.0 | −0.89 | +13.09 | **0.92** | −0.17 | −$1,375 |
| mid ± **$0.15** slip + stop | 80 | 70.0 | −0.57 | +12.14 | **0.94** | −0.16 | −$959 |
| mid ± **$0.10** slip + stop | 80 | 70.0 | +0.01 | +12.21 | **1.00** | 0.00 | +$37 |
| **mid + stop + SPX>SMA50** | 51 | **78.4** | +4.81 | +13.15 | **1.87** | **1.67** | +$5,330 |
| mid + stop + SPX>SMA200 | 56 | 71.4 | +2.31 | +12.62 | 1.33 | 0.74 | +$2,819 |
| **mid ±$0.15 + down-day + SMA50** | 25 | **84.0** | +3.85 | +12.61 | **1.64** | **1.20** | +$2,120 |

### Takeaways

1. **COB-style $0.15 package slip kills the unfiltered Monday book** (PF 0.94). Edge needs either tighter fills (~$0.10 → flat) or filters.
2. **`--down-day` works as Carl described:** fewer trades, fewer stops (5 vs 14), PF 1.60 / Sharpe~1.18 on mid.
3. **`--no-stop` does not deliver ~80% profitable expectancy:** WR 74% but PF 0.92 — uncapped time-exits / deep losers dominate. Stop is doing real work.
4. **SPX > SMA50** is the strongest single overlay on this window (PF 1.87 / Sharpe~1.67) — strips most 2022 structural shorts. SMA200 weaker.
5. **Filtered + $0.15 slip** (down-day ∩ SMA50) still PF 1.64 but **n=25** — interesting, not promote-ready (sample thin; occupancy / live combo fills untested).
6. vs **Forward Factor calendars:** FF mid panel was already weak and ask/bid deeply red. BWB with down-day/SMA50 mid looks **better on mid PF** than FF's mid books, but both fail honest marketable stress without a combo-fill model. Prefer neither for live alert-and-execute until combo NBBO or broker fill logs exist.

## Notes

- Rolling defense (debit-spread harvest → roll credit) not modeled; doubles max risk.
- Stop is still **package-debit** based, not spot-at-short-K (possible follow-up).
- Marks: `mid`, `ask_bid`, `mid_slip0.15` (or `--package-slip 0.15`); filters `--down-day`, `--trend-ma 50|200`; suite `--variants friction`.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_spx_put_bwb.py --start 2022-01-01 --end 2025-08-29 --marks mid,ask_bid
python scripts\research\claimcheck_spx_put_bwb.py --variants friction --outdir reports\research\spx_put_bwb\2026-10-02_friction
```
