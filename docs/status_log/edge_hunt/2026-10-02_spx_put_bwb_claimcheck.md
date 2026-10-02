# Carl Allen 21 DTE SPX Put BWB — claim-check (2026-10-02)

**Source:** [Broken Wing Butterfly: How to Collect 10-13% in 2 Weeks (SPX Options)](https://youtu.be/8ea0DwpJtgg) — Theta Profits / Carl Allen (datadrivenoptions.com).

**Verdict: no promote.** The 25×50 structure is not the edge — **regime + execution quality** are. Unfiltered Monday mid dies at **±$0.15 package slip** (PF 0.94). Monday down-day∩SMA50 mid keeps PF ~1.6 under slip but only **~7 trades/year**. Expanding to **daily** scan under the same filters lifts n (~18/yr) but **dilutes** the edge (mid PF 1.13; slip PF 0.80). **Spot-below-short-K stops** are too aggressive on the daily book (WR ~56%, PF 0.72–0.86).

## Mechanical rules (video)

| Rule | Implementation |
|------|----------------|
| Structure | Put BWB 25x50: long K+25, short 2x K, long K-50 |
| DTE | ~21 DTE at open (SPXW weekly) |
| Short delta | ~0.28-0.32 (scroll for credit) |
| Credit | 12-15% of narrow wing on **pre-slip mid** |
| Take profit | Close when package debit <= $0.50 |
| Stop | `debit` = 3x fill credit **or** `spot` = SPX close <= short_k (+ optional buffer) |
| Time | Time-stop at 7 DTE left |
| Claim | ~80% WR no stop / ~70% with stop; 10-13% RoR in ~2 weeks |

## Data

| Piece | Source |
|-------|--------|
| Options | Theta Terminal `:25503` — SPXW EOD NBBO |
| Spot | Yahoo `^GSPC` (Theta index EOD 403 on FREE for long windows) |
| Cadence | Mondays **or daily**; one trade at a time |
| Friction | $0.65/leg x 8; optional **package slip** open −slip / close +slip |

Harness: `utils/research/spx_put_bwb.py`, `scripts/research/claimcheck_spx_put_bwb.py`  
Artifacts: `2026-10-02/` (baseline), `2026-10-02_friction/`, `2026-10-02_daily_regime/`

## A) Baseline (leg mid vs sum-of-legs ask/bid)

| Book | n | WR% | meanRoR% | medRoR% | PF | Sharpe~ | PnL |
|------|--:|----:|---------:|--------:|---:|--------:|----:|
| mid + debit stop | 80 | 71.2 | +1.34 | +12.95 | 1.17 | 0.39 | +$2,349 |
| ask/bid + debit stop | 94 | 36.2 | −31.32 | −33.88 | 0.12 | −4.17 | −$63,879 |

Ask/bid = 4-leg natural + EOD wing blowouts — **not** CBOE COB. Stress caricature only.

## B) Friction / filter suite (Monday mid)

| Book | n | WR% | PF | Sharpe~ | PnL |
|------|--:|----:|---:|--------:|----:|
| mid + stop | 80 | 71.2 | 1.17 | 0.39 | +$2,349 |
| mid + stop + **down-day** | 44 | 77.3 | **1.60** | **1.18** | +$3,659 |
| mid **no stop** | 73 | 74.0 | **0.92** | −0.17 | −$1,375 |
| mid ± **$0.15** | 80 | 70.0 | **0.94** | −0.16 | −$959 |
| mid ± **$0.10** | 80 | 70.0 | 1.00 | 0.00 | +$37 |
| mid + **SMA50** | 51 | 78.4 | **1.87** | **1.67** | +$5,330 |
| mid + SMA200 | 56 | 71.4 | 1.33 | 0.74 | +$2,819 |
| mid ±$0.15 + down-day + SMA50 | 25 | 84.0 | **1.64** | **1.20** | +$2,120 |

## C) Daily regime scan (down-day ∩ SMA50)

Hypothesis: Monday cadence starved sample (n=25). Scan **every session** but only enter when filters fire.

| Book | n | WR% | meanRoR% | medRoR% | PF | Sharpe~ | PnL | Exits |
|------|--:|----:|---------:|--------:|---:|--------:|----:|-------|
| daily mid + debit stop | **65** | 70.8 | +1.06 | +12.29 | **1.13** | 0.29 | +$1,492 | tp 40 / time 13 / sl 12 |
| daily mid ±$0.15 + debit | 63 | 66.7 | −2.11 | +11.35 | **0.80** | −0.56 | −$2,888 | tp 34 / time 15 / sl 14 |
| daily mid + **spot** stop | 73 | 56.2 | −1.24 | +11.59 | **0.86** | −0.45 | −$1,947 | tp 35 / **sl_spot 30** / time 8 |
| daily mid ±$0.15 + spot | 71 | 56.3 | −2.71 | +10.13 | **0.72** | −0.93 | −$4,174 | tp 31 / sl_spot 29 / time 11 |
| daily mid ±$0.15 + spot+5 | 71 | 50.7 | −2.58 | +0.92 | **0.71** | −1.00 | −$3,989 | sl_spot 33 / tp 29 / time 9 |
| **Monday** mid ±$0.15 + spot + filters | 27 | 74.1 | +2.86 | +12.61 | **1.45** | 1.05 | +$1,685 | tp 18 / sl_spot 7 / time 2 |

### Takeaways

1. **Daily scan solves velocity (~18 trades/yr vs ~7) but dilutes the Monday cherry-pick.** Same filters: mid PF falls **1.64 → 1.13**; with $0.15 slip **→ 0.80** (no promote).
2. **Spot-below-short-K is too tight for this delta (~0.30 OTM):** ~40% of daily exits are `sl_spot`, WR collapses to ~56%. Monday-only spot stop still PF 1.45 / n=27 — regime filter mattered more than stop mode.
3. **spot+5 (close <= K+5) is worse**, not better — fires earlier into noise.
4. **No-stop myth remains busted** (Monday PF 0.92). Debit stop still needed on the Monday regime books.
5. **Intraday stop risk** still unmeasured (EOD marks only). Spot stop was the structure proxy; it did not rescue the daily slip book.
6. **Live COB fill profiling** deferred: paper 1-lot SPXW BWB combos later to calibrate `package_slip` (noted in harness docstring / summary `followups`).

## Follow-ups (parked)

- [ ] Paper-trade 1-lot combo fills vs mid (true COB friction)
- [ ] Optional: 1m path walk for intraday 3x-debit stop counts (if Theta load allows)
- [ ] Do **not** promote daily down-day∩SMA50 under $0.15 slip

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_spx_put_bwb.py --variants friction --outdir reports\research\spx_put_bwb\2026-10-02_friction
python scripts\research\claimcheck_spx_put_bwb.py --variants daily_regime --outdir reports\research\spx_put_bwb\2026-10-02_daily_regime
```
