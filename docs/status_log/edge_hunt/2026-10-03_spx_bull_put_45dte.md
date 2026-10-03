# 45 DTE SPXW bull put — claim-check (2026-10-03)

**Verdict: no promote.** A 2-leg 25-wide put credit spread collects the ~$4 credit the sketch expects, and a nickel of package slip barely moves the result. The book still loses money. Winners are capped at half the credit. About one trade in three hits the stop, and the close is often past 2x credit. Median trade +9.5% of risk, mean trade −1.2%, PF 0.88.

## Rules tested

| Rule | Implementation |
|------|----------------|
| Structure | Short put, long put 25 points lower. SPXW Friday expiry |
| DTE | 40–50 at entry (target 45) |
| Short delta | ~0.22 (accepted 0.18–0.26; realized average 0.24) |
| Entry | Mondays, credit at least $1.00 after slip |
| Take profit | Buy-to-close debit <= 50% of fill credit |
| Stop | Buy-to-close debit >= 2x fill credit (loss about 1x credit before fees) |
| Time | Close once 21 DTE or less remains |
| Friction | $0.65/leg x 4, plus mid package slip |
| Spot | Yahoo `^GSPC` (Theta index history returns 403 on the FREE plan for this window) |
| Options | Theta Terminal `:25503` SPXW EOD NBBO |

Harness: `utils/research/spx_bull_put.py`, `scripts/research/claimcheck_spx_bull_put.py`  
Artifacts: `reports/research/spx_bull_put/2026-10-03/` (`summary_baseline.json`, per-book CSVs)  
Wall clock: **630 s**

EOD marks only. A 50% or 2x touch that reverts before the close is invisible. One-lot equity below starts at $100,000 and adds closed P&L. That is not a Sharpe-versus-SPY book.

## Results (2022-01-03 → 2025-08, span 1,327 days)

| Book | n | /yr | WR% | mean RoR% | med RoR% | PF | PnL | Sharpe~ | 2022 PF |
|------|--:|----:|----:|----------:|---------:|---:|----:|--------:|--------:|
| mid, slip $0 | 55 | 15 | 67.3 | −1.08 | +9.38 | **0.90** | −$1,038 | −0.33 | 0.49 |
| **mid, slip $0.05 (spec)** | **55** | **15** | **67.3** | **−1.21** | **+9.46** | **0.88** | **−$1,173** | **−0.37** | **0.48** |
| mid, slip $0.10 | 54 | 15 | 66.7 | −1.59 | +9.24 | 0.84 | −$1,590 | −0.47 | 0.42 |
| mid, slip $0.15 | 54 | 15 | 64.8 | −1.91 | +8.61 | 0.81 | −$1,895 | −0.59 | 0.47 |
| natural ask/bid | 59 | 16 | 40.7 | −10.14 | −13.75 | **0.25** | −$12,873 | −3.13 | 0.30 |
| mid $0.05, no debit stop | 48 | 13 | 72.9 | −3.19 | +9.73 | **0.74** | −$2,950 | −0.55 | 0.35 |
| mid $0.05, up to 4 open | 89 | 24 | 71.9 | +1.34 | +9.79 | **1.24** | +$2,974 | 0.43 | **0.70** |

Spec exits: take-profit 36, stop 18, 21 DTE 1. Average credit $4.25. Average hold 11 days.

### Year split, spec book (mid ± $0.05, one at a time)

| Year | n | WR% | mean RoR% | PF | PnL |
|------|--:|----:|----------:|---:|----:|
| 2022 | 14 | 50.0 | −7.56 | 0.48 | −$2,146 |
| 2023 | 18 | 66.7 | −0.19 | 0.98 | −$62 |
| 2024 | 13 | 76.9 | +2.63 | 1.54 | +$826 |
| 2025 | 10 | 80.0 | +0.86 | 1.14 | +$209 |

Stack of 4, same slip: 2022 n=20 PF 0.70 PnL −$1,372; 2023–25 carry the pooled PF 1.24.

## What this says about the three traps

1. **Friction.** The sketch assumed a 2-leg combo gives up $0.05–$0.10. On these EOD quotes the median cost to cross both legs is **$0.80** (90th percentile about **$1.23**). A nickel of slip changes PF from 0.90 to 0.88. The BWB-fatal $0.15 slip lands at PF 0.81. Crossing both markets (sell the bid, buy the ask) lands at PF 0.25. A live combo can fill inside that natural market. The zero-slip mid book is already below PF 1, so the nickel was not the leak.

2. **Payoff, not a tight filter.** No SMA or down-day gate. About two trades in three win, and the win keeps about half of a $4 credit (~+9% of the ~$2,100 defined risk). The stop exits are fewer than the BWB spot-stop rate (18/55) and they are large enough that the average trade is red. 2022 is the bear-grind year the sketch named: PF 0.48.

3. **The stop.** Turning the 2x debit stop off and holding to 21 DTE drops PF from 0.88 to 0.74. 2022 mean RoR goes from −7.6% to −14.5%. The stop reduces the hole. It does not produce a winning book.

One position at a time yields about **15 trades a year**, because the average hold is 11 days and the next Monday is blocked. Allowing 4 overlapping spreads yields about **24 a year**. The other Mondays had no listed put inside the delta band, or no 25-point wing quote (spec book: 64 busy, 33 no delta, 13 no wing). That is short of the 40–50 trades/year in the sketch.

Defined risk per 1-lot is about **$2,100**. On a $100k account that 1-lot spec book ends at $98,827 (MDD ~3%). The small drawdown is the position size. Scaling toward a $20k risk budget is about 9 of these spreads and multiplies both the 2022 loss and the later recovery. The stack-of-4 Sharpe proxy is 0.43.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_spx_bull_put.py --variants baseline --outdir reports\research\spx_bull_put\2026-10-03
```
