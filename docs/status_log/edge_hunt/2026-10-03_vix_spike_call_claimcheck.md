# VIX spike call credit spread — claim-check (2026-10-03)

**Verdict: no promote.** The pasted rule does collect a credit when cash VIX first closes at or above 32, and a dime of package slip is not what decides the book. The credit is about **$1.10** on a 10-point spread (about 11% of the width), not the 30–40% the sketch assumed. The fills are the milder aftershocks. Volmageddon, the March 2020 crash day, and April 2025 never cleared an $0.80 credit at cash+10. Selling the call 10 points above the parity future does take the March 2020 spike, and that one hold-to-expiry-week trade loses **$410**, about 45% of the ~$900 defined risk.

## Rules tested

| Rule | Implementation |
|------|----------------|
| Structure | Short VIX call, long call 10 points higher |
| Trigger | Cash VIX close >= 32. Fresh cross, one position at a time |
| Expiry | Standard Wednesday monthly, nearest with 30–60 calendar DTE |
| Short strike | `round((reference + 10) / 5) * 5`, and that strike plus 10 must be listed |
| Reference | Cash VIX (the pasted code), or the future from put-call parity |
| Take profit | Buy-to-close debit <= 40% of fill credit (60% decay) |
| Stop | None |
| Time | First session with <= 3 calendar days left, else the last mark inside the sample |
| Friction | $0.65/leg x 4, plus mid package slip |
| Spot | Yahoo `^VIX` (Theta index history is 403 on this plan) |
| Options | Theta Terminal `:25503` VIX EOD NBBO |

Harness: `utils/research/vix_spike_call.py`, `scripts/research/claimcheck_vix_spike_call.py`  
Artifacts: `reports/research/vix_spike_call/2026-10-03/` (`summary_baseline.json`, per-book CSVs)  
Wall clock: **235 s**

A leg whose bid-ask is wider than 3 points is dropped. An ask stuck at 10 against a dime bid made two forward exits look like +$507 and +$387; those marks are not in this table. Put-call parity is dropped when the call spread plus the put spread is wider than 2 points (that removed the 2018-02-05 future of 17, whose 25-call was 2.80 / 12.00).

The expiry exit is the EOD mark with 3 calendar days left. VIX options settle to the special opening quotation the next morning. This harness does not have that print. EOD marks also miss a 60% decay that reverts before the close.

`budget_*` sizes each 1-lot to a $15,000 defined-risk cap (about 16 lots when risk is ~$900) and adds that P&L to a $100,000 account. Trade Sharpe uses actual entries per year. It is not a daily Sharpe versus SPY.

## Results (2016-01-01 → 2025-08-29)

| Book | n | /yr | WR% | mean RoR% | med RoR% | PF | PnL | avg credit | 2020 PF |
|------|--:|----:|----:|----------:|---------:|---:|----:|-----------:|--------:|
| **spot+10, slip $0.10 (pasted code)** | **8** | **1.9** | **100** | **+7.78** | **+6.05** | **—** | **+$545** | **$1.11** | — |
| spot+10, slip $0 | 8 | 1.9 | 100 | +8.62 | +7.34 | — | +$600 | $1.17 | — |
| spot+10, natural ask/bid | 6 | 2.4 | 100 | +8.15 | +6.96 | — | +$427 | $1.15 | — |
| **future+10, slip $0.10** | **11** | **1.5** | **90.9** | **+2.42** | **+6.60** | **1.57** | **+$232** | **$1.02** | **0.71** |
| future+10, slip $0 | 11 | 1.5 | 90.9 | +3.39 | +7.51 | 1.83 | +$323 | $1.13 | 0.85 |
| future+10, natural ask/bid | 9 | 1.7 | 88.9 | +0.75 | +5.73 | **1.14** | +$58 | $0.97 | **0.50** |
| future+10, trigger 30, slip $0.10 | 8 | 1.8 | 100 | +7.22 | +6.45 | — | +$514 | $1.03 | — |
| future+10, trigger 35, slip $0.10 | 5 | 1.0 | 80.0 | −2.28 | +6.63 | **0.72** | −$113 | $1.15 | **0.44** |
| future+10, trigger 40, slip $0.10 | 4 | 0.8 | 75.0 | +4.42 | +8.40 | 2.43 | +$147 | $1.20 | 1.85 |
| future+10, stay-above, slip $0.10 | 12 | 1.7 | 91.7 | +2.69 | +6.45 | 1.69 | +$285 | $1.00 | 0.85 |

PF is blank when every trade won. Average hold on the spot book is about 17 days. Median defined risk is about **$908**. Spec spot exits: take-profit 8, expiry-week 0. Spec skips: 13 credits under $0.80, 4 missing quotes, 3 signals while a trade was still open.

### The one crash that filled

Future+10, 2020-03-05. Cash VIX 39.6, parity future 27.5, short the 35 / long the 45, credit $0.98, half-spread to cross both legs $0.13. Held to the expiry-week mark on 2020-04-13, debit $5.05, P&L **−$410** (−45% of risk). Ten other future+10 trades won about **$64** each. That is the payoff the sketch described, at a much smaller credit: one held spike erases about six typical winners. 2020 on that book is PF 0.71.

Raising the trigger to 35 keeps that same March loser and drops the book to PF 0.72. Trigger 40 is four trades.

### Year split, spot+10 slip $0.10

| Year | n | WR% | mean RoR% | PnL |
|------|--:|----:|----------:|----:|
| 2020 | 4 | 100 | +9.87 | +$337 |
| 2022 | 3 | 100 | +5.68 | +$156 |
| 2024 | 1 | 100 | +5.73 | +$52 |

2016, 2017, 2018, 2019, 2021, 2023, and 2025 have no fill. 2022 here is three take-profits, not a bear-grind hole, because the days cash VIX first crossed 32 did not pay $0.80 at the cash+10 strike (2022-03-01 credit $0.75, 2022-04-26 $0.65, 2022-06-13 $0.63, 2022-06-16 $0.55, 2022-09-26 $0.75).

## What this says about the sketch

1. **The credit is not 30–40% of the width.** A 10-point VIX call spread would need a $3–$4 credit for that claim. The spot+10 fills average **$1.11**. A 60% decay of that credit is about **+$68** per spread before you notice the round-turn commission, not +$200. The $200 figure belongs to a $4 credit.

2. **The panics are the days that fail the $0.80 screen.** Cash+10 on a 40–45 VIX print is a call so far above the future that the market will not pay $0.80. Refused: 2018-02-05 (no 45/55 quote), 2020-02-27 (credit $0.03), 2020-03-05 (credit $0.18), 2025-04-04 (credit $0.37 on the 55). The 100% win rate is those days staying out. The future+10 rule does sell the March 2020 spike, and that is the −$410 trade.

3. **$0.10 of slip is small next to a stuck quote, and it is not always small next to the credit.** Median cost to cross both legs on the spot fills is **$0.16**. 2024-08-05 is the exception in the fill list: credit after $0.10 slip was **$0.85**, and crossing both markets cost **$1.38**. The natural ask/bid book does not take that trade. Two raw forward exits (2022-05-26 and 2022-06-21) had a long-call ask of 10 against a bid under $0.40. Those mids are excluded. After the filter both trades are ordinary ~$60 take-profits.

4. **A $15,000 risk budget is about 16 of these spreads.** Sixteen times the March 2020 loss is about **−$6,600** (budget MDD ~6.5% of the $100k account). The spot book's budget P&L of +$9,040 is sixteen copies of eight small winners and still contains no crash fill. The future+10 budget P&L is +$3,901 with that crash included. Neither is a Sharpe-versus-SPY book. The spot trade-Sharpe of 3.65 is eight similar wins.

The Treasury-bill sleeve in the sketch was not tested.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_vix_spike_call.py --variants baseline --outdir reports\research\vix_spike_call\2026-10-03
```
