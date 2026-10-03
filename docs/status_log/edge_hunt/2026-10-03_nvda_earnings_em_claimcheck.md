# NVDA earnings expected-move sale — claim-check (2026-10-03)

**Verdict: no promote.** Two of the stock facts match the video, and the put sale at the expected-move strike did win every report in the sample. The iron condor does not survive the next morning's option quote. This is one ticker and 12 events. It is not a Sharpe-versus-SPY book.

Video: [Best Earnings Options Trading Strategy](https://youtu.be/FtX1JshVLvg) (Options With Ravish, 2026-02-24). The example on screen is NVDA near $190, a 5.7% expected move, the 27 Feb 2026 expiry, and a 15:45 entry / next-open exit.

## Rules tested

| Rule | Implementation |
|------|----------------|
| Expected move | ATM call mid + ATM put mid on the first listed expiry after the report. Spot is put-call parity at 15:45 ET, not the official close |
| Short put | Highest listed strike strictly below spot − expected move |
| Near put | Highest listed strike strictly below spot (the "just under the price" variant) |
| Wing | 5/190 of spot, snapped to a listed strike. A literal 5-point wing is priced when that strike differs |
| Iron condor | Short put and short call just outside the band, wings as above |
| Entry | 15:45 ET NBBO on an after-close report |
| Exit | First two-sided NBBO at/after 09:30 ET the next session (09:30 itself is often 0/0) |
| Friction | $0.65 per leg, both ways. Mid, mid minus $0.05/$0.10 per leg (clamped to the bid/ask), and natural bid/ask |
| Sample | 12 reports 2023-02-22 through 2025-11-19 (the window before the video). Feb/May/Aug 2026 are out of sample |

Harness: `utils/research/nvda_earnings_em.py`, `scripts/research/claimcheck_nvda_earnings_em.py`  
Artifacts: `reports/research/nvda_earnings_em/2026-10-03/` (`trades.csv`, `summary.json`)  
Quotes: Theta Terminal `:25503` 1-minute NBBO. Yahoo daily OHLC, multiplied by 10 before the 2024-06-10 split so strikes match. Theta stock history before 2024 is 403 on this plan, and intraday stock quotes are 403, so the open used for the gap is the Yahoo official open.  
Wall clock: **49 s** for the 15-event pass (two events were already in the quote cache from a smoke run).

The Friday straddle also prices the sessions after the open, so it is a bit wider than a pure overnight move. That makes "inside the band" easier to hit.

## What the stock actually did

| Fact in the video | This sample |
|-------------------|-------------|
| Stayed inside the expected move 9 of 12 | **9 of 12** |
| Never broke the downside | **0 of 12** |
| Opened flat or higher 11 of 12, one small red | **9 of 12 green**. Reds: 2023-11-21 −0.18%, 2025-08-27 −0.43%, **2024-08-28 −3.38%** |
| Biggest gap about +26% | **2023-05-24 +26.15%** (305 to 385) |

The three upside breaks are 2023-02-22 (+12.9% vs a 6.9% band), 2023-05-24 (+26.2% vs 6.9%), and 2024-02-21 (+11.2% vs 9.5%).

## Option results, claim window (n=12)

P&L is dollars per one-lot, after commission. RoR for a spread is that P&L divided by width minus credit. RoR for a short put is P&L divided by strike minus credit (cash-secured).

| Book | Spot held | Mid WR | Mid mean | Mid total | Mid PF | Mid RoR | Natural total | Natural PF |
|------|----------:|-------:|---------:|----------:|-------:|--------:|--------------:|-----------:|
| **Short put at the expected move** | **12/12** | **100%** | **+$319** | **+$3,833** | — | **+1.06%** | **+$3,771** | — |
| **Put credit spread, scaled wing** | **12/12** | **100%** | **+$131** | **+$1,567** | — | **+16.6%** | **+$1,471** | **—** |
| Put credit spread, literal 5-wide (9 dates) | 9/9 | 100% | +$75 | +$674 | — | +18.1% | +$576 | — |
| **Iron condor, scaled wing** | **9/12** | **67%** | **+$9** | **+$110** | **1.11** | +5.6% | **−$483** | **0.67** |
| Iron condor, literal 5-wide (9 dates) | 7/9 | 67% | +$23 | +$203 | 1.45 | +5.9% | −$333 | 0.63 |
| Short put just under spot | 11/12 | 100% | +$1,102 | +$13,228 | — | +3.0% | +$13,055 | 2073 |
| Put credit spread just under spot | 11/12 | 92% | +$309 | +$3,705 | 45.9 | +51% | +$3,406 | 34.2 |

$0.10 per leg of slip does not change the expected-move put or the expected-move put spread (both still 12/12; the spread's total goes from +$1,567 to +$1,476). The same $0.10 turns the iron condor from +$110 to **−$164** (PF 0.86).

Average credit on the expected-move short put is **$3.34**. On the put spread it is **$1.40**. On the iron condor it is **$2.82**. The video's February 2026 sketch ($1.95 put, ~$1.00 spread, ~$1.51 condor) is the post-split $190 regime. The 2023–early-2024 puts were on a $200–$950 pre-split stock, so the dollar credits are larger.

### The iron condor losses

| Report | Gap | Short call | Mid P&L | Natural P&L | If closed at intrinsic of the official open |
|--------|----:|-----------:|--------:|------------:|--------------------------------------------:|
| 2023-02-22 | +12.9% | 222.5 | −$207 | −$329 | −$345 |
| 2023-05-24 | +26.2% | 325 | −$421 | −$576 | −$505 |
| 2024-02-21 | +11.2% | 740 | −$267 | −$379 | −$418 |
| 2024-05-22 | +7.5% | 1025 | −$91 | −$189 | **+$815** |

The first three opened through the short call. 2024-05-22 is the one the video's win-rate would count: official open 1020, short call 1025, expected-move top 1023, so the open print is inside. The 09:30 option quote is 0/0. The first two-sided market is 09:31, with the stock's synthetic about **1025** and the 1025 call at **16.85 / 17.45** (the morning straddle there is still about $34). Buying the condor back cost more than the $8.20 credit. "Inside at the open print" and "a fill at the first option quote" are not the same trade.

## After the video (3 reports)

| Report | Gap | Inside band | EM put spread | Iron condor mid | Near put spread |
|--------|----:|:-----------:|--------------:|----------------:|----------------:|
| 2026-02-25 | −0.66% | yes | +$25 | +$91 | −$30 |
| 2026-05-20 | −0.53% | yes | +$40 | +$101 | −$13 |
| 2026-08-26 | +6.30% | no (through the call) | +$54 | −$83 | +$192 |

The expected-move put and put spread are 3/3 on mid and on natural. The iron condor is 2/3 and still net positive on this short sample (+$109 mid, +$81 natural). The nearer put spread lost the two small down opens, because those opens were under the short strike (195 and 222.5) even though they were inside the expected-move band. The video's 5.7% band, measured the day before at ~$190, was 5.98% at 15:45 on 25 Feb with the stock at 197.

## What this says about the video

1. **The containment claim is real on this window.** Nine of twelve opens landed inside the straddle band, and none broke the downside. The +26% May 2023 gap is the one he cites.

2. **The "11 of 12 opened green" claim is not.** Nine opened at or above the prior close. Two reds were under half a percent. 28 Aug 2024 opened down 3.4%, and that is the day the near-the-money put spread loses (−$83 mid, −$103 natural). The expected-move put (112 strike, open 121) still wins.

3. **The 100% put and put-spread win rate replicates, including the bid/ask cross.** It replicates because the open never reached the short strike, not because the model ignored spreads. Cash-secured, that win is about **1% of the strike**. On the defined-risk spread it is about **+16% of the max loss** per report, before you notice that one upside gap of the May 2023 size would have been irrelevant to this put — the losses live on the call side.

4. **The 75% iron-condor win rate is a stock count.** Nine of twelve opens sat between the short strikes. Marking the condor at the first morning quote wins eight and loses four. Mid-price the book is +$110 (PF 1.11). Crossing the bid/ask it is **−$483** (PF 0.67). A dime of slip per leg is already a losing book. The three real gap-ups are −$207, −$421, and −$267 at mid, against typical inside wins of about +$30 to +$240.

5. **Selling the put just under spot is a different trade.** It won the claim window at mid and lost one natural fill (28 Aug 2024, −$6). Two of the three later reports opened through that strike. The video's 100% figure belongs to the expected-move strike, where the downside was never broken.

Position sizing at 1–2% of the account was not a portfolio test. One lot of the put spread risked a few hundred dollars post-split and a few thousand pre-split. Massive/Polygon historical NBBO was not used; Theta had the 1-minute quotes.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\claimcheck_nvda_earnings_em.py
```
