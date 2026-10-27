# QQQ 0-DTE 9:45 short put — claim-check (2026-10-24)

**Verdict: the backtest claim replicates on listed mids. The $556k headline is a fully margined compounder, and it does not survive selling the bid and buying the ask.** Not wired.

Video: Options With Ravish, [7S-mGV4D-fI](https://www.youtube.com/watch?v=7S-mGV4D-fI) (published 2026-10-24). No transcript was available. Rules below are the stated mechanical setup.

## Rules tested

| Rule | Implementation |
|------|----------------|
| When | Every QQQ expiration morning, 09:45 ET. Same-day expiry only |
| Strike | Put nearest −0.30 delta. A 2.5-point-below book is the video's other wording |
| Take profit | Keep 95% of the credit (buy back at 5% of the credit) |
| Stop | Loss of 200% of the credit. Sell $1.00, buy back at $3.00, loss $2.00 |
| Time | Buy back at 15:45 ET if neither bracket has printed |
| Marks | Theta 1-minute NBBO. `target` fills at the bracket when the mid crosses it. `natural` sells the bid and buys the ask |
| Window | 2022-05-02 → 2026-10-23 (the video's May 2022 start) |

Harness: `utils/research/qqq_0dte_put.py`, `scripts/research/claimcheck_qqq_0dte_put.py`  
Trades: `reports/research/qqq_0dte_put/2026-10-24/trades.csv`  
Wall clock: **475 s** (4 workers, Theta `:25503`)

1,075 QQQ expirations in the window. One day had no 09:45 chain. The 30-delta book is **1,074** trades (video: 1,067). Median distance to the strike is **2.2 points**, so "30 delta" and "2–3 points below" are the same trade on average. Mean entry credit **$0.89**.

## What matches

The video's $85 winner and $174 loser are the bracket fills, not the average of every green or red day.

| | Video | Replay, no slippage |
|--|------:|--------------------:|
| Trades | 1,067 | 1,074 |
| Win rate | 73.3% (782 / 1,067) | 72.9% |
| Take-profit fill | $85 | **$84.59** (610 fills) |
| Stop fill | $174 | **$173.32** (247 fills) |
| 15:45 flatten | — | 217 (173 of them still green, mean +$35) |
| One contract, all exits | $16,880 implied by the averages | **+$16,431** |

Exits on the no-slippage book: take-profit 610, stop 247, 15:45 flatten 217. Longest losing streak 4. Dropping the 10 largest winners leaves **+$14,096** of the +$16,431, so the one-lot result is not a handful of outliers.

Partial 2022 (from May, n=118) is green: PF 1.42, +$2,046 per contract. Every later year is also green on this book (PF 1.28–1.46). January–April 2022 is outside the video's window and was not tested.

## The $556,000 account

One contract cannot produce $556,000. The quoted averages imply about **$16,880**. The account number is that stream, compounded, with as many contracts as buying power allows.

Standard naked-put margin here is `max(20% of spot − out-of-the-money amount, 10% of spot) + credit`, times 100. Across these fills the requirement averages **$9,398** (median $9,321). That is the video's "roughly $10,000."

| Sizing | Profit | End equity | CAGR | Max DD | Sharpe | Avg contracts |
|--------|-------:|-----------:|-----:|-------:|-------:|--------------:|
| Video | $556,000 | $656,000 | 52% | 13% | — | — |
| No-slippage, flat $10k | $358,027 | $458,027 | 40.5% | 16.5% | 2.00 | 23 |
| **No-slippage, Reg-T** | **$550,586** | **$650,586** | **51.9%** | **14.4%** | **2.21** | **34** |
| Sell bid / buy ask, Reg-T | $257,536 | $357,536 | 32.9% | 17.4% | 1.48 | 24 |
| Bid/ask + $0.65/side, Reg-T | $201,984 | $301,984 | 28.0% | 18.5% | 1.30 | 22 |
| Bid/ask, flat $10k | $171,225 | $271,225 | 25.0% | 19.6% | 1.30 | 17 |

Same window, buy and hold, $100k start: QQQ CAGR 20.1%, max DD 22.8%, Sharpe 0.94. SPY CAGR 16.1%, max DD 18.8%, Sharpe 0.97.

A flat $10,000 reserve understates the early account, because margin was about $6,000 when QQQ was near $300 and about $14,000 when it was near $700. Reg-T uses that schedule. It was not fit to the $556k.

## What the spread does

Selling the bid and buying the ask keeps the win rate (72.2%) and the profit factor (1.22). One contract makes **+$11,157** instead of +$16,431. On Reg-T the account profit falls from $551k to **$258k**, and the drawdown rises from 14% to **17%**. Adding $0.65 a side per contract leaves about **$202k**, Sharpe 1.30, drawdown 18.5%.

That bid/ask Reg-T book is still ahead of SPY on Sharpe (1.48 vs 0.97) and drawdown (17.4% vs 18.8%). The flat-$10k bid/ask book is not: its 19.6% drawdown is deeper than SPY's 18.8%.

54 of 247 stops were already more than $20 worse on the minute mid than the $3.00-style bracket. A stop that spikes and reverts inside the minute is invisible on this tape. The 15:45 flatten is the NBBO at 15:45. A close that does not fill, and the assignment window after the cash close, are not in these results.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_qqq_0dte_put.py
```
