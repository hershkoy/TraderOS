# Large-cap earnings expected-move ticker hunt (2026-10-03)

**Verdict: no promote.** COST is the NVDA-like name on the stock screen (14/14 opens inside the 15:45 straddle, never through the downside). Walk-forward put credit spreads on that name still lose once you mark the first morning quote. Mag7 pooled P&L is worse.

Video method: find names that historically stay inside the expected move, then sell the put (or an iron condor). This pass classifies on **trailing reports only**, then prices **the next** report from 2025-01-01. It does not keep the names that won the same 12 dates used to rank them.

## Screen (Phase 1)

Universe: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, AVGO, AMD, NFLX, COST, CRM.

Expected move is the ATM call+put mid at **15:45 ET** on the entry session (report day for AMC, prior session for BMO). Report-day Theta EOD (~17:15) is not used. Yahoo OHLC is unadjusted with `Ticker.splits` so strikes match.

Inventory **n=173** events, 2023-02-01 through 2026-08. Wall clock **196 s**.

| Symbol | n | Inside | Broke down | Opened green | WF PCS | WF IC |
|--------|--:|-------:|-----------:|-------------:|-------:|------:|
| AAPL | 15 | 12 | 2 | 7 | 0 | 7 |
| MSFT | 14 | 8 | 2 | 6 | 2 | 1 |
| GOOGL | 15 | 9 | 3 | 8 | 0 | 0 |
| AMZN | 15 | 10 | 2 | 10 | 5 | 2 |
| META | 14 | 4 | 3 | 9 | 0 | 0 |
| TSLA | 14 | 8 | 4 | 5 | 0 | 0 |
| NVDA | 15 | 11 | 0 | 10 | 3 | 0 |
| AVGO | 14 | 8 | 3 | 8 | 1 | 0 |
| AMD | 14 | 8 | 4 | 5 | 1 | 0 |
| NFLX | 14 | 8 | 3 | 6 | 0 | 0 |
| **COST** | **14** | **14** | **0** | **2** | **8** | 0 |
| CRM | 15 | 9 | 1 | 10 | 0 | 0 |

Walk-forward: last 8–12 completed reports, min 6. Inside-band ≥75% and never broke the downside -> **PCS**. Inside ≥75% with some downsides -> **iron condor**. Else skip. First six reports of each name are always skip (no trail).

META, TSLA, GOOGL, NFLX, CRM never arm. They do not look like the NVDA tape. COST opens inside every time and is almost always red at the open (2/14 green) — containment, not a bullish gap.

## Priced next events (Phase 2)

Holdout **2025-01-01 → 2026-09-01**. 23 candidates, 19 fills, 4 AAPL condors missing a wing quote. Theta 15:45 / first two-sided 09:30 NBBO, $0.65/leg both ways. Wall clock **40 s**.

| Book | n | Spot held | Mid WR | Mid total | Mid PF | Mid RoR | Natural total | Natural PF |
|------|--:|----------:|-------:|----------:|-------:|--------:|--------------:|-----------:|
| **PCS** | **13** | **11/13** | **61.5%** | **−$401** | **0.67** | **−2.0%** | **−$943** | **0.40** |
| **Iron condor** | **6** | **4/6** | **50%** | **−$712** | **0.21** | **−21%** | **−$903** | **0.15** |
| Combined | 19 | | 57.9% | −$1,114 | 0.47 | −7.9% | −$1,846 | 0.30 |

$0.10/leg slip: combined **−$1,459** PF **0.38**.

By name (mid): NVDA PCS 3/3 +$120; AAPL IC 3/3 +$184 (four other AAPL condors had no quote); COST PCS 6 trades 50% WR **−$74** mid / **−$550** natural; AMZN mixed −$276; MSFT 0/2 **−$1,068**.

COST is the object lesson. The short put was still out of the money on all six holdout opens (`spot_held` 6/6). Three of those still lose at the morning bid/ask because the debit to buy the spread back is larger than the credit. Staying inside the expected-move band is not the same as collecting the crush.

## What this says

1. **The video’s ticker hunt, made causal, does not pay.** Names that look like NVDA on trailing containment (COST) and names that only sometimes look like it (AMZN, MSFT) lose on natural marks.
2. **NVDA’s three post-video PCS fills still win** (~+$37–$54 natural). That is the original name, not a panel.
3. **Pooled Mag7 PF is not a promote.** n=19 is one event per name per quarter. Do not n-weight COST’s 14/14 inside-band rate.

Harness: `utils/research/earnings_em.py`, `scripts/research/scan_earnings_em_tickers.py`, `scripts/research/claimcheck_earnings_em_panel.py`  
Artifacts: `reports/research/earnings_em/2026-10-03/` (`inventory.csv`, `wf_trades.csv`, summaries)

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\scan_earnings_em_tickers.py --outdir reports\research\earnings_em\2026-10-03
python scripts\research\claimcheck_earnings_em_panel.py --inventory reports\research\earnings_em\2026-10-03\inventory.csv --outdir reports\research\earnings_em\2026-10-03
```
