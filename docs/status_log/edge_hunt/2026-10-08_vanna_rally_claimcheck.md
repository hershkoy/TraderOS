# Vanna Rally overnight long call — claim-check (2026-10-08)

Source: [Multi Strat Mark — I Backtested a Trade That Made 24% a Year on 1% of the Account](https://www.youtube.com/watch?v=2GqrWos6sJw) (published 2026-10-08).

**Verdict: no promote.** With as-traded SPY spots + listed overnight marks, the book **loses money**. Headline mid + 8x TP: **CAGR −2.6%**, WR **33%**, PF **0.90**, MDD **−31%**, 2022 **−9.0%** calendar (claim **+18%**). SPY on the same window is **+12.7%** CAGR / Sharpe **0.71**. Ask/bid and $0.05 slip are worse. Idle T-bills can print a low-single-digit CAGR while the option sleeve stays red — that is not the claimed 24% engine.

## Rules tested

| Rule | Implementation |
|------|----------------|
| Trigger 1 | VIX open &lt; prior VIX close (Yahoo `^VIX`) |
| Trigger 2 | VIX low ≤ 0.99 × VIX open (daily proxy for “drops 1% after the open”) |
| Trigger 3 | SPY high ≥ 1.005 × SPY open (daily proxy for “S&P up 0.5% from the open”) |
| Structure | Long SPY call, DTE 7–10, strike nearest 2.5% OTM inside a 2–3% band |
| Risk | 2.5% of equity per trade |
| Exit | Next session EOD; optional TP at 8x / 10x entry mid |
| Friction | Mid; mid ±$0.05; ask/bid; $0.65/contract round-turn |
| Idle cash | Optional 5% APY T-bill |
| Premium band | Optional SPY mid in $0.70–$1.00 (SPX $7–$10 / ~10) |

Honesty limits: daily OHLC cannot prove the three intraday conditions were simultaneous; entry is signal-day EOD; next-day TP only if the EOD exit mark clears the multiple. SPY spot is TimescaleDB as-traded (`load_spy_daily`) — Yahoo `auto_adjust=True` back-adjusts dividends and made early-year “2.5% OTM” picks look deep ITM (first pass discarded).

Harness: `utils/research/vanna_rally.py`, `scripts/research/claimcheck_vanna_rally.py`, `tests/unit/test_vanna_rally.py`  
Artifacts: `reports/research/vanna_rally/2026-10-08/`  
Wall clock: **1m 09s** (Theta cache warm)  
Signals: **276** days (2018-01-01 → 2026-09-30). **No 2019 fills** (no signal days cleared the three filters that year on this spot join).

Data: Yahoo VIX; TimescaleDB SPY 1d; SPY OptionsDX EOD through 2023-12-29; Theta EOD from 2024.

## Results

| Book | n | WR% | PF | PnL | CAGR | Sharpe | MDD | 2022 cal |
|------|--:|----:|---:|----:|-----:|-------:|----:|---------:|
| **mid + TP 8x (headline)** | **274** | **33.2** | **0.90** | **−$20.7k** | **−2.6%** | **−0.17** | **−31.1%** | **−9.0%** |
| mid slip $0.05 + TP 8x | 273 | 27.8 | 0.40 | −$87.1k | −19.3% | −1.34 | −87.0% | −15.4% |
| ask/bid + TP 8x | 274 | 31.0 | 0.79 | −$41.2k | −5.8% | −0.40 | −49.0% | −10.5% |
| mid + TP 10x | 274 | 33.2 | 0.92 | −$17.3k | −2.1% | −0.13 | −31.1% | −9.0% |
| mid no TP | 274 | 33.2 | 0.97 | −$6.5k | −0.7% | −0.04 | −31.1% | −9.0% |
| mid TP 8x + T-bill 5% | 274 | 33.2 | 0.91 | −$21.9k* | +1.5% | 0.10 | −23.8% | −4.2% |
| mid TP 8x + T-bill + $0.70–$1 prem | 274 | 33.2 | 0.92 | −$18.9k* | +1.7% | 0.12 | −23.6% | −3.0% |
| **SPY buy & hold** | — | — | — | — | **12.7%** | **0.71** | **−25.4%** | — |

\*Option-leg PnL; CAGR includes T-bill on idle cash.

Claimed (video): strategy **24.4%** CAGR / **−15.6%** MDD / **~40%** WR / **+18%** in 2022; SPY **14.4%** / **−33.7%**; $100k → **$671k**.

### Vs claim

| Claim | Ours (mid TP8x) |
|-------|-----------------|
| 24.4% CAGR | **−2.6%** |
| −15.6% MDD | **−31.1%** (worse than SPY −25.4%) |
| ~40% WR | **33%** |
| +18% in 2022 | **−9.0%** |
| 10 best ≈ 90% of profit | Top-10 ≈ **35%** of gross wins |
| 11-loss streak | Max consecutive losses **18** |
| Avg ~1% capital deployed | Mostly idle overnight (true) |

TP 8x/10x almost never fills on next-day EOD (**1 / 274**). Mean return on premium is slightly positive (~+4%) with a **negative median** — fat left tail, not a 700% right tail.

$0.05 slip is lethal on sub-dime premiums (many 2018 nickels → `no_mark` after slip). The video’s $7–$10 SPX (~$0.70–$1 SPY) band still loses with T-bill carry only (~1.7% CAGR).

Promote gate (Sharpe &gt; 1, Sharpe ≥ SPY, MDD better than SPY): **fails** on every book.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_vanna_rally.py --variants baseline --outdir reports\research\vanna_rally\2026-10-08
pytest tests\unit\test_vanna_rally.py -v
```
