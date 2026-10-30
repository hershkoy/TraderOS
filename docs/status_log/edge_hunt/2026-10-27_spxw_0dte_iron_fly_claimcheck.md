# SPXW 0DTE short iron fly (Jason Rutschel) — claim-check (2026-10-27)

**Verdict: the short late-day iron fly claim replicates on listed Theta mids and on natural bid/ask.** Follow-up path + Phase-5: **soft-promote, not wired** — see [path_phase5](2026-10-27_spxw_0dte_iron_fly_path_phase5.md).

Source: Theta Profits interview transcript (Jason Rutschel). Automated Options Omega book lets the fly expire; discretionary 50% TP is not in this replay.

## Rules tested

| Rule | Implementation |
|------|----------------|
| When | Same-day SPXW expiry, 13:45 ET |
| Shorts | Put nearest 50 delta, call nearest 40 delta |
| Wings | 15 points (long put = short put − 15, long call = short call + 15) |
| Filter | VIX session open < 20; Mon/Tue/Thu/Fri only (skip Wed) |
| Exit | Cash settle at SPX EOD close (SPXW PM-settled); no path management |
| Marks | Theta 1m first-order greeks at 13:45. `mid` = mid in/out; `natural` = sell shorts at bid, buy wings at ask |
| Window | 2024-10-01 → 2026-10-24 (~2y, matches claimed sample) |

Harness: `utils/research/spxw_0dte_iron_fly.py`, `scripts/research/claimcheck_spxw_0dte_iron_fly.py`  
Trades: `reports/research/spxw_0dte_iron_fly/2026-10-27/trades.csv`  
Wall clock: **436 s** (4 workers, Theta `:25503`)

Eligible days after weekday + VIX filters: **316**. Filled **311** (5 skipped: missing put/call/wing quote).

## Claim vs replay

| | Claimed | Mid | Natural |
|--|--------:|----:|--------:|
| Trades | 299 | 311 | 311 |
| Win rate | 57.1% | **58.2%** | **57.2%** |
| Avg win | $513 | $540 | $528 |
| Avg loss | $492 | $477 | $488 |
| Expectancy / trade | ~+$80 (from averages) | **+$115** | **+$93** |
| PF | — | 1.57 | 1.45 |
| Start equity | $25,000 | $25,000 | $25,000 |
| End (1 lot) | ~doubled | $60,620 | $54,047 |
| CAGR | 44% | **53.8%** | **45.4%** |
| Max DD (equity) | — | 12.9% | 14.2% |

Natural bid/ask is the closer match to the interview's 44% CAGR and 57.1% win rate. Mid is optimistic on the credit but still the same trade shape (win rate within ~1pp).

## Caveats

- Settlement uses SPX EOD close from Theta index EOD (monthly chunks; FREE index plan). Not the official AM SOQ (irrelevant for PM-settled SPXW).
- VIX filter uses session **open** from Theta/Yahoo. Interview wording is looser ("VIX below 20").
- No 50% early take-profit path (interview automation expires; desk discretionary TP not modeled).
- One lot only. Buying-power / multi-lot compounding not claimed beyond the $25k example.
- Gamma / pin path + Phase-5 vs SPY + paper BAG profiler: [2026-10-27_spxw_0dte_iron_fly_path_phase5.md](2026-10-27_spxw_0dte_iron_fly_path_phase5.md) (**soft-promote, not wired** until RTH BAG fills).

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_spxw_0dte_iron_fly.py
python scripts\research\claimcheck_spxw_0dte_iron_fly.py --limit 20
```
