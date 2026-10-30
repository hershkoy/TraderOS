# SPXW 0DTE long reverse iron condor (Jason Rutschel) — claim-check (2026-10-27)

**Verdict: trade count and win/loss size are in the ballpark; the claimed ~37% CAGR / ~49% WR / 15% MDD do not replicate.** Not wired.

Source: Theta Profits interview transcript (Jason Rutschel). Companion to the short 13:45 iron fly ([short fly](2026-10-27_spxw_0dte_iron_fly_claimcheck.md)).

## Rules tested

| Rule | Implementation |
|------|----------------|
| When | Skip Monday; cutoff 11:00 ET |
| Trigger | VIX gaps down vs prior close, then down ≥0.5% from the open |
| Structure | Buy ~$6 / 30-wide put debit + ~$3 / 30-wide call debit (nearest mid) |
| Exit | Cash settle at SPX close; no 100% TP path |
| Marks | Theta SPXW 1m greeks at entry clock. `mid` / `natural` (buy asks, sell bids) |

### VIX trigger modes

| Mode | Data | Honesty |
|------|------|---------|
| **hourly** (primary) | Yahoo `^VIX` 1h (Theta index OHLC needs STANDARD) | Causal; Yahoo 1h capped ~730d |
| **eod_low** (diagnostic) | Session low ≤ open×0.995 + gap; enter 10:30 | Timing-optimistic (low may print after 11:00) |

Harness: `utils/research/spxw_0dte_long_ric.py`, `scripts/research/claimcheck_spxw_0dte_long_ric.py`  
Artifacts: `reports/research/spxw_0dte_long_ric/2026-10-27/` and `.../2026-10-27_eod_low/`

## Claim vs replay

| | Claimed | Hourly mid | Hourly natural | EOD-low mid | EOD-low natural |
|--|--------:|-----------:|---------------:|------------:|----------------:|
| Window | ~4y | ~2y (Yahoo 1h) | ~2y | 2022-05-11→2026-10-24 | same |
| Trades | 338 | 140 | 140 | **329** | 329 |
| Win rate | ~49% | 37.9% | 37.1% | 42.9% | 42.2% |
| Avg win | ~$1400 | **$1432** | $1422 | $1198 | $1192 |
| Avg loss | ~$739 | **$761** | $788 | $795 | $810 |
| Expectancy | — | +$69 | +$33 | +$59 | +$36 |
| PF | — | 1.15 | 1.07 | 1.13 | 1.08 |
| CAGR ($25k 1-lot) | **36.7%** | 18.2% | 9.1% | 13.8% | 9.1% |
| Max DD | **15%** | 32.6% | 38.9% | 32.6% | 36.4% |

Wall: hourly **195 s**; eod_low **115 s** (cache warm).

## Read

- EOD-low n=329 is close to claimed 338 — the filter frequency matches.
- Average winner ≈ 1.5× average loser matches the “don’t need >50% WR” story, but realized WR stays ~38–43%, not ~49%.
- CAGR and MDD miss by a wide margin on both mid and natural. Natural is worse (as expected).
- Hourly (honest trigger) is the weaker book on WR; eod_low lifts WR a bit via look-ahead on the low timing, still far from the claim equity path.
- Do not promote / do not wire. Positive expectancy alone is not enough (fat DD vs claimed 15%).

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_spxw_0dte_long_ric.py
python scripts\research\claimcheck_spxw_0dte_long_ric.py --vix-mode eod_low
python scripts\research\claimcheck_spxw_0dte_long_ric.py --limit 15
```
