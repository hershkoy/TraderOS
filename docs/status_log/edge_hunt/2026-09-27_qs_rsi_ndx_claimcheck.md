# QS RSI+IBS Nasdaq-100 claim-check (2026-09-27)

**Source:** YouTube [tjL4wcI403E](https://www.youtube.com/watch?v=tjL4wcI403E) / Substack
"RSI Trading Strategy for Nasdaq Stocks (Only 17% Drawdown)" (2026-09-14).

**Verdict: does not replicate / no promote.**

## Claimed (NDX stocks, Norgate, no survivorship bias)

| Metric | Claim |
|--------|------:|
| n | 1321 |
| E | +1.1% |
| avg win / loss | +3.2% / 3.4% |
| WR | 68% |
| PF | 2.0 |
| CAGR | 11% |
| Exposure | 16% |
| MDD | 17% |
| Max open | 5 |

## Rules tested (public video summary; article rules paywalled)

- Entry (signal close, fill next open): `close > SMA200`, `RSI(3) < 20`, `IBS < 0.3`
- Exit (signal close, fill next open): `close > prior high` (QS exit)
- Occupancy: max 5, rank by lowest RSI
- 0% friction baseline (QS excludes commissions/slippage); also `fric 0.25%` RT

## Ours (TimescaleDB, current NDX members ∩ ALPACA 1d, IB prefix; panel ~2018-11 → 2026-09)

Wall-clock ~12–14s after OHLCV cache hit. 101 symbols.

### Video rules RSI(3)<20 + IBS<0.3, occ-5, fric 0

| Metric | Claim | Ours | Match? |
|--------|------:|-----:|:------:|
| n | 1321 | 1185 | ~ |
| WR% | 68 | 62.3 | no |
| E% | 1.1 | 0.38 | no |
| avg win% | 3.2 | 2.92 | ~ |
| avg loss% | 3.4 | 3.83 | worse |
| PF | 2.0 | 1.26 | no |
| CAGR% | 11 | 10.6 | coincident |
| exposure% | 16 | 57 | no |
| MDD% | 17 | 26 | worse |
| Sharpe | — | 0.53 | fails gate |
| geo year-PF | — | 1.22 | — |

Year buckets (entry date): 2018-19 PF 1.16; 2020-21 1.48; 2022-23 **1.04**; 2024-26 1.26.

### Same book, fric 0.25% RT

n=1185, WR 55.5%, E **−0.12%**, PF **0.93**, CAGR **−5.0%**, MDD 46%, Sharpe −0.25 — dead under modest friction.

### Side check: RSI(2)<10 + IBS<0.3 (Connors-ish), fric 0

Slightly better trade book (n=1203 E +0.58 PF 1.42 CAGR 17.5 Sharpe 0.88) but still far from claim PF 2 / E 1.1, exposure still ~58%, Sharpe < 1 — not a promote and not the video rule set.

## Caveats

1. Exact Substack rules are paywalled; we used the public video summary.
2. Universe is **current** NDX membership (survivorship), not Norgate point-in-time + delistings.
3. Daily panel for the equity curve starts ~2018-11 (ALPACA-heavy), not a multi-decade Norgate sample — trade count can still look similar because recent years dominate signals.
4. QS CAGR≈ours is **not** a validation: we use ~3.5× the capital exposure with worse MDD / PF / expectancy.

## Artifacts

- Harness: `utils/research/qs_rsi_ndx.py`, `scripts/research/claimcheck_qs_rsi_ndx.py`
- Reports: `reports/research/qs_rsi_ndx/qs_rsi_ndx_r3lt20p0_ibs0p3_occ5_fric0p0_*`
- Friction kill: `..._fric0p25_*`

## Gate

Per `docs/assessing_strategies.md` + edge-hunt: PF/expectancy/MDD/Sharpe matter more than WR. Headline 68% WR is not the edge. Book fails PF≈2 and E≈1.1%; fric0.25 kills it; Sharpe 0.53 < 1.0. **No promote.**
