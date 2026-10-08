# SPY Wheel / Covered Strangle claim-check (2026-10-07)

Source: [Outlier Trading — Can You Beat The Market With The Wheel?](https://www.youtube.com/watch?v=wKIxZ9Z9mSE) (2007–Sep 2026 export).

Code: `utils/research/spy_wheel.py`, runner `scripts/research/claimcheck_spy_wheel.py`, tests `tests/unit/test_spy_wheel.py`. Artifacts: `reports/research/spy_wheel/`.

## Verdict

**Claim replicates; soft-interesting vs SPY on the OptionsDX window; no hard Phase-5 promote.**

- Wheel mid **~8.25% CAGR** ≈ video ~8%. CS mid **~10.6%** ≈ video ~10% and ≈ book SPY CAGR.
- On this window CS/Wheel both **beat book SPY on Sharpe** (0.72 / 0.71 vs 0.66) and **MDD** (−20% vs −25%).
- Hard gate still fails: Sharpe **&lt; 1.0**. The video’s SPY **−55% MDD** is real (Yahoo) and is almost entirely **2008–09** — options history here starts **2010**, so we cannot re-score the GFC MDD edge.

## Data

| Source | Coverage | Notes |
|--------|----------|-------|
| OptionsDX `option_quotes` | **2010-01-04 → 2023-12-29** | Listed bid/ask/delta; primary book |
| Theta Standard greeks/eod | **2024+** works; **2008/2010 hist 403** | Cannot fill 2007–09; list/expirations min ~2012 |
| Video window | 2007 → 2026-09 | Local book **misses 2007–09** (GFC) and uses ODX through 2023 as headline |

Wall-clock ODX mid+ask_bid × wheel+CS: **~130s**.

## Mechanical rules (frozen)

**Wheel `WHEEL_30_30`:** 30 DTE / 30Δ CSP → assign → CC; hold to expiry; unit-lot (cash may go negative for collateral so the book does not stall when SPY outruns premium).

**CS `CS_20_30_c50_SMA200_tp75`:** buy 200 shares day 0; 50% call cover (1 call); SMA200 put gate (prior close &gt; prior SMA); 20 DTE / 30Δ; close shorts at 75% max profit.

Marks: mid (claim) and ask/bid (sell bid / buy ask; short MTM at ask). No early assignment; dividends not credited.

## Results (OptionsDX 2010-01-04 → 2023-12-29)

| Book | CAGR | MDD | Sharpe | vs SPY CAGR | Closed legs |
|------|------|-----|--------|-------------|-------------|
| SPY buy-hold (same NAV) | **10.82%** | **−25.4%** | 0.66 | — | — |
| Wheel mid | **8.25%** | **−20.2%** | 0.71 | −2.6 ppt | 153 |
| Wheel ask/bid | 8.14% | −20.5% | 0.69 | −2.7 ppt | 153 |
| CS mid | **10.63%** | **−20.4%** | 0.72 | −0.2 ppt | 590 |
| CS ask/bid | 10.50% | −20.0% | 0.71 | −0.3 ppt | 585 |

Claimed (video, 2007–2026): SPY ~11% / −55% MDD; Wheel ~8% / −32%; optimized CS ~10% / −31%.

### Yahoo SPY daily (auto-adjusted) — MDD check

| Window | CAGR | MDD | Sharpe |
|--------|------|-----|--------|
| 2007-01-03 → 2026-09-30 | **10.92%** | **−55.2%** | 0.56 |
| 2010-01-04 → 2023-12-29 (ODX overlap) | 12.90% | −33.7% | 0.74 |
| 2007–2009 only | −5.62% | **−55.2%** | −0.19 |

Worst drawdown date **2009-03-09** (peak→trough −55.2%). Matches the video’s SPY ~11% / ~55% MDD table. Book SPY above is TimescaleDB price NAV (no dividends) so ODX-window MDD (−25%) is milder than Yahoo total-return (−34%).

### Vs claim

- **Wheel CAGR ~8.25% vs ~8%** — replicates on 2010–2023 mid.
- **CS CAGR ~10.6% vs ~10%** — in line; near book-SPY CAGR with better Sharpe/MDD on-window.
- Strategy MDD ~−20% vs claimed Wheel/CS ~−31–32% — milder without 2008; fair comparison for the video’s “cut SPY MDD in half” story needs GFC options (not in ODX/Theta Standard here).
- Ask/bid haircut is small (~0.1–0.2 ppt CAGR).

### Promote gates

`passes_phase5_gates`: **FAIL (Sharpe ≤ 1.0)** — best CS mid Sharpe **0.72**. Relative to book SPY, though: **Sharpe ≥ SPY** and **MDD better than SPY** both hold on 2010–2023. Treat as claim-ok / soft vs SPY, not a Sharpe&gt;1 promote.

## Implementation notes

1. Strict cash-secured gating (`cash >= strike*100`) **stalled** the wheel after ~2014 (NAV stuck ~$15k while SPY rose). Claim-check uses **unit-lot** accounting so one contract stays on; cash can go negative through assignment; equity = cash + shares − short marks.
2. CS scale is **200 shares** so 50% cover is one integer call.
3. Theta 2024+ path is wired via `HybridSource`; headline numbers above are **ODX-only** (`--odx-only`). Sidecar: `reports/research/spy_wheel/summary_theta_2024_2026.json`.

## Theta segment (2024-01-02 → 2026-09-30, mid only)

Short bullish window (not a promote sample):

| Book | CAGR | MDD | Sharpe | Closed |
|------|------|-----|--------|--------|
| Wheel mid | 13.27% | −19.3% | 1.10 | 34 |
| CS mid | 17.04% | −20.9% | 1.12 | 138 |

Wall ~1760s (Theta chain/EOD pulls). Do not treat this slice as the claim-check — ODX 2010–2023 is the headline.

## Next (not done)

- Stitch ODX + Theta into one continuous equity curve.
- Optional dividend credit on long shares.
- Do not promote without Sharpe &gt; 1 on the full (incl. GFC) window.
