# Last-15m buy-write vs wheel + earnings flatten≤10 (2026-11-03)

**Verdict: buy-write + earn looks stronger on paper, but it is 100% constant-IV BS (no listed call fills under current Theta FREE). Wheel + earn is the only listed book — mid improves a bit; ask/bid still fails. No promote.**

Same full-pool spine as the 2026-10-10 wheel claimcheck (ADV≥$2M, buy≥2022, n=1118 attempted). Earnings: Yahoo cache, flatten≤10 (entry skip ahead==0; wheel also skips CSP-into-print).

## Headline (pooled)

| Book | n | E | PF | geo | 2022 E / PF |
|------|--:|--:|---:|----:|------------:|
| wheel mid baseline | 817 | +1.33 | 1.90 | 1.80 | +0.60 / 1.25 |
| **wheel mid + earn** | 686 | +1.46 | 2.04 | 1.94 | +0.73 / 1.36 |
| wheel ask/bid baseline | 817 | +0.03 | 1.02 | 1.01 | −0.79 / 0.75 |
| wheel ask/bid + earn | 686 | +0.15 | 1.08 | 1.07 | −0.70 / 0.75 |
| buy-write BS mid baseline | 1118 | +1.97 | 1.90 | 1.89 | +1.76 / 1.66 |
| **buy-write BS mid + earn** | 1103 | **+2.22** | **2.20** | **2.22** | **+2.32 / 2.09** |
| buy-write BS ask/bid +$0.05 fric + earn | 1099 | +2.10 | 2.11 | 2.13 | +2.19 / 2.01 |

Matched on wheel-filled rows only (n=817): BW mid baseline E +1.87 PF 1.81; BW mid+earn n=808 E +2.09 PF 2.08 vs wheel mid+earn n=686 E +1.46 PF 2.04.

## What the earnings overlay did

| Book | skip | early flatten | notes |
|------|-----:|--------------:|-------|
| wheel | 131 | 0 | almost all `skip_csp_into_print` (print before put expiry) |
| buy-write | 15 | 155 | real stock-leg early exits @ last 15m mid + BTC intrinsic floor |

## Data caveats (binding)

1. **Buy-write listed hit rate = 0.** Theta Terminal reported Options FREE; live call chains empty. All BW rows are `call_source=bs` (name ATM IV / ATR IV proxy). Same optimism class as the prior short-put BS book (do not promote from BS alone).
2. **Notional basis differs.** Wheel gain% = cash-secured on put strike; buy-write = stock notional. Do not rank dollar equity without occupancy.
3. **Wheel ask/bid is the honest listed stress** — earn helps only ~0.1pp E; 2022 stays red. Prior MTM occupancy fail still applies.
4. Theta cache prefer-fix: live expiry lists no longer crowd out on-disk chains; empty 403 responses no longer overwrite nonempty chain CSVs.

## Promote gates

**No promote / do not wire `/hot`.** Prefer buy-write+earn only as a research hypothesis until listed call NBBO (Theta Standard / Polygon) fills a real mid and ask/bid book, then occupancy + option MTM.

## Artifacts

- `reports/ascending_channels/2026-11-03_cc_earnings/last_15m_cc_earnings_summary.csv`
- `last_15m_buy_write_{mid,ask_bid}.csv`, `*_earn_flat_detail.csv`
- Runner: `scripts/research/claimcheck_last_15m_cc_earnings.py`
- Helpers: `utils/research/last_15m_buy_write.py`, `utils/research/last_15m_cc_earnings.py`

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\claimcheck_last_15m_cc_earnings.py --outdir reports\ascending_channels\2026-11-03_cc_earnings
```
