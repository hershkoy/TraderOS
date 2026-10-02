# Forward Factor (Volatility Vibes) — Theta listed retest (2026-10-02)

**Source:** [This Simple Options Strategy Crushes SPY (27% CAGR, 2.4 Sharpe)](https://youtu.be/6ao3uXE5KhU) — Volatility Vibes.

**Verdict: no promote.** Retest with ThetaData Standard confirms yesterday's mid picture and **closes the ask/bid gap**: panel FF≥0.20 mid is only weakly positive (n=10), and **Theta EOD ask/bid turns it sharply negative** (mean −50% / PF 0.31). Massive/Polygon cannot host this claim-check on the current plan (historical option NBBO 403; older day aggs 403 beyond ~2y window).

## Data sources

| Source | Role | Result |
|--------|------|--------|
| **ThetaData** Terminal `:25503` | Primary listed EOD / 15:45 NBBO | Used for SPY full book + panel mid + FF ask/bid remarck |
| **Massive (Polygon)** | Attempted day-close mid / half-spread remarck | Day aggs OK on recent SPY sample; **403** on 2020 equity-option day bars and on `/v3/quotes` (Advanced). Not usable for full FF panel |

## A) SPY sequential Mondays 2020-01 → 2025-08 (wall ~11.1m)

Artifacts: `reports/research/forward_factor/2026-10-02_spy/`

| Book | n | WR% | meanRet% | medRet% | PF | Sharpe~ | PnL |
|------|--:|----:|---------:|--------:|---:|--------:|----:|
| mid unfiltered | 32 | 43.8 | +5.3 | −7.4 | 1.26 | 0.22 | +$592 |
| **ask/bid unfiltered** | 32 | 37.5 | **−8.3** | −17.4 | **0.74** | **−0.32** | **−$896** |
| mid FF≥0.20 | 2 | 100 | +262 | — | — | — | +$644 |
| ask/bid FF≥0.20 | 2 | 50 | +40 | — | 15.3 | — | +$310 |
| mid FF≥0.41 | 1 | 0 | −43 | — | 0 | — | −$118 |
| ask/bid FF≥0.41 | 1 | 0 | −77 | — | 0 | — | −$395 |

Matches 2026-10-01 SPY v2 exactly. SPY alone cannot host the playbook threshold.

## B) 14-name panel, month-start Mondays, mid then ask/bid remarck

Symbols: SPY,QQQ,AAPL,TSLA,NVDA,AMD,META,AMZN,GOOGL,NFLX,BA,AES,COIN,PLTR  
Artifacts: `reports/research/forward_factor/2026-10-02_panel/`  
Note: dedicated `mid_ff0.2` pass aborted mid-run (Theta **503** after unfiltered finished); FF≥0.20 mid stats taken by filtering the completed unfiltered book (`panel_mid_ff0.2_from_unfiltered.csv`). Ask/bid via paced EOD remarck (`panel_ff0.2_ask_bid_*`).

| Book | n | WR% | meanRet% | medRet% | PF | Sharpe~ | PnL |
|------|--:|----:|---------:|--------:|---:|--------:|----:|
| mid unfiltered | 192 | 42.2 | +5.3 | **−11.1** | **0.76** | 0.19 | **−$9.2k** |
| mid FF≥0.20 | **10** | 50.0 | **+12.3** | −3.7 | **1.27** | **0.32** | +$223 |
| **ask/bid FF≥0.20** | **9** | **22.2** | **−50.1** | **−83.7** | **0.31** | **−1.08** | **−$1.58k** |

Ask/bid skipped AMZN 2024-07-01 (missing EOD on a leg). Mid FF sleeve is already far from claim Sharpe 2.4; **paying the spread flips the filter from +PF to PF 0.31**.

Vs 2026-10-01 panel mid FF (n=15 E +30.5 PF 1.60 Sharpe~0.62): same unfiltered n=192, but this filter pass keeps 10 names (no SPY month-start FF≥0.20 in the book; max SPY FF on month-start is ~0.09). Prior n=15 / +30% was an optimistic mid upper bound; **honest ask/bid kills it**.

## Interpretation

1. Blind ATM calendars lose after frictions (SPY ask/bid mean −8% / PF 0.74; panel mid unfiltered PF 0.76).
2. FF≥0.20 is a mid-only mirage on this panel once Theta bid/ask is applied.
3. Claim ~27% CAGR / Sharpe 2.4 does not reprint under listed Theta marks.
4. Massive Starter cannot substitute for Theta on multi-year equity-option NBBO / deep history day bars.
5. **Do not promote.**

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_forward_factor.py --start 2020-01-01 --end 2025-08-29 --symbols SPY --marks mid,ask_bid --ff-mins none,0.20,0.41 --outdir reports\research\forward_factor\2026-10-02_spy
python scripts\research\claimcheck_forward_factor.py --start 2020-01-01 --end 2025-08-29 --symbols SPY,QQQ,AAPL,TSLA,NVDA,AMD,META,AMZN,GOOGL,NFLX,BA,AES,COIN,PLTR --marks mid --ff-mins none,0.20 --scan month_start --outdir reports\research\forward_factor\2026-10-02_panel
python scripts\research\_filter_ff_mid.py
python scripts\research\_remark_ff_askbid.py
```
