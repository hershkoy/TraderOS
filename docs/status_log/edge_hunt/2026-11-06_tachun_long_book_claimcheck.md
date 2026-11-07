# Tachun long book — survivor-biased / estimate-free claim-check

Date: 2026-11-06. Runner: `scripts/research/backtest_tachun_long_book.py`. Logic: `utils/research/tachun_long_book.py`. Artifacts: `reports/research/tachun_long_book/tachun_*_20261106_114945.*`.

## Setup

- No forward EPS. No delisted tape. Label: **survivor-biased and estimate-free**.
- Trailing FY screen on `fundamentals_quarterly` (as-of `filed`): mkt cap > $2B, 5y sales and EPS CAGR > 0, EPS YoY > 5%, ROE > 15%, current ratio > 1, gross > 40%, operating > 20%, net > 10%, 5y revenue / NI / OCF end > start, debt/EBITDA ≤ 4 (or debt_zero), interest/OCF ≤ 40%. One failed filing allowed while held; two consecutive exits.
- 20y fade DCF on FCF (else NI), g0 = clipped 5y sales CAGR, rf = FRED DGS10, ERP = 5%, beta vs SPY with fixed haircut (opt ≤ 1.5, pess = max(0.85×raw, opt+0.2)). Buy / keep at or below 1.5× optimistic value; new names must be ≤ optimistic.
- Month-end equal-weight up to 15; unused weight is cash. Prices: ALPACA 1d with IB prefix from 2006. Window: 2009-01-02 → 2026-08-21 (SPY IB last bar).

Wall-clock: **103.3s**.

## Result

| Book | CAGR | Sharpe | MDD | % invested |
|------|------|--------|-----|------------|
| Tachun EW≤15 | 6.80% | 0.33 | −43.2% | 47% |
| SPY | 12.70% | 0.71 | −34.1% | 100% |

Phase-5 vs SPY: **FAIL** (Sharpe≤1, Sharpe\<SPY, MDD≥SPY).

Screen: 1,616 FY events on the price-loaded candidates → **40** pass events, **28** symbols, **40** DCF valuations. Average names per rebalance ≈ **2.6** (never fills 15). Flat / near-cash years: 2009–2012, 2017–2018 (thin early XBRL + rare passers). Pass list includes MSFT, CSCO, AMAT, LRCX, ASML, NTES, RMD, CTAS, LULU, among others.

## Read

The trailing quality stack is so selective on this survivor panel that the book sits in cash about half the time and never runs a true 8–15 name sleeve. Versus buy-and-hold SPY it loses on CAGR, Sharpe, and drawdown. This is not a promote. Forward EPS gates and a delisted tape are still missing from the course recipe; this run only answers the free, estimate-free path.
