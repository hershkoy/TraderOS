# Forward Factor v2 harness claim-check (2026-10-02)

Follow-up to [2026-10-01_forward_factor_claimcheck.md](2026-10-01_forward_factor_claimcheck.md).

**Verdict: still no promote.** Closing the structural gaps (Friday/short DTE, earnings skip, dynamic exits, mid_slip, ±35Δ double-cal) makes the book **more honest and still weak**. Blind calendars keep losing; FF≥0.20 remains rare after earnings filtering (**n=4 ATM panel, all red**; SPY double-cal FF sleeve **n=5 Sharpe~−1.51**) — nowhere near ~27% CAGR / Sharpe 2.4.

## What changed (v2)

| Gap | Change |
|-----|--------|
| Monthly-only expiries | Default **Friday** pool + monthly fallback; default DTE **30/60** (tol 10) |
| Earnings inflation | Skip single-name if earnings in `[entry, front]` (yfinance cache) |
| Hold-to-front | Dynamic: **TP +25%**, **SL −40%**, **FF→0.05**, **time-stop @15 DTE** |
| ATM tent drift | Optional `structure=double_cal` (±35Δ) wired |
| Harsh ask/bid | Default **`mid_slip`** = mid ± 10% of (front+back) spread widths |
| Weekly EOD holes | Probe expiry candidates; EOD then **15:45 NBBO** fallback |

Harness: `utils/research/forward_factor.py`, `scripts/research/claimcheck_forward_factor.py`  
Artifacts: `reports/research/forward_factor/2026-10-02_v2_spy_30_60/`, `.../2026-10-02_v2_panel_30_60b/`

## Results

### A) SPY Monday 30/60, mid_slip, dynamic exits (wall ~12m)

| Book | n | WR% | meanRet% | medRet% | PF | Sharpe~ | Notes |
|------|--:|----:|---------:|--------:|---:|--------:|-------|
| mid_slip unfiltered | 63 | 33.3 | −2.6 | −1.1 | 0.17 | −3.61 | early FF-exit bug on non-spike entries (fixed later) |
| mid_slip FF≥0.20 | **1** | 0 | −49.2 | — | 0 | — | 2020-03-16 SL |

Even with Fridays + 30/60, SPY FF≥0.20 is still nearly empty.

### B) 14-name panel, month-start, mid_slip, earnings skip, dynamic exits

Symbols: SPY,QQQ,AAPL,TSLA,NVDA,AMD,META,AMZN,GOOGL,NFLX,BA,AES,COIN,PLTR.

| Book | n | WR% | meanRet% | medRet% | PF | Sharpe~ | exits |
|------|--:|----:|---------:|--------:|---:|--------:|-------|
| **mid_slip unfiltered** | **110** | 43.6 | **−11.9** | −4.5 | **0.71** | **−1.4** | time_stop 83 / sl 19 / tp 5 / ff_norm 3 |

Blind calendars still lose after mid_slip + dynamic management (avg hold ~15d, mostly time-stops).

**FF≥0.20 sleeve (month-start, mid_slip, earnings skip; wall ~62m):** **n=4 WR 0% mean −24.4% PF 0** — all red/scratch:

| Symbol | Entry | FF | Ret% | Exit |
|--------|-------|---:|-----:|------|
| SPY | 2020-11-02 | 0.22 | −45.5 | sl |
| SPY | 2024-08-05 | 0.25 | −46.5 | sl |
| SPY | 2025-04-07 | 0.34 | −1.9 | ff_normalize |
| QQQ | 2025-04-07 | 0.29 | −3.7 | ff_normalize |

Skip mix: `no_quotable_pair` 834 / `ff_below` 87 / `iv_fail` 3. Monday FF≥0.20 panel earlier finished **n=0** (earnings 264 vs ff_below 17). Earnings skip removes almost all single-name “spikes” — matches the video’s warning: most equity FF≥0.20 is earnings, not tradable term-structure edge.

Artifacts: `reports/research/forward_factor/2026-10-02_v2_panel_ff20/`.

### C) SPY ±35Δ double calendar, Monday 30/60, mid_slip (wall ~24m)

| Book | n | WR% | meanRet% | medRet% | PF | Sharpe~ | Notes |
|------|--:|----:|---------:|--------:|---:|--------:|-------|
| mid_slip unfiltered | 91 | 54.9 | −1.4 | +4.6 | 0.85 | −0.29 | time_stop 65 / sl 12 / tp 7 |
| mid_slip FF≥0.20 | **5** | 40 | **−5.5** | −0.3 | 1.07 | **−1.51** | 1 TP / 1 SL / 3 ff_norm |

Wider tent does **not** clear the gate. Unfiltered still negative expectancy; FF sleeve is tiny and Sharpe-negative. Artifact: `reports/research/forward_factor/2026-10-02_v2_spy_double_cal/`.

## Interpretation

1. **v1 monthly_only was partly a data necessity** (Theta weeklies often EOD-empty). v2 Friday+fallback restores more candidates, but SPY still rarely prints FF≥0.20.
2. **Earnings filter is binding** — after it, FF≥0.20 on this liquid panel is **n=4** and all lose under mid_slip + dynamic exits.
3. **Dynamic exits do not rescue the edge** — unfiltered ATM PF 0.71 / Sharpe −1.4; filtered ATM sleeve PF 0.
4. **Double calendar does not rescue either** — SPY unfiltered PF 0.85 Sharpe −0.29; FF≥0.20 n=5 Sharpe −1.51.
5. **Do not promote.** Still no path to claimed 27% / 2.4 Sharpe.

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_forward_factor.py --dte-front 30 --dte-back 60 --marks mid_slip --ff-mins none,0.20 --structure atm_call --outdir reports\research\forward_factor\2026-10-02_v2_spy_30_60
python scripts\research\claimcheck_forward_factor.py --symbols SPY,QQQ,AAPL,TSLA,NVDA,AMD,META,AMZN,GOOGL,NFLX,BA,AES,COIN,PLTR --dte-front 30 --dte-back 60 --marks mid_slip --ff-mins 0.20 --scan month_start --outdir reports\research\forward_factor\2026-10-02_v2_panel_ff20
python scripts\research\claimcheck_forward_factor.py --symbols SPY --structure double_cal --dte-front 30 --dte-back 60 --marks mid_slip --ff-mins none,0.20 --outdir reports\research\forward_factor\2026-10-02_v2_spy_double_cal
```
