# Forward Factor (Volatility Vibes) — claim-check (2026-10-01)

**Source:** [This Simple Options Strategy Crushes SPY (27% CAGR, 2.4 Sharpe)](https://youtu.be/6ao3uXE5KhU) — Volatility Vibes (2025-10-10).

**Verdict: no promote.** Blind ATM calendars lose once you pay the spread (matches the video's baseline). FF ≥ 0.20 flips mid marks positive on a 15-name panel, but **n is tiny**, **Sharpe proxy ≪ 2.4**, and the claimed ~27% CAGR / 2.4 Sharpe quarter-Kelly book does **not** reprint under listed Theta marks. Independent ORATS replications circulating online (~7% mean vs ~24% claimed) point the same way.

## Claim (video)

| Item | Claim |
|------|--------|
| Signal | FF = (IV_front − IV_fwd) / IV_fwd; IV_fwd from variance additivity |
| Entry | Long ATM call calendar (or ±35Δ double cal) when FF high; playbook ≥0.20; ATM 60/90 table uses 0.41 |
| Exit | Close as a spread just before front expiry |
| Headline | ATM call calendar **60/90**, quarter Kelly: **~26.7% CAGR / Sharpe 2.40** |
| Sample | ~19y, multi-name, ~300k spread instances, frictions + capacity |

## Ours (ThetaData listed)

Harness: `utils/research/forward_factor.py`, `scripts/research/claimcheck_forward_factor.py`  
Artifacts: `reports/research/forward_factor/2026-10-01_spy_v2/`, `.../2026-10-01_panel/`  
Marks: mid vs ask/bid (sell front@bid / buy back@ask; reverse on exit); $0.65/leg ×4; monthly+month-end expiries (Theta weeklies often 472).  
**Not** claimed: ex-earn IV, double calendar, fractional Kelly equity, 19y full OPRA universe.

### A) SPY sequential Mondays 2020-01 → 2025-08 (wall ~12.5m)

| Book | n | WR% | meanRet% | medRet% | PF | Sharpe~ | PnL |
|------|--:|----:|---------:|--------:|---:|--------:|----:|
| mid unfiltered | 32 | 43.8 | +5.3 | −7.4 | 1.26 | 0.22 | +$592 |
| **ask/bid unfiltered** | 32 | 37.5 | **−8.3** | −17.4 | **0.74** | **−0.32** | **−$896** |
| mid FF≥0.20 | 2 | 100 | +262 | — | — | — | +$644 |
| ask/bid FF≥0.20 | 2 | 50 | +40 | — | — | — | +$310 |
| mid FF≥0.41 | 1 | 0 | −43 | — | 0 | — | −$118 |
| ask/bid FF≥0.41 | 1 | 0 | −77 | — | 0 | — | −$395 |

SPY almost never prints FF≥0.20 on 60/90 monthlies (2 signals in ~5.7y; only Mar-2020 crisis + one thin-debit Oct-2024). Cannot host a Sharpe-2.4 sleeve alone.

### B) 15-name panel, month-start Mondays, mid (wall ~35m)

Symbols: SPY,QQQ,IWM*,AAPL,TSLA,NVDA,AMD,META,AMZN,GOOGL,NFLX,BA,AES,COIN,PLTR (*IWM no daily bars in DB — skipped).

| Book | n | WR% | meanRet% | medRet% | PF | Sharpe~ | PnL |
|------|--:|----:|---------:|--------:|---:|--------:|----:|
| mid unfiltered | 192 | 42.2 | +5.3 | **−11.1** | **0.76** | 0.19 | **−$9.2k** |
| mid FF≥0.20 | **15** | 53.3 | **+30.5** | +15.3 | **1.60** | **0.62** | +$592 |

FF filter does what the video claims directionally on **mid**: unfiltered PF&lt;1 / fat losers; FF≥0.20 mean/med/PF flip up. But **n=15**, Sharpe proxy **0.62 vs claim 2.4**, and this is still optimistic mid (no Kelly CAGR path).

Ask/bid panel full rescan and EOD remarck of the 15 FF trades both hit Theta **429 / terminal hang** after the mid pass (Terminal `:25503` ReadTimeout). SPY ask/bid already shows the friction path: unfiltered mean **−8.3% / PF 0.74**, and the rare FF≥0.20 sleeve cuts from mid +262% to ask/bid +40% (n=2). Treat panel mid FF **+30% / PF 1.6 / Sharpe~0.62** as an optimistic upper bound — still far from **27% CAGR / Sharpe 2.4**.

## Interpretation

1. **Blind calendars lose after frictions** — SPY ask/bid mean −8% / PF 0.74; panel mid unfiltered PF 0.76 / −$9k. Matches the video's "conditioning matters" baseline.
2. **FF≥0.20 is real as a mid-filter** on this liquid panel (15 trades, PF 1.6), not as a 27%/2.4 Sharpe product. Sample is far from 300k; capacity/Kelly not walked.
3. **SPY is the wrong host** for the playbook threshold; crisis spikes are too rare. Edge (if any) is single-name backwardation — still needs ask/bid + earnings skip + larger panel before any soft interest.
4. **Do not promote.** No Sharpe>1 vs SPY occupancy path; claim CAGR/Sharpe not reproduced.

## Gaps vs video (intentional)

- No ex-earnings IV / earnings calendar skip (equity panel only; SPY clean)
- No 35Δ double calendar
- No fractional Kelly / 2–8% sizing equity curve
- Month-start scan on panel (not every Monday) — fewer signals than a full hunt
- Not 2007–2025 full OPRA universe

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_forward_factor.py --start 2020-01-01 --end 2025-08-29 --symbols SPY --marks mid,ask_bid --ff-mins none,0.20,0.41 --outdir reports\research\forward_factor\2026-10-01_spy_v2
python scripts\research\claimcheck_forward_factor.py --start 2020-01-01 --end 2025-08-29 --symbols SPY,QQQ,AAPL,TSLA,NVDA,AMD,META,AMZN,GOOGL,NFLX,BA,AES,COIN,PLTR --marks mid --ff-mins none,0.20 --scan month_start --outdir reports\research\forward_factor\2026-10-01_panel
python scripts\research\_remark_ff_askbid.py
```
