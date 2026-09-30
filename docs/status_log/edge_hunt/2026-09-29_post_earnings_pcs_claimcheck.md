# Post-earnings put credit spread claim-check (2026-09-29)

**Source:** Theta Profits / Brian Terry — YouTube [H2G8_jisZP8](https://www.youtube.com/watch?v=H2G8_jisZP8) (2025-07-20).

**Verdict: interesting mechanical sketch / no promote.**

Claim is a **2–3 month personal track** (~80% WR, ~8d hold, 8–10% RoR), not a multi-year backtest. Our book uses BS marks + IV-decay, not listed option quotes.

## Claimed

| Metric | Claim |
|--------|------:|
| Win rate | ~80% |
| Avg hold | ~8 days |
| Avg return on risk | 8–10% |
| Sample | "last group" over ~2–3 months |

Rules (public interview): post-earnings **bull put credit** only; skip big drops (~10–15%); prefer flat/moderate gap-up; short ~**20 delta** below pre-earn price; width ~$5; credit ≥$0.80 on $5; **30–45 DTE**; TP **50%** credit; SL **2×** credit; OptionSlam-style prior-8-quarter hold filter.

## Ours (mechanical)

- Universe: 49 liquid names (mega/large-cap weekly-options proxies)
- Earnings: yfinance (AMC if hour≥16 else BMO)
- Stock bars: TimescaleDB ALPACA+IB prefix
- Marks: BS put vertical; entry IV = ATR-IV × 1.35, linear decay to ATR-IV over 10 sessions
- Hist filter: prior 8 events, ≥5 with close@+21 ≥ max(pre, entry) (OptionSlam stand-in)
- Wall ~1–1.5 min after OHLCV cache

Harness: `utils/research/post_earnings_pcs.py`, `scripts/research/claimcheck_post_earnings_pcs.py`  
Artifacts: `reports/research/post_earnings_pcs/2026-09-29/` (+ `_nohist`, `_noivcrush`)

### A) Hist filter + IV elev/crush (closest rule set)

| Metric | Claim | Ours | Match? |
|--------|------:|-----:|:------:|
| n | small | 178 | — |
| WR% | 80 | **92.1** | high |
| hold d | 8 | **11.5** | ~ |
| avg RoR% | 8–10 | **11.1** | ~ |
| PF | — | 5.54 | optimistic |
| TP% / SL% | — | 91.6 / 7.3 | — |

Year buckets: 2018-19 WR 100; 2020-21 97.6; 2022-23 94.5; **2024-26 WR 83.1 E_RoR 7.2 PF 2.37**.

Reasons: ok 178 / skip_hist 1094 / skip_drop 200 / skip_gap 11 / no_credit 6.

### B) No hist filter (same IV crush)

n=**652** WR **93.7** hold 10.8 avg RoR **11.8** PF **7.0** — more trades, still BS-optimistic. Gap-up-only screen is already a strong put-credit filter under decaying IV.

### C) Hist filter, **no IV elev** (`iv_elev=1.0`, crush_days=1)

| Metric | Claim | Ours | Match? |
|--------|------:|-----:|:------:|
| n | small | 166 | — |
| WR% | 80 | **79.5** | yes |
| hold d | 8 | **14.3** | no |
| avg RoR% | 8–10 | **5.3** | no |
| PF | — | 1.97 | — |

2024-26: WR **71.4** E_RoR **1.6** PF **1.20** — soft without the IV-crush gift.

## Interpretation

1. **Headline 80% WR is easy to paraphrase** once you only sell puts after non-negative post-earn gaps and manage TP50/SL2x — especially under BS. Ablation C lands ~80% WR without elevated IV.
2. **8–10% RoR on winners is nearly mechanical** at TP50: credit $0.80 on $5 wide → profit $0.40 / risk $4.20 ≈ **9.5% RoR**. Reporting winner RoR ≈ claiming “we hit TP a lot,” not a separate edge metric.
3. **Avg hold ~8d needs fast IV crush** in the mark model. Constant-IV book holds ~14d and loses the RoR band; elev+decay prints faster TPs and inflated PF.
4. **Hist filter** cuts n hard (1094 skips) and is only a coarse OptionSlam proxy (no fundamentals / liquidity / overbought screen).
5. **Not a promote:** no listed bid/ask (Polygon spot-check still needed), BS+decay is friendly to short premium after gap-ups, claim sample is tiny, occupancy/capital not walked, Sharpe-vs-SPY gate not applicable until live marks exist.

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
pip install yfinance
python scripts\research\claimcheck_post_earnings_pcs.py --symbols AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD,NFLX,AVGO,COST,CRM,ADBE,INTC,QCOM,MU,AMAT,LRCX,KLAC,SNPS,CDNS,ORCL,CSCO,IBM,INTU,NOW,SNOW,PANW,CRWD,PLTR,UBER,SHOP,COIN,BA,CAT,DE,GE,HON,UNH,JPM,GS,MS,V,MA,HD,NKE,DIS,WMT,XOM --start 2019-01-01
python scripts\research\claimcheck_post_earnings_pcs.py --symbols ... --no-hist-filter --outdir reports\research\post_earnings_pcs\2026-09-29_nohist
python scripts\research\claimcheck_post_earnings_pcs.py --symbols ... --iv-elev 1.0 --iv-crush-days 1 --outdir reports\research\post_earnings_pcs\2026-09-29_noivcrush
```

## Next (only if revisiting)

- ~~Polygon listed vertical mid/half-spread spot-check on a recent subset~~ **done 2026-09-29** — see below.

## Polygon listed vertical realism (2026-09-29)

Runner: `scripts/research/run_post_earnings_pcs_live_realism.py`  
Helpers: `utils/research/post_earnings_pcs_live.py`  
Artifacts: `reports/research/post_earnings_pcs/2026-09-29_live/`  
Wall ~17.4 min (Polygon ~4 rpm; 44 rows × 2 legs).

Recent hist+IV BS rows with entry ≥2024-10 (n=44 attempted).

| slice | n | WR% | hold d | avg RoR% | PF | notes |
|-------|--:|----:|-------:|---------:|---:|-------|
| honest mid | 25 | 44 | 6.8 | **−1.08** | **0.61** | day close as mid |
| honest half-spread | 25 | **16** | 6.8 | **−3.13** | **0.11** | TP 16% / SL **84%** |
| BS same 25 rows | 25 | 76 | 12.7 | +3.66 | 1.49 | same keys |

- Hit: status_ok 30 / slid 5 / honest 25; misses: no_credit 6, no_bars 5, no_entry_bar 3.
- **entry_mid / bs_credit med ≈ 0.32** — BS credits were ~3× live vertical mids (elevated ATR-IV overstates OTM put verticals).
- Half-spread on thin credits makes SL 2× almost automatic noise (tiny credit → tiny stop distance vs $0.05 floor haircuts).
- **Confirms no promote** harder than BS alone: listed marks reverse the optimistic BS book.

```bat
venv\Scripts\activate
set PYTHONPATH=.
set POLYGON_REQUESTS_PER_MINUTE=4
python scripts\research\run_post_earnings_pcs_live_realism.py --n 44 --min-entry 2024-10-01
```

## Open-vs-close + mid-implied 20Δ (2026-09-29)

Follow-up to “would better marks help?” — free-tier **historical snapshots are 403**, so no paid OPRA; used Polygon **day open/close** aggs and **BS IV inverted from mid** to pick ~20Δ short.

Runner: `scripts/research/run_post_earnings_pcs_open_delta.py`  
Helpers: `utils/research/post_earnings_pcs_open_delta.py`  
Artifacts: `reports/research/post_earnings_pcs/2026-09-29_open_delta/`  
Wall ~21.5 min; **12 liquid** earns (2024-10 → 2025-12); ok=9.

| credit source | med open | med close | med vs BS open | meet min* open | meet min* close |
|---------------|---------:|----------:|---------------:|---------------:|----------------:|
| BS model | — | — | (med BS **$2.28**) | — | — |
| Same BS strikes, listed | **$0.89** | $0.67 | 0.65× | — | — |
| Mid-implied ~20Δ vertical | **$1.60** | $1.10 | 0.74× | **78%** | **44%** |

\*min = 0.16 × width (Brian’s ≥$0.80 on $5).

- **Open + correct ~20Δ recovers the credit claim** often enough (med $1.60; 7/9 meet min). Prior close-only / ATR-IV-strike spot-check was the harsh case.
- Close still weaker (44% meet min); open>close only **44%** of rows — residual IV is not a guaranteed same-day crush on day bars.
- This is an **entry-credit** check only — not a managed P&L re-walk with open fills. Still **no promote** until open-entry TP50/SL2x is marked.
- Paid OPRA would refine bid/ask and true exchange deltas; the direction of the correction (open/20Δ ≫ prior close/wrong-strike) is already visible on free aggs.

```bat
venv\Scripts\activate
set PYTHONPATH=.
set POLYGON_REQUESTS_PER_MINUTE=4
python scripts\research\run_post_earnings_pcs_open_delta.py --n 12 --ladder-n 8
```

## Starter minute open / first-hour pilot (2026-09-30)

Massive/Polygon **Options Starter** unlocks option **1-minute aggregates** (live snapshot path still 400 on OCC ticker alone — not required).

Runner: `scripts/research/run_post_earnings_pcs_minute_pilot.py`  
Helpers: `utils/research/post_earnings_pcs_minute.py`  
Artifacts: `reports/research/post_earnings_pcs/2026-09-30_minute_pilot/`  
Wall ~6.2 min @ 30 rpm; **20 liquid** earns attempted (2025-04→2026-07); **ok=15** (5 no day-ladder 20Δ pick).

Entry clocks from minute prints; ~20Δ strikes reused from day-open ladder; exits = day-close TP50 / SL 2× on **mid** (no 8% half-spread — that haircut zeroed open credits on expensive legs and forced same-day SL).

| entry mark | med credit | meet min% | book n | WR% | hold d | avg RoR% | PF |
|------------|-----------:|----------:|-------:|----:|-------:|---------:|---:|
| RTH open (09:30) | **1.70** | **60** | 15 | **80** | 3.6 | **+8.44** | **3.66** |
| ~10:00 ET | 1.13 | 33 | 14 | 71 | 6.5 | +1.73 | 1.37 |
| first-hour VWAP | 1.29 | 27 | 11 | 73 | 6.4 | +8.30 | 3.54 |
| day close | 1.20 | 27 | 15 | 67 | 7.7 | +0.84 | 1.17 |

- RTH open > day-close credit on **73%** of rows (med open/close ratio **1.70×**) — COST-style open crush shows up in the liquid sample.
- RTH-open mid book lands near Brian’s WR/RoR on this tiny n (hold shorter ~3.6d vs claimed ~8d).
- **Still no promote:** n=15 mega-liquid only; trade prints ≠ NBBO; same-day TP when open ≫ close is optimistic vs real fill; half-spread / bid-ask not modeled; 2y window skips 2022 stress.

```bat
venv\Scripts\activate
set PYTHONPATH=.
set POLYGON_REQUESTS_PER_MINUTE=30
python scripts\research\run_post_earnings_pcs_minute_pilot.py --n 20
```
