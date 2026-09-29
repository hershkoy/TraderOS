# Research & data — current status

Last updated: **2026-09-27**

Working notes live under `docs/status_log/` (`edge_hunt/`, `edge_hunt/channel_touch/`, `weekly_bigvol/`, `daily/`).

---

## Stock market data (TimescaleDB `market_data`)

Stocks in `market_data`. Option chains in `option_contracts` / `option_quotes` (OptionsDX EOD). Queried via `utils.db.timescaledb_client.get_timescaledb_client()`.

| Timeframe | Symbols | Bars (approx) | Date range / freshness |
|-----------|---------|---------------|------------------------|
| Daily (`1d`) | **~2,217** | ~7.6M+ | ALPACA primary; gap-fill 2026-08-23 + nightly multi-symbol refresh. CronRunner `channel_touch_nightly` is **17:00 ET Mon–Fri** (Friday session = Friday 17:00 ET; not 16:00 ET cash close). Stale as_of is a failed run; retries +1h/+2h/+3h. 2026-09-19 Saturday catch-up after Friday scanned Thursday. |
| 15-minute (`15m`) | **1,478** | ~58.7M (EXPLAIN est.) | IB primary, ~2018-01-02 → **2025-12-02** (stale vs 2026-08-30). Catch-up now runs **16:30 ET** via `after_rth_ib_backfill` (`backfill_ib_15m_universe.py`, client 8822) then 5m. ALPACA 15m unused aside from leftover `AEO`. |
| 5-minute (`5m`) | starting (A, AAL full; rest pending) | — | IB, same symbol set as 15m. Ingest `backfill_ib_5m_universe.py` (client 8823) **year-first**: 2025-through-now all symbols, then 2024..2020. Yields to RTH (stop 09:15 ET; resume 16:30 ET after 15m catch-up; weekends continuous). Hourly watchdog restarts 5m if the after-RTH job died. Playbook: [ib_5m_backfill](../features/ib_5m_backfill.md). |
| Weekly | **0** | — | Not stored — resample from daily |

### Option chains (`option_contracts` / `option_quotes`)

OptionsDX EOD (bid/ask/last/IV/greeks). Snapshot `eod`. 120-day quote retention **removed** so this history is kept; chunks compress after 90 days. Reload: `python scripts/db/ingest_optionsdx.py`.

| Underlying | Contracts | Quote rows (with SPY/VIX) | Date range |
|------------|-----------|---------------------------|------------|
| SPY | **368,500** | **21,318,650** quotes total | 2010-01-04 → 2023-12-29 (16:00 ET) |
| VIX | **42,498** | (included in total) | same |
| QQQ | 1,032 | 0 quotes (old Polygon leftover) | — |

2024+ still Polygon-only (plan lookback ~2y). No SPX files in this drop.

### Data sources

| Source | Role | Persist? |
|--------|------|----------|
| **Alpaca** | Historical OHLCV (primary daily) via `utils/data/fetch_data.py` / `update_universe_data.py`; IEX default; nightly `--multi-symbol` batches | Yes — provider `ALPACA` |
| **IBKR** | Historical OHLCV (primary 15m; 5m ingest starting 2026-09-04); Gateway ~4001 | Yes — provider `IB` |
| **TradingView** | Chart / Pine verify only (workspace MCP, CDP 9222) | No |
| **OptionsDX** | Historical EOD option chains (SPY, VIX 2010–2023) via `scripts/db/ingest_optionsdx.py` | Yes — `option_quotes` snapshot `eod` |

### Ops notes

- Full-table heavy `COUNT`/`GROUP BY` on `market_data` is slow; prefer per-timeframe/provider queries and `EXPLAIN` estimates.
- `ticker_universe` ~6,072 tickers; daily and 15m sets are subsets.
- For 15m universe backtests prefer the IB 15m symbol set over the full daily universe.
- ALPACA daily panel coverage for many names starts ~2020–2022; compute SPY SMA on full IB SPY history then align to the panel index.
- Gap-fill detail: [daily/2026-08-23](daily/2026-08-23.md) (and archived plan in that folder).
- Alpaca last snapshots (`fetch_alpaca_last_prices`): alpaca-py 0.13 default read timeout is None; live path binds 5s/15s and retries 10054. Minute proximity is a hub backup (skip if last_price_ts younger than 30s). See [Alpaca snapshot timeout](edge_hunt/channel_touch/2026-09-15_alpaca_snapshot_timeout.md).

---

## Research status (2026-09-19)

### Edge hunt (SPY-beating portfolio sleeves)

Status: **`docs/status_log/edge_hunt/`**

| Phase | Verdict |
|-------|---------|
| 4 | **KEEP** `Blend_SPY70_BV30` |
| 5 | No promote (XS mom + swing MR). 2026-09-13 2022-only BB `mean_reversion`: unconstrained n=1041 E −0.16 PF 0.95; occ-15 CAGR +11.4% Sharpe 0.59 (fails Sharpe>1). Not a last-15m rescue. See [mean reversion 2022](edge_hunt/2026-09-13_mean_reversion_2022.md). 2026-09-13 choppy-regime Connors RSI-2 + 3-day pullback **no promote** (first book SMA200/VIX>30/fric0.25: RSI-2 n=5451 E +0.46 PF 1.36, occ-15 PF 0.96, 2022 n=10; refine SMA50/no VIX skip/fric0.10: SMA50 occ-15 PF 1.30, 2022 n=194 E +1.08; confirm RSI cross-back occ-15 fric0.10 PF 1.21 vs threshold 1.30, max concurrent 185→110; pullback close-loc0.7 2023 PF 0.49). See [choppy regime](edge_hunt/2026-09-13_choppy_regime.md), [refine](edge_hunt/2026-09-13_choppy_regime_refine.md), [confirm](edge_hunt/2026-09-13_choppy_regime_confirm.md). |
| 6 | No full-sample promote; vol-target near-miss |
| 6b | Near-KEEP `Blend_VT60_BV40` (VT SPY + BigVol) — prefer freeze/productionize over more Sharpe>1 fishing |
| 6c | No promote / no lift vs plain vol-target (VIX curve) |

Promotion gate (vs SPY): Sharpe > 1.0, Sharpe ≥ SPY, MDD better than SPY (CAGR reported, not ranking objective).

### HA + 55 EMA + MACD swing (Animesh / Dhan)

Status: [2026-09-19_ha_ema_macd_swing](edge_hunt/2026-09-19_ha_ema_macd_swing.md). Strategy: [README](../strategies/ha_ema_macd_swing/README.md). Causal daily HA break above EMA(55) High + MACD>0; next-open fill; stop EMA Low; 3R / Friday / MACD-signal cross. ALPACA 1d unique **n=4047 E +0.23 PF 1.20** geo 1.26; **75% Friday exits**, median hold **1 day**; occ-15 Sharpe **0.51** vs SPY 0.67; fric0.25 E −0.02 PF 0.99. Weekly resample n=634 E +2.64 PF 1.55 is a 2023 leftover (2022 n=2). **No promote.**

### Weekly BigVol + TTM Squeeze

Status: **`docs/status_log/weekly_bigvol/`** — Phase 1 TV verify + Phase 2 edge smoke done. Research execution default: no fixed TP + ~10% stop (+ MA exit). Scanner `squeeze` is TTM zero-cross only.

### VCP AVWAP (research translation)

Status: [2026-09-13_vcp_avwap](edge_hunt/2026-09-13_vcp_avwap.md). Causal 15m signal-close buys + N+1 mid sells on the old Playbook #1 detector. 300 ADV names **n=2071 E +0.55 PF 1.26 geo 1.27**; fric0.25 E +0.30 PF 1.13. Calendar **2022 n=198 E −1.40 PF 0.38** (the green 2022-23 bucket **E +0.92 PF 1.46** is 2023 n=295 E +2.47). 78% hard-stop, median −2.0, drop-top-3 E +0.21, max concurrent 55. **No promote.** Do not use as a 2022 last-15m H2 switch.

### SMB LOD / 9 EMA x VWAP reclaim

Status: [2026-09-13_vwap_ema_reclaim](edge_hunt/2026-09-13_vwap_ema_reclaim.md), [filters](edge_hunt/2026-09-13_vwap_ema_reclaim_filters.md). 15m next-mid buys + same-session N+1 mid measured-move sells, fric 0.10. Top 300 ADV **n=227915 E +0.005 PF 1.02**; fric0.10 **E −0.10 PF 0.67**. Signal-close analog n=245796 E +0.009 PF 1.03 / fric E −0.09 PF 0.72. Max concurrent 170. Realized target +0.44% vs stop −0.59% (not 3:1). Selectivity: daily top1 n=1959 E −0.11 PF 0.85; unit>=5 n=867 E +0.01 PF 1.01 geo 0.92 (drop-top-3 red). **No promote.**

### Choppy-regime RSI-2 MOC + 3-day pullback

Status: [2026-09-13_choppy_regime](edge_hunt/2026-09-13_choppy_regime.md), [refine](edge_hunt/2026-09-13_choppy_regime_refine.md), [confirm](edge_hunt/2026-09-13_choppy_regime_confirm.md). Connors-style daily (not Phase 5 SMA20 RSI-2). First book SMA200 + VIX>30 + fric 0.25: unique **n=5451 E +0.46 PF 1.36**; occ-15 **PF 0.96**; 2022 n=10. Refine SMA50, no VIX sit-out, fric 0.10, pullback time-stop 6: SMA50 occ-15 **n=2082 E +0.37 PF 1.30**; calendar 2022 **n=194 E +1.08 PF 2.57**. Extreme RSI<2 no-MA occ-15 2022 **PF 0.95**. Confirm: RSI cross-back occ-15 fric0.10 **PF 1.21** (max concurrent 185→110, median +0.61 vs threshold +1.15); pullback close-loc0.7 2023 **PF 0.49**. Live RSI-2 scan uses IB 15m 15:45 developing close (not Alpaca EOD). **No promote.** Do not occupancy-walk onto last-15m.

### Deep Pullback SPY (live selective sleeve)

Status: [universe](edge_hunt/2026-09-17_deep_pullback_universe.md). Strategy: [README](../strategies/deep_pullback/README.md). Playbook: [deep_pullback_spy](../features/deep_pullback_spy.md).

QS rules on **SPY**: 15d lowest close + 10d lowest 1d return + close > SMA200; exit close > prior high. Entry is **EOD/MOC**, not an intraday chase. Yahoo ~1993–2026 rebuild **n=127 WR 75.6% E +0.74% PF 2.99** CAGR 2.82% exp 5.9% MDD 14.0% — **largely replicates** the QS claim (n=128 WR 78% PF 3.0 E 0.8% CAGR 3% MDD 14%). Friction 0.25% RT softens PF to ~2.1. HTML: `reports/research/deep_pullback/deep_pullback_spy.html`.

Universe TimescaleDB scan (306 names, fric 0, 2006+): all **n=6156 E +0.75 PF 1.59** geo 1.68; indexes n=484 E +0.47 PF 1.96; 2022-23 n=1280 E +0.26 PF 1.16. **Not a Sharpe>1 promote** — small selective sleeve (~95% cash).

**Live `/hot` Detector DP** (`timeframe=dp`; CronRunner `deep_pullback_spy` ~5m RTH, every minute **15:45–16:00** ET). Developing armed uses Alpaca last as today’s close proxy. Telegram **approaching** from **15:45** with MOC target (`moc_max` on Resist; SMA200 on Support); **BUY NOW** at **15:57** only if still green (Arca MOC locks 15:59 — leave ~2 min). Code: `utils/scanning/deep_pullback_spy.py` / `scripts/scanners/deep_pullback_spy.py`.

### QS RSI+IBS Nasdaq-100 stocks (research claim-check)

Status: [2026-09-27](edge_hunt/2026-09-27_qs_rsi_ndx_claimcheck.md). Video tjL4wcI403E / Substack 2026-09-14.
Public rules: SMA200 + RSI(3)&lt;20 + IBS&lt;0.3, QS exit, next-open, max 5.
Ours (current NDX, panel ~2018-11→2026-09) occ-5 fric0: **n=1185 E +0.38 PF 1.26** WR 62% CAGR 10.6% exposure **57%** MDD 26% Sharpe **0.53** vs claim n=1321 E +1.1 PF 2 WR 68% CAGR 11% exp 16% MDD 17%.
fric0.25: E **−0.12** PF **0.93** CAGR **−5%**. **No promote** (does not replicate; friction kills).

### ROC Rebound SPY (research claim-check)

Status: [2026-09-22](edge_hunt/2026-09-22_roc_rebound_spy.md). Strategy: [README](../strategies/roc_rebound/README.md).

QS rules: buy next open when 7d ROC closes below −3%; sell next open when ROC closes above +3%; no stop/TP. Yahoo ~1993–2026 fric 0 **n=147 WR 71.4% E +1.65% PF 2.40** eq 9.11 MDD 29% — **replicates** claim (n=146 WR 71.2% PF 2.40 eq ~9.15 MDD ~29%). All three eras PF > 2. Friction 0.25%: PF 2.12. **No promote** (thin n, large MDD, in-sample). Code: `utils/research/roc_rebound.py` / `scripts/research/backtest_roc_rebound.py`.

**Universe sample (2026-09-22):** seed 42, 50 names (12 ETF / 19 large / 19 small) with RT slippage 0.10 / 0.25 / 0.40%. Pooled PF **1.54 / 1.34 / 1.12** — edge thins off SPY; **no universe promote**. Status: [universe](edge_hunt/2026-09-22_roc_rebound_universe.md). Runner: `scripts/research/run_roc_rebound_universe.py`.

**ETF overlay grid (2026-09-22):** 18 ETFs, fric 0.10%. **A3 SMA200+vol-scale** n=781 PF **2.38** E +1.42 MDD -32 (vs A0 1.69 / +1.04 / -66); eras PF>2; drop-top-3 holds. Time-stops no promote. **Soft-interesting** research sleeve, not live-wired. Status: [etf grid](edge_hunt/2026-09-22_roc_rebound_etf_grid.md).

**Walk-forward (2026-09-22):** Expanding OOS 2005–26 frozen A3 **n=694 PF 2.34**; holdout 2016+ PF **2.49**; IS always picks A3. Fragile 2008/2011/2020. Verdict **SOFT_PASS_OOS_FRAGILE_YEAR**. Status: [walk-forward](edge_hunt/2026-09-22_roc_rebound_walkforward.md).

**Single-slot vs SPY (2026-09-22):** Holdout 2018–26 occ n=56 PF **1.20**; Sharpe 0.12 vs SPY 0.72; MDD worse than SPY. **FAIL** Phase-5 gates — pooled PF~2.3 does not survive max_open=1. Status: [occupancy](edge_hunt/2026-09-22_roc_rebound_occupancy.md).

**Capital fixes (2026-09-22):** time-stop 7 is the main lift; best B9 (regime+tstop+4slot+Rf4%) Sharpe **0.93** MDD −9% beats SPY but **FAIL** Sharpe>1. Verdict **SOFT_IMPROVE_NO_GATE**. Status: [capital fix](edge_hunt/2026-09-22_roc_rebound_capital_fix.md).

### QS Five Swing SPY (live selective sleeve)

Status: [claim-check](edge_hunt/2026-09-18_qs_five_swing.md), [5m/15m realism](edge_hunt/2026-09-18_qs_five_swing_realism.md), [1m MOC clocks](edge_hunt/2026-09-18_qs_five_swing_1m_moc.md). Strategy: [README](../strategies/qs_five_swing/README.md).

Five short-term MR sleeves on **SPY** (S1 Band+IBS, S2 Turnaround Tuesday, S3 5-Day Low, S4 Vol+ADX, S5 10d High Reject), combined **single-slot** S1→S5; shared QS exit (close > prior high). Yahoo 1993–2026 fric 0.03 **combined n=958 E +0.39 PF 1.93** CAGR 11.31% exp 40.3% MDD 23.8% vs claim n=975 E 0.38 CAGR 11.1% exp 41% MDD 23% — **largely replicates**. Friction 0.25% RT: CAGR ~4.5% PF ~1.36. HTML: `reports/research/qs_five_swing/qs_five_swing_spy.html`.

**Fill realism (2018–2026):** oracle MOC is the upper bound (n=256 E +0.36 PF 1.78 CAGR **10.86%**). IB **1m @ 15:57 ≈ 15:58** recover **~10.4–10.6%** CAGR (prefer **15:57** for live headroom). Honest **5m@15:55** is much weaker (~7.5%). Do **not** treat **15m@15:45** as an early decision (that bar already spans into the close). Next-open keeps ~10% CAGR but MDD **−29%**. **Not a Sharpe>1 promote.**

**Live `/hot` Detector QS** (`timeframe=qs`; CronRunner `qs_five_swing_spy`; same MOC funnel as DP). Code: `utils/scanning/qs_five_swing_spy.py` / `scripts/scanners/qs_five_swing_spy.py`. Research: `utils/research/qs_five_swing.py`. SPY IB 1m candidate-day pulls: `scripts/research/backfill_ib_1m_symbol.py` (client **8825**).

### Channel-touch / ascending channels (active production track)

Status: **`docs/status_log/edge_hunt/channel_touch/`**

**Latest (2026-09-23):** 3x MOC DCA on current_best last-15m (signal day + next 2 closes; pause below rail; ATR stop during scale-in). Allocated n=2295 E **+0.03** PF **1.01** geo **1.03** vs baseline +0.27 / 1.09 / 1.10; deployed E **−0.37**; 2022 still red. **No promote.** See [MOC DCA](edge_hunt/channel_touch/2026-09-23_last15m_moc_dca.md).

**Prior (2026-09-19):** Causal last-15m long ↔ short H3+ regime switch (WR/PF, TLT, UUP, SPY SMA). Same sensors as the L3 switch; WR leftover 2022 still red; PF20 greens 2022 only by becoming ~82% short; macros leave 2023 red or blunt 2020-21. **No promote.** See [short regime switch](edge_hunt/channel_touch/2026-09-19_last15m_short_regime_switch.md).

**Prior (2026-09-19):** Portfolio occupancy overlay of current_best last-15m long H2 + 15m short H3+. Concat flattens 2022 only by short n-weight (117 long / 1031 short). Equalized fric0.25 keeps 2022 red (UNIFIED 15 E −0.17). **No promote.** See [last15m + short blend](edge_hunt/channel_touch/2026-09-19_last15m_resist_short_blend.md).

**Prior (2026-09-16):** Short H3+ resistance fade after causal H2 (cover support + ATR k=2). Daily unique span365 **n=250 E +0.06 PF 1.02** — **no promote**. 15m unique span10 **n=7004 E +0.21 PF 1.47** geo 1.45 (all years +; drop-top-50 holds; fric0.25 fragile E +0.06) — research-interesting, **not a hard promote**. See [resist short H3](edge_hunt/channel_touch/2026-09-16_resist_short_h3.md).

**Prior (2026-09-16):** Causal consecutive-loss sit-out on last-15m n=2297 (L5/S3/P2, L5/S3/P3, dollar ≤−15). Diagnostic: 2022-23 is **not** longer-tailed vs 2020-21 (max 21 vs 26). Locked L5/S3/P2 lifts geo to 1.20 but leftover 2022 **E −2.62** (worse); 2022-23 stays red. **No promote.** Do not fish L/S/P. See [loss streak](edge_hunt/channel_touch/2026-09-16_last_15m_loss_streak.md).

**Prior (2026-09-15):** Alpaca snapshot `read timeout=None` / WinError 10054 was Telegram-ing `Channel-touch 15m FAILED` every hung poll. Live fetch now 5s/15s timeout + retry; minute proximity skips if hub `last_price_ts` is younger than 30s; flaky Alpaca does not abort IB fills. See [Alpaca snapshot timeout](edge_hunt/channel_touch/2026-09-15_alpaca_snapshot_timeout.md).

**Prior (2026-09-14):** Pine VX2/VX1 slope < 4 on last-15m, **ratio in Python** from TV daily dumps (`CBOE_DLY:VX1!` / `VX2!`, paged 2017→2026). **VX12_lt4** n=1645 E **−0.00** PF **1.00** geo **1.00** vs +0.27 / 1.09 / 1.10; 2022 leftover n=62 E **−2.45**; 2020-21 blunt; 2023 leftover worse. **No promote.** Do not fish 5/8. See [VX curve](edge_hunt/channel_touch/2026-09-14_last_15m_vx_curve_skip.md).

**Prior (2026-09-13):** Occupancy-walked last-15m H2 + 15m L3: unified FCFS 8/15/20 vs dedicated 15+5 / 12+3 / 8+4. Unified **15 keeps 76% of L3** (not a lockout); cap 8 starves **H2** (1705 cap skips — last-15m arrives late). Dedicated 15+5 keeps L3 (1706/1744, 5 sleeve_full). **2022 stays red** on every walk (unified 15 E −0.65; dedicated 15+5 −0.66). Pooled E +0.54 is mixed-scale + H2 crowding. **No promote.** See [blend occupancy](edge_hunt/channel_touch/2026-09-13_h2_l3_blend_occupancy.md).

**Prior (2026-09-13):** Unconstrained last-15m H2 + 15m L3 wait-12 overlay on one timeline: **max concurrent 98 unique names** (calendar 102). H2 alone already 97 / 100; L3 max is **7** (median hold 0.81d vs H2 15.75d). 2022 overlay peak **29** is still H2 27 + L3 4. Occupancy not re-walked; no pooled P&L. **Not a promote.** See [H2+L3 concurrent](edge_hunt/channel_touch/2026-09-13_h2_l3_concurrent_overlay.md).

**Prior (2026-09-13):** Sit 2022 out vs TLT as a last-15m **global gate**. Oracle cash-skip 2022 is n=2180 E +0.41 PF 1.13 geo 1.15 (peeking) and **2023 stays −0.33 / 0.90**. TLT<SMA100 does skip 113/117 of 2022 but is off **1502/2297** fills (2019-10 → 2026-08): leftover 2023 n=99 E **−1.69**, 2020-21 **E −0.65 PF 0.81 blunt**, geo 1.02 vs 1.10. SMA200 / 63d ROC same. **No promote.** See [TLT sit-out](edge_hunt/channel_touch/2026-09-13_tlt_global_gate_sitout.md).

**Prior (2026-09-13):** Causal last-15m H2 vs 15m L3 switch (rolling paper WR/PF, TLT below SMA, UUP above SMA). WR20 is off half the sample and leftover 2022 is **worse** (−2.82). TLT<SMA100 tags 2022 then **blunts 2020-21**. UUP>SMA50 loses pooled PF; SMA100 leaves 2023 red. **No promote.** See [regime switch](edge_hunt/channel_touch/2026-09-13_last_15m_regime_switch.md).

**Prior (2026-09-13):** Last-15m next-session MOO without chasing an overnight gap, else same-session 15m pullback holding the rail. Chase-all-opens n=2297 E **−0.01** PF 1.00; skip-gap n=553 E −0.02; pullback n=1236 E +0.16 PF 1.05 vs last-15m +0.27 / 1.09. 2022 worse. **No promote.** See [open no-chase](edge_hunt/channel_touch/2026-09-13_last_15m_open_no_chase.md).

**Prior (2026-09-13):** AND last-15m H2 with Deepvue weekend take-4. Trade-week overlap **n=1** (MRNA 2021-06-22 last-15m −6.29%; Deepvue +86% that week). Signal-day n=0. Ever-on-list n=161 wrecks 2020-21 (E −1.31 PF 0.66). Coiled-near-MA vs already-above-rail almost never meet. **No promote.** See [Deepvue AND](edge_hunt/channel_touch/2026-09-13_last_15m_deepvue_and.md).

**Prior (2026-09-12):** Causal Deepvue-style weekend scan (QQQ rising 21 EMA, RS 85, RMV tightness, max 4) with next-session 09:30 15m-mid buys and 15m N+1 mid sells. 300 ADV names **n=179 E +2.25 PF 1.85 geo 0.985**; median **−2.94**; 2022-23 **E −2.16 PF 0.17**. Fat tails / 83% hard-stop. **No promote.** HTML: `reports/weekly_deepvue/2026-09-12/weekly_deepvue.html`. See [weekly Deepvue](edge_hunt/2026-09-12_weekly_deepvue.md).

**Prior (2026-09-12):** Universe SMA20 breadth + last-15m RTH exhaustion skips on n=2297. B50 turns 2020-21/2024-26 red (blunt winner$). X_d1 pooled E +0.35 vs +0.27 but 2022 leftover **−2.71**. **No promote.** 15m L3 wait-12 stays the hostile-regime book; productionizing it is a live-path change (H5 today), not a 1d skip. See [breadth/exhaustion](edge_hunt/channel_touch/2026-09-12_last_15m_breadth_exhaustion.md).

**Prior (2026-09-12):** Causal prior-session VIX backwardation + expanding 2/3-state HMM skip on last-15m **n=2297**. VIX_back makes 2022 worse (−2.69 vs −2.37). HMM2 leftover 2022 is bear-rally chases (n=24 E −3.71). HMM3 zeros 2022 by also wrecking 2020-21. Macro HMM2 geo 1.11 vs 1.10, 2022 still −2.14. **No promote.** Same failure mode as SPY SMA. See [HMM/VIX skip](edge_hunt/channel_touch/2026-09-12_last_15m_hmm_vix_skip.md).

**Prior (2026-09-12):** TARS-style H2 support-cancel then later resist-break on the last-15m symbol set (8909 span365 H2s). **6330** cancel before fill; **28.6%** later break the same rails (shallowness does not lift that rate). TARS-like (shallow + recovered) **239 / 679**. Last-15m + ATR sleeve n=232 E +1.15 PF 1.38 geo 1.21 — 2020-21 fails, drop-top-3 E +0.41; TARS would-have is **−5.5%** (Jan-5 hard-stop, not the melt-up). Occupancy not re-walked. **No promote.** See [cancel reclaim](edge_hunt/channel_touch/2026-09-12_h2_cancel_reclaim.md).

**Prior (2026-09-12):** Yellow-below-support area / channel area (L1 through buy, daily lows) on the last-15m 1d book **n=2297**. Spearman vs gain **+0.013**, point-biserial vs loser **−0.008**; loser rate flat ~66–70% across quintiles. Drop-top-quintile n=1999 E +0.32 PF 1.10 geo 1.12 vs +0.27 / 1.09 / 1.10; winner $ 922 vs loser $ 957; 2022-23 still red. **No promote.** See [outside area](edge_hunt/channel_touch/2026-09-12_channel_outside_area.md).

**Prior (2026-09-12):** Clip-exit hot-cross left `current_best/`. 1d slot is last-15m-open-mid + 15m N+1 mid **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01). 2022-23 fails. **Not a promote.** Hot-cross HTML is `1d_unrealistic/1d_hot_cross.html`. See [slot swap](edge_hunt/channel_touch/2026-09-12_channel_touch_1d_hot_cross_out_of_current_best.md).

**Prior (2026-09-12):** `current_best` 1d hot-cross kept-sell is the daily occupancy clip. Honest overlay (occupancy not re-walked): **15m N→N+1 mid n=4079 E −0.83 PF 0.77**; **EOD → next 09:30 mid n=4078 E −0.65 PF 0.83** vs published **n=4079 E −0.19 PF 0.94**. See [hot-cross realistic sells](edge_hunt/channel_touch/2026-09-12_channel_touch_hot_cross_realistic_sells.md).

**Prior (2026-09-12):** IB historical top-of-book puller for AMPL/VST/TARS last-15m losers (not L2 — IB has no hist DOM). Gateway **2110** TWS-server broken; qualify Error 200 / CCP 10159. Reconnect-to-IB, then `pull_ib_historical_l1.py --smoke --no-ticks`. **No promote.** See [IB L1](edge_hunt/channel_touch/2026-09-12_ib_historical_l1_ampl_vst_tars.md).

**Prior (2026-09-12):** Synthetic oracle splice: last-15m 1d H2 except calendar 2022 = 15m L3 wait-12. n=2423 E +0.37 PF 1.13 geo 1.16 vs +0.27 / 1.09 / 1.10. 2022-23 bucket still −0.16 / 0.92 (2023 last-15m stays). No switch signal. Occupancy not re-walked. **Not a strategy. No promote.** See [synthetic 2022 L3](edge_hunt/channel_touch/2026-09-12_synthetic_2022_l3_splice.md).

**Prior (2026-09-12):** 2022-23 on last-15m H2 is a failed-breakout regime, not a missing overlay. 2022 n=117 E −2.37 PF 0.41; 2023 n=300 E −0.33 PF 0.90. WR collapses, hard-stops 55%. Prior-session SPY SMA200/20d **worsen** 2022 (bear-rally chases). 15m L3 wait-12 is already green there (2022 +0.05 / 1.09, 2023 +0.30 / 1.56). **No promote** of last-15m H2 or SPY gates. See [2022-23 regime](edge_hunt/channel_touch/2026-09-12_last_15m_2022_23_regime.md).

**Prior (2026-09-12):** Isolated follow-ups on the same 2297 last-15m + 15m N+1 mid book. Doji-only (70 fills) is flat (E +0.26 PF 1.08 geo 1.10). Failed-breakout (1451 fills) catches AMPL/VST but clips winner $ (−3457 vs loser +3256) and drops E/geo. Session-OHLC 2× seller-55% (1543 fills) lifts geo to 1.18 but wrecks winners the same way overlay A did; AMPL still missed; 2022-23 still −0.14 / 0.94. Occupancy not re-walked. **No promote.** See [follow-up overlays](edge_hunt/channel_touch/2026-09-12_last_15m_followup_overlays.md).

**Prior (2026-09-11):** Last-15m volume-delta / geometry overlays on the 2297-trade 15m-next-mid book. Overlay A (Evening Doji Star or 2 seller sessions) n=2297 E +0.46 PF 1.19 geo 1.30 vs +0.27 / 1.09 / 1.10 — clips winner $ (−2129 vs loser +2569); 2022-23 still −0.27 / 0.90. Form 0.25 skip hurts (E +0.13). Pos 1.25/1.50 mild. Delayed 2nd-close 13 fills E −2.82. AMPL's crash is a gap; textbook doji/seller rules do not fire first. **No promote.** See [volume geometry](edge_hunt/channel_touch/2026-09-11_last_15m_volume_geometry.md).

**Prior (2026-09-11):** Rebuilt last-15m HTML after the LAUR detector fix. Unique signal-close span365 **n=2310 E +2.50 PF 2.03**. Last-15m + 15m N+1 mid **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01). LAUR 2021 fill is 2021-09-10 @ 16.82 (Mar-4 rails); Oct 28 gone. HTML: `…165800.html`. **No promote.** See [LAUR false shakeout](edge_hunt/channel_touch/2026-09-11_laur_false_shakeout.md).

**Prior (2026-09-11):** Last-15m-open-mid kept-sell was the daily same-bar stop clip (WVE 2023-12-06 buy 6.85 / next-day 6.0254). Realistic sells, occupancy not re-walked: **15m N→N+1 mid n=2779 E +0.35 PF 1.11** (fric0.25 E +0.10 PF 1.03); **daily-close → next 09:30 15m mid n=2778 E +0.47 PF 1.14** (fric0.25 E +0.22 PF 1.06). vs kept-sell n=2779 E +0.96 PF 1.29. 2022-23 fails. **No promote.** See [realistic sells](edge_hunt/channel_touch/2026-09-11_channel_touch_last_15m_realistic_sells.md).

**Prior (2026-09-07):** H2 close-confirm (15m close > daily rail, fill confirm close) + formation containment ≤0.25: full unique span365 **n=1947 E +0.24 PF 1.07** (2022-23 fails). Slightly above hot-cross −0.19/0.94; **no promote.** ADM Nov-23 dropped (formation_beyond 1.88). See [close-confirm formation](edge_hunt/channel_touch/2026-09-07_channel_touch_h2_close_confirm_formation.md).

**Prior (2026-09-07):** Close-cross MAE ridge (year split, train < 2023) on 50 then 300 names: confirm-bar features do not beat unconstrained 10/18 trail as a stop (300 holdout trail n=712 E +1.28 PF 1.33 vs ridge stop E −0.11 PF 0.97). No promote. See [MAE ridge](edge_hunt/channel_touch/2026-09-07_channel_touch_close_cross_mae_ridge.md).

**Prior (2026-09-06):** 1d `--intraday-trigger close-cross` (15m close > daily rail, fill next 15m mid) + `--trail-mae` on RDWR only: 6 fills, Jun-13 never prints (Jun-5 11:15 ET already filled). Diagnostic, not a promote. See [close-cross MAE](edge_hunt/channel_touch/2026-09-06_channel_touch_close_cross_mae.md).

**Prior (2026-09-06):** Detect-once exit sweep is a CLI option (`--stop-pct-sweep 0.02,0.03,0.04` or `--atr-stop-mult-sweep 1,1.5,2`): one OHLCV load, one detect/fill pass per symbol, occupancy walk per stop. Not a promote. See [exit sweep](edge_hunt/channel_touch/2026-09-06_channel_touch_exit_sweep.md).

**Prior (2026-09-06):** `current_best/` 1d slot is live-executable buy-now hot-cross (even though it loses): unique span365 **n=4079 E −0.19 PF 0.94** (`1d_hot_cross.html` / `1d_channel_touch.html`). That is the number to beat. 15m keeper stays L3 wait-12 signal-close **n=1744 E +0.13 PF 1.22**. Clip HTML stays in `1d_unrealistic/`. See [hot-cross in current_best](edge_hunt/channel_touch/2026-09-06_channel_touch_current_best_hot_cross.md).

**Prior (2026-09-06):** Honest 1d next-mid + shakeout (`_as_session_date`) unique span365 **n=1986 E +2.24 PF 1.89** (pre-fix HTML was n=1239 / +2.27 / 1.92). Cap-2 wait lift to +3.01 / 2.25 was a pre-fix artifact (honest cap-2 n=1395 E +2.29 PF 1.91). Post-EOD overlay, not the `current_best` 1d slot. See [next-mid rebuild](edge_hunt/channel_touch/2026-09-06_channel_touch_next_mid_session_date.md).

**Prior (2026-09-06):** Clip 1d HTML left `current_best/` (EOD close-confirm then buy the rail/wick). Files in `reports/ascending_channels/1d_unrealistic/`. Hot-cross later filled the 1d `current_best` slot as the live-executable (losing) baseline. See [1d out of current_best](edge_hunt/channel_touch/2026-09-06_channel_touch_1d_out_of_current_best.md) and [realistic purchasing](../features/realistic_purchasing.md).

**Prior (2026-09-05):** 1d next-mid joined IB 15m on the **wrong session** until `_as_session_date` (midnight UTC stayed the calendar date; converting to New York dropped Mondays and used the prior day Tue–Fri). RDWR 2025-06-13 BUY **24.64** is Thursday's 10:30 mid blend, not Friday's open. Honest Friday next-mid **wild-cancels**. HTML n=1239 / crowding n=933 are **pre-fix** until rebuilt. Nightly never used this join. Same-bar rail clip is **unrealistic**, not optimistic. See [session date](edge_hunt/channel_touch/2026-09-05_channel_touch_session_date.md) and [realistic purchasing](../features/realistic_purchasing.md).

**Prior (2026-09-05):** `/hot` live Alpaca quotes and SELL NOW Telegram moved to always-on `hot_price_server.py` (`backTraderTest\HotPriceHub`, **:5001**). ChartingServer `:5000` is HTML/REST only. See [price WS](edge_hunt/channel_touch/2026-09-05_channel_touch_hot_price_ws.md).

**Prior (2026-09-05):** 1d H2 A/B of `--touch-error-pct` 1.2 vs 0 and same-bar close-fill vs `--realistic-fill-mode next-open`. Unique span365 + shakeout: B12 close **n=4558 E +2.62 PF 2.10** (keeper); tick-above **n=5649 E +1.38 PF 1.49**; next-open 1.2 **n=4894 E +0.45 PF 1.14** (2020-21 E −0.07). RDWR Jun 9 next-open 24.525 works on that name; the book does not. **No promote.** See [next-open A/B](edge_hunt/channel_touch/2026-09-05_channel_touch_next_open_ab.md).

**Prior (2026-09-05):** 1d `--realistic-fill-mode open-cross` (first 15m **open above resist**, buy that close). Unique span365 + shakeout **n=2845 E +2.06 PF 1.775** — more fills than next-mid n=1239, worse E/PF. RDWR 2025-06-13 **25.71 @ 09:45 ET**. **No promote.** See [open-cross](edge_hunt/channel_touch/2026-09-05_channel_touch_open_cross.md).

**Prior (2026-09-05):** `/hot` **Bought** column + tab tracks live ATR k=2 (1.5%–6%) + 10% trail from Alpaca last; stop hit fires **SELL NOW** (browser + Telegram). See [Bought tab](edge_hunt/channel_touch/2026-09-05_channel_touch_hot_bought.md).

**Prior (2026-09-04):** `current_best/15m_channel_touch.html` replaced with wait-12 **signal-close + `--touch-error-pct 0`** (**n=1744 E +0.13 PF 1.22**; RTH entry/exit clocks; IB SPY overlay from 2018-11-06). Beats prior next-mid freeze and touch 0.24 signal-close; 2018-19 repaired. Prefer `--touch-error-pct 0` for 15m L3; do not set `--error-pct 0`. See [touch 0](edge_hunt/channel_touch/2026-09-04_channel_touch_touch_error_0.md). Other `current_best` 15m/1d books still on earlier next-mid realistic fill until regen. Nightly scanner still close fills.

**Prior (same day):** 15m L3 wait-12 A/B of touch 0 vs 0.24 under signal-close; signal-close default vs next-mid (wait-1 worse); no-buy-below; IB 5m ingest; realistic-fill `current_best/` next-mid. See [signal-close](edge_hunt/channel_touch/2026-09-04_channel_touch_signal_close.md), [no buy below](edge_hunt/channel_touch/2026-09-04_channel_touch_no_buy_below.md), [realistic fill](edge_hunt/channel_touch/2026-09-04_channel_touch_realistic_fill.md), [5m backfill](edge_hunt/channel_touch/2026-09-04_ib_5m_backfill.md).

**Prior (2026-09-03):** full-universe **causal** rescan. Live daily H2 span365 unique-symbol **n=3364 E +2.40 PF 2.02** (was leaky n=2253 E +2.80 PF 2.21). 15m H5 + overshoot p80 + vol≥2 **n=8212 E +0.92 PF 3.57**. Full IB 15m H2 RS top1 **n=1770 E +0.21 PF 1.33** — still do not promote 15m H2. AAPL 1d 2021-11-30 is in the live book. See [causal full universe](edge_hunt/channel_touch/2026-09-03_channel_touch_causal_full_universe.md).

**Prior (2026-09-02):** causal **walk-replay** vs full-series batch (`replay_channel_touch_walk.py`). AAPL 1d H2 resist-break now **matches 3/3** (batch 0.1s): freeze first H2 (`causal_h2`) and drop L3 support-tag fills *before* occupancy (`h2_resist_break_only`). The extra 2021-11-30 fill was blocked by a phantom L3 tag, not missing rails. See [walk-replay](edge_hunt/channel_touch/2026-09-02_channel_touch_walk_replay.md).

**Prior (2026-08-30):** live nightly is still **daily H2 resist-break** (unique-symbol/day). Full IB 15m unique-symbol H5 (`volume_rel>=1`) **n=39442 WR 37% E +0.34 PF 1.63**. Tight stack **H5 + overshoot train p80 + vol>=2**: **n=6306 WR 48.8% E +0.93 PF 3.61** (drop-top-1% PF 3.22; all year buckets +). Research live loop (armed watchlist in TimescaleDB + Alpaca minute proximity + `/hot` dashboard; **not** nightly, **not** TV alerts): [15m live](../features/channel_touch_15m_live.md). IB 15m still ends **median 2025-11-28** (1,478/1,479 names have no 2026 bars) until `backfill_ib_15m_universe.py` is run. See [H5 refine](edge_hunt/channel_touch/2026-08-30_channel_touch_15m_h5_refine.md), [15m monitor](edge_hunt/channel_touch/2026-08-30_channel_touch_15m_live_monitor.md), and [hot dashboard](edge_hunt/channel_touch/2026-08-30_channel_touch_15m_hot_dashboard.md).

| Item | State |
|------|--------|
| Detector | Classical Edwards/Magee; Pine `indicators/pine/ascending_channel_3touch.pine` (**v1 unchanged** — quality via post-filters, not rewrite) |
| Backtest keepers | RS vs SPY **top1**; squeeze trail 10%/18%; **ATR hard-stop k=2.0** (clamp 1.5%–6%); friction 0.25%. Tester **−6.25%** rows are the 6% ceiling plus round-trip friction, not a 3% `--stop-pct` (that is ATR-off fallback only). |
| Soft promote | `--require-in-channel` + `--max-channel-span-days 365` — ~22% of RS-top1 entries were already above resist (drag); filtered live ATR stack **n=374 E +2.66% PF 1.96** vs unfiltered **n=496 E +2.29% PF 1.82** |
| Beyond-width (2026-08-28) | `--max-beyond-width 0.25` on IB-windowed ATR keeper: **n=700 E +1.14% PF 1.40** vs off **n=1001 E +0.97% PF 1.32**. `0.0` too few; `0.5`/`1.0` worse. SRCE 2026 pierce was 0.19 (kept). Entry-feature CSV + univariate mining — do not promote RSI/calendar/ADV from raw quintiles. See [entry features](edge_hunt/channel_touch/2026-08-28_channel_touch_entry_features.md) |
| L3 rail-touch entry (2026-08-28) | Bounce-from-above `l3_touch` then opt loops: abort if tagged <6 bars after H2, `--max-rsi 50`, cancel close-above-resist. **n=1024 E +1.77% PF 1.62** (all year buckets +). Beats pivot n=700 E +1.14 PF 1.40 on this window. **Retired as nightly 2026-08-29** (rollback `--no-h2-resist-break ...`). HTML: `1d_unrealistic/1d_l3_touch.html`. See [opt loops](edge_hunt/channel_touch/2026-08-28_channel_touch_l3_opt_loops.md) and [nightly l3](edge_hunt/channel_touch/2026-08-29_channel_touch_nightly_l3.md) |
| History gap | Research window starts 2018-11 but **kept trades from 2023+** (Alpaca IEX per-symbol starts often ~2020–22; RS top1). **IB 1d prefix backfill 2026-08-26:** ~1900+ symbols filled via `backfill_ib_daily_prefix.py` (one-shot + 0.25s pacing); remainder mostly recent IPOs with **no prefix gap** (IB history starts at/after Alpaca) — tracked in `logs/data/ib_prefix_no_gap_symbols.txt`. |
| IB-fallback windowed (2026-08-28) | Default 504/252 window + IB prefix: **n=1001 E +0.97% PF 1.32**; **106 pre-2020 buys (2019)** vs 0 non-windowed. Edge weaker — more history, lower quality density. CSV: `channel_touch_trades_20260828_000726.csv` |
| Fixed 2% stop (2026-08-28) | Same windowed IB-fallback stack, `--stop-pct 0.02` (no ATR): **n=1014 E +0.84% PF 1.45** WR 16.7% hold 16.3d; 80% hard-stop exits. PF up vs ATR k=2, E down — **do not replace keepers**. CSV: `channel_touch_trades_20260828_023127.csv`. See [stop 2%](edge_hunt/channel_touch/2026-08-28_channel_touch_stop2pct.md) |
| Robustness (014435) | Drop BETR: E **+1.70** PF **1.61**; top-3 tail **24.8%**; bootstrap P(mean&lt;0) **0.3%**; **median trade −3.05%** (fat-tail); drop top-5% winners kills PF; concurrent opens max **29** / med **11** — see [robustness](edge_hunt/channel_touch/2026-08-25_channel_touch_robustness.md) |
| Confidence size (2026-08-29) | Ridge on `gain_pct_net`, expanding WF + purge/embargo 21d, size 0.25x–2.0x from train pred percentiles. **1d RS-top1 OOS:** scale-all E **+2.29 vs +2.46** equal (spearman **-0.05**). **15m unranked** scale-all lifts E/PF but **RS-top1 reverses** (E +0.44 vs +0.46; MDD 41 vs 26). Skip-neg not useful. **No promote; nightly stays equal-dollar.** See [confidence size](edge_hunt/channel_touch/2026-08-29_channel_touch_confidence_size.md) |
| 1d loser-filter (2026-09-05) | Ridge/logistic keep-skip on current_best unique next-mid **n=1239** (**pre-`_as_session_date`**). Honest rails/wait features: expanding 2024–26 ridge-thr looks strong (kept n=143 E +3.72 PF 2.71) but **2023 holdout E does not lift** (+2.91 vs +2.96) and 2024 drop-top-3 PF 0.97. Fill-day RSI/close is a next-mid leak and does not help. Spearman |rho|<0.07. **No promote.** See [loser filter](edge_hunt/channel_touch/2026-09-05_channel_touch_1d_loser_filter.md) |
| 1d crowding (2026-09-05 / rebuilt 2026-09-06) | Unique-symbol equal-dollar concurrent **max 85 / med 18** — not a live book. Pre-fix cap-2 wait **n=933 E +3.01 PF 2.25** vs 1239 / +2.27 / 1.92. Honest next-mid unique **n=1986 E +2.24 PF 1.89**; cap-2 wait **n=1395 E +2.29 PF 1.91** — the +3.01 lift was a session-join artifact. Do not wire cap-2. See [crowding](edge_hunt/channel_touch/2026-09-05_channel_touch_1d_crowding.md) and [next-mid rebuild](edge_hunt/channel_touch/2026-09-06_channel_touch_next_mid_session_date.md) |
| Shakeout rebuy (2026-08-29) | After L3, rebuy if close-through-support then from-above reclaim within N bars (not hold-through). **10 bars:** shakeout-only n=73 E **+0.89** PF **1.29** (2020-21 E **-3.19** PF 0.17); combined+RS n=1064 E **+1.66** PF **1.57** vs keeper **+1.77 / 1.62**. CSTM 2019-06-06 would have been **+27%**. **No promote.** See [shakeout rebuy](edge_hunt/channel_touch/2026-08-29_channel_touch_shakeout_rebuy.md) |
| Shakeout then second H2 break (2026-09-05) | After a taken H2 resist-break, if price closes back inside and support holds, fill the next close above resistance (rebuy, not hold-through). **Not** L3 support-reclaim. Close-fill **any-closed min_inside=1:** sleeve n=1439 E **+3.16** PF **2.32**; combined unique n=4558 E **+2.62** PF **2.10**. **Nightly on** (unrealistic close / rail-clip + re-arm; `--no-shakeout-breakout` to disable). Realistic next-mid unique **n=1986 E +2.24 PF 1.89** (2026-09-06 `_as_session_date` rebuild; pre-fix was n=1239 E +2.27 PF 1.92) vs frozen n=713 E +1.78 PF 1.72 — HTML in `1d_unrealistic/1d_channel_touch.html` (still the pre-fix overlay until regen; not `current_best`). SXI extra **+46%** net on next-mid (2020-11-05). Hard-stop-only and min_inside=5 are weaker. See [shakeout breakout](edge_hunt/channel_touch/2026-09-05_channel_touch_shakeout_breakout.md) |
| Shakeout-confirm-only skip first (2026-09-12) | Arm on first close above resist; do not buy it; fill after inside then 2nd close. Live last-15m N+1 **n=1191 E +0.36 PF 1.12** vs current_best n=2297 E +0.27 PF 1.09. **2022 n=67 E −3.73 PF 0.11** (worse than first-chase). Clip unique n=1942 E +2.61 is lookback. Candle/vol occupancy-walk skipped. **No promote.** See [confirm-only](edge_hunt/channel_touch/2026-09-12_shakeout_confirm_only.md) |
| 1d hot-cross buy-now (2026-09-06) | `--intraday-trigger hot-cross` `--hot-cross-fill lerp85`: first 15m high >= daily rail after wait, no EOD close. Unique span365 + shakeout **n=4079 E −0.19 PF 0.94**. Occupancy sells are the daily clip. Honest 15m N+1 **n=4079 E −0.83 PF 0.77**. **No promote.** Demoted to `1d_unrealistic/1d_hot_cross.html` 2026-09-12. See [hot-cross](edge_hunt/channel_touch/2026-09-06_channel_touch_1d_15m_trigger.md) and [slot swap](edge_hunt/channel_touch/2026-09-12_channel_touch_1d_hot_cross_out_of_current_best.md) |
| 1d open-cross fill (2026-09-05) | First IB 15m open above resist, fill at that close. Unique span365 + shakeout **n=2845 E +2.06 PF 1.775** vs next-mid HTML n=1239 E +2.27 PF 1.92 (pre-fix). **No promote.** Nightly stays close fills. RDWR **25.71 @ 09:45 ET**. See [open-cross](edge_hunt/channel_touch/2026-09-05_channel_touch_open_cross.md) |
| 1d next-open A/B (2026-09-05) | `--touch-error-pct` 1.2 vs 0 and same-bar close vs `--realistic-fill-mode next-open` (next session MOO, no 15m). Unique span365 + shakeout: B12 close **n=4558 E +2.62 PF 2.10** (keeper); tick-above **n=5649 E +1.38 PF 1.49**; next-open 1.2 **n=4894 E +0.45 PF 1.14** (2020-21 E −0.07). **No promote.** See [next-open A/B](edge_hunt/channel_touch/2026-09-05_channel_touch_next_open_ab.md) |
| 1d next-mid session date (2026-09-05 / 2026-09-06) | Daily midnight UTC must stay the calendar date for the 15m join. Pre-fix HTML n=1239 used prior-session 15m (RDWR 24.64). **Rebuilt:** unique **n=1986 E +2.24 PF 1.89**. Not `current_best`. See [session date](edge_hunt/channel_touch/2026-09-05_channel_touch_session_date.md) and [rebuild](edge_hunt/channel_touch/2026-09-06_channel_touch_next_mid_session_date.md) |
| H2 resist-break (2026-08-29) | Fill close-above-resistance after H2. Sleeve span365 **n=2253 E +2.80 PF 2.21**. **Nightly trigger as of 2026-08-29** (`--h2-resist-break-only`, no RSI/in-channel/beyond). HTML: `1d_unrealistic/1d_channel_touch.html` (unrealistic rail clip; not `current_best`). 15m 300-name analog span<=10 + RS top1 **n=1664 E +0.46 PF 1.87** vs 15m L3 +0.31 / 1.57. **Full IB 15m reverses:** H2 span10+RS n=1766 E +0.30 PF 1.48 vs L3 RS n=1775 E +0.34 PF 1.61 (research only; do not promote 15m H2). See [nightly H2](edge_hunt/channel_touch/2026-08-29_channel_touch_nightly_h2_break.md) and [15m full](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_full_h2_break.md) |
| Unique-symbol/day (2026-08-29) | RS top1 was a 1-name/day **capacity cap**, not a quality filter. Live nightly takes every distinct symbol (earliest fill per ticker). Daily H2 span365 **n=2253 E +2.80 PF 2.21** (med 2 names/day, max 30) vs RS top1 n=981 E +3.11 PF 2.34. Full IB 15m H2 span10 unique **n=65561 E +0.30 PF 1.55** vs RS top1 n=1766 E +0.30 PF 1.48. Optional `--max-entries-per-day 1` restores RS top1. See [unique symbol](edge_hunt/channel_touch/2026-08-29_channel_touch_unique_symbol_day.md) |
| 15m unique filters (2026-08-29) | Five pre-registered gates on unique-symbol 15m H2 (prior-bar only, expanding-year OOS). Only **H5 `volume_rel_20 >= 1`** lifts: **n=39442 E +0.34 PF 1.63**. Tight-break, RS floor, ATR ceiling, and quiet-day hurt or flat. Post-hoc wide overshoot (train p80) n=12779 E +0.61 PF 2.43. Do not use full-day name counts. See [unique filters](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_unique_filters.md) |
| 15m H5 refine (2026-08-30) | Train-year overshoot + stricter volume on H5. **H5 + overshoot p80 + vol>=2:** **n=6306 WR 48.8% E +0.93 PF 3.61** (drop-top-1% PF 3.22 WR 48.3; all year buckets +; ~2.5 fills/RTH day). H5 + overshoot p50 keeps ~20k (WR 39 PF 2.18). Volume-alone / narrower width / logistic-on-H5 did not beat the overshoot cap. **Research only — not nightly.** See [H5 refine](edge_hunt/channel_touch/2026-08-30_channel_touch_15m_h5_refine.md) |
| Rejected | Hard ADV/ATR floors; SPY SMA50 alone; structure-exit bundle; H3 lower-40% geometry; `entry_mode=reclaim` until look-ahead fixed; ridge P&L confidence sizing on RS-top1; **1d unique next-mid loser-filter ridge/logistic**; L3 shakeout rebuy (5/10/15 bars); in-channel / RSI 50 on H2 breakouts; 15m unique H1–H4 (RS floor, tight-break, ATR ceiling, quiet-day); full-day 15m name counts; wiring 15m H5 into nightly; **1d `open-cross` and `next-open` as a replacement for nightly close fills**; **daily `--touch-error-pct 0` (tick-above)**; **1d `--intraday-trigger hot-cross` buy-now (n=4079 E −0.19 PF 0.94)**; **`--shakeout-confirm-only` skip-first last-15m (n=1191 E +0.36 PF 1.12; 2022 E −3.73 PF 0.11)**; **causal VIX backwardation / expanding HMM skip on last-15m (2022 leftover worse or zeros 2022 by killing 2020-21)**; **Pine VX2/VX1 slope<4 on last-15m (TV dumps, ratio in Python; VX12_lt4 n=1645 E −0.00 PF 1.00 geo 1.00; 2022 leftover −2.45; 2020-21 blunt; do not fish 5/8)**; **universe SMA20 breadth / last-15m RTH exhaustion skip (B50 blunt; X_d1 2022 leftover −2.71)**; **last-15m AND Deepvue weekend take-4 (n=1 MRNA; ever-focused 2020-21 E −1.31 PF 0.66)**; **last-15m H2 vs L3 causal switch via rolling WR/PF, TLT SMA, or UUP SMA (WR leftover 2022 worse; TLT blunt 2020-21; UUP leaves 2023 red)**; **TLT as a last-15m global gate / sit-2022 (SMA100 skips 113/117 of 2022 but 1502/2297 fills; leftover 2023 E −1.69; 2020-21 blunt; geo 1.02)**; **last-15m H2 + 15m L3 occupancy blend (unified FCFS 8/15/20 and dedicated 15+5; 2022 stays red; cap 8 starves H2; L3 not locked out at 15)**; **last-15m consecutive-loss sit-out L5/S3/P2 (geo 1.20 but leftover 2022 E −2.62; 2022-23 not longer-tailed vs 2020-21; do not fish L/S/P)** |
| Long-history window | 2018-11-01 → 2026-08-23; density still mostly **2024–2026** |
| Interactive report | **`current_best/`** live-executable: `1d_last_15m.html` / `1d_channel_touch.html` (last-15m-open-mid + 15m N+1 mid **n=2297 E +0.27 PF 1.09**), `15m_channel_touch.html` (L3 wait-12 signal-close **n=1744 E +0.13 PF 1.22**), plus realistic 15m H2/H5 (losing/flat). **Clip / clip-exit 1d:** `reports/ascending_channels/1d_unrealistic/` (`1d_h2_resist_break.html` nightly H2 + shakeout, `1d_hot_cross.html` lerp85 occupancy-sell n=4079 E −0.19, `1d_l3_touch.html` retired L3). Fill rules: [realistic purchasing](../features/realistic_purchasing.md). |
| Walk-replay (2026-09-02) | Feed detector bars `0..t` only. AAPL 1d live H2 book: **batch=walk n=3** after `causal_h2` + L3 tags excluded from occupancy. CLI: `scripts/research/replay_channel_touch_walk.py --symbol AAPL`. See [walk-replay](edge_hunt/channel_touch/2026-09-02_channel_touch_walk_replay.md) |
| Causal full-universe (2026-09-03) | Replaced `current_best/`. Live 1d H2 unique **n=3364 E +2.40 PF 2.02** (leaky 2253 / +2.80 / 2.21). 15m H5 p80+vol2 **n=8212 E +0.92 PF 3.57**. Full 15m H2 RS **n=1770 E +0.21 PF 1.33** — do not promote. See [causal rescan](edge_hunt/channel_touch/2026-09-03_channel_touch_causal_full_universe.md) |
| Live monitor | **Daily nightly:** Alpaca 1d refresh → H2 resist-break → Telegram (`channel_touch_nightly.py`). **15m research loop:** IB 15m backfill → armed H2 watchlist in TimescaleDB → minute Alpaca last → `/hot` dashboard → bar-close H5 (`channel_touch_15m.py`). Do not stream 1478 via IB. Optional `--max-entries-per-day 1` restores RS top1. Rollback L3: `--no-h2-resist-break --require-in-channel --max-rsi 50 --max-beyond-width 0.25`. |
| TV draw caveat | Left-endpoint time-snap can float long rails — verify vs `watchlist_channels_draw.json` ([playbook](../features/tv_channel_trendline_alert.md)) |
| 15m hunt | **Research, not live.** IB 15m + native SPY RS, 2018-11-01→2025-12-02. Pivot: n=1764 E **+0.15%** PF **1.24**. **2026-08-29 l3_touch loops (300 names):** L3 + min_wait 12, **no** daily beyond-0.25: **n=1751 E +0.32% PF 1.58** (all year buckets +). Daily `--max-beyond-width 0.25` hurts 15m L3. L4 (`--entry-touch 4`) has **fewer** signals (n=1312 E +0.20 PF 1.36), not more. Entry MLP/logistic and `min_close_loc` wick filter not promoted (same-bar close leak). **Same-bar leak fix:** features at `fill_i-1`; 15m lagged rescan **n=1751 E +0.31 PF 1.57**. Honest lagged `min_close_loc` 0.6 dies (E +0.06). Daily `--intraday-fill 15m` hybrid on the same 300: **n=260 E +1.31 PF 1.47** vs leaky daily wick **n=314 E +1.73 PF 1.63** — does not beat wait-6/RSI-50; **nightly stays daily H2 resist-break** (not 15m hybrid). **15m H2 resist-break (300 names)** wait-12 span<=10 + RS top1 **n=1664 E +0.46 PF 1.87** vs L3 wait-12 +0.31 / 1.57. **Full IB 15m (1177 loaded, 3064s):** H2 span10+RS **n=1766 E +0.30 PF 1.48** loses to L3 RS **n=1775 E +0.34 PF 1.61** — do not promote 15m H2 over wait-12 L3. Unique-symbol / H5 / tight overshoot+vol rows above. See [15m opt](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_opt_loops.md), [same-bar leak](edge_hunt/channel_touch/2026-08-29_channel_touch_samebar_leak.md), [nightly H2](edge_hunt/channel_touch/2026-08-29_channel_touch_nightly_h2_break.md), [15m full](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_full_h2_break.md), [unique filters](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_unique_filters.md), and [H5 refine](edge_hunt/channel_touch/2026-08-30_channel_touch_15m_h5_refine.md) |

### What is frozen vs in motion

- **Frozen / prefer ship:** Phase 6b near-KEEP blend; channel-touch **H2 resist-break** nightly (min-wait 6 / span365 / unique-symbol/day / `--shakeout-breakout`; no RSI/in-channel/beyond; **unrealistic** same-bar rail clip, not the realistic purchaser). **Deep Pullback** and **QS Five Swing** live on `/hot` (DP / QS) as selective SPY MOC sleeves — not edge-hunt Sharpe>1 promotes.
- **Not fishing further (same windows):** Edge-hunt Sharpe>1 on Phase 5/6/6c overlays; reclaim until look-ahead fixed; SPY SMA hard filter; **causal VIX backwardation / expanding HMM skip on last-15m** (same bear-rally leftover as SMA; HMM3 zeros 2022 by wrecking 2020-21); **Pine VX2/VX1 slope<4 (TV `CBOE_DLY:VX1!`/`VX2!` dumps, ratio in Python; VX12_lt4 n=1645 E −0.00; leftover 2022 worse, 2020-21 blunt; do not fish 5/8)**; **universe SMA20 breadth or last-15m RTH run/ATR skip** (B50 blunt; X_d1 2022 leftover worse); **AND last-15m H2 with Deepvue weekend take-4** (n=1 MRNA; ever-focused wrecks 2020-21; do not occupancy-walk Deepvue RS/RMV/QQQ onto last-15m); **last-15m H2 vs L3 causal switch via rolling WR/PF, TLT SMA, or UUP SMA** (WR leftover 2022 worse; TLT blunt 2020-21; UUP leaves 2023 red); **last-15m H2 vs short H3+ regime switch via same sensors** (WR 2022 still red; PF20 greens only as ~82% short; macros leave 2023 red); **TLT last-15m global gate / calendar sit-2022** (oracle cash n=2180 E +0.41 still leaves 2023 −0.33; live SMA100/200/ROC63 not a 2022-only switch); **last-15m H2 + 15m L3 occupancy blend** (unified FCFS vs dedicated sleeves; 2022 red; do not cap-fish); **last-15m consecutive-loss sit-out** (L5/S3/P2 geo 1.20 but leftover 2022 E −2.62; 2022-23 not longer-tailed vs 2020-21; do not fish L/S/P); lower-40% geometry; l3_touch max-age/max-wait caps; bb%B 0.2; 15m `min_close_loc` wick filter (leaky or lagged); 15m entry MLP as a hard gate; ridge P&L confidence sizing (RS-top1 OOS); **1d unique next-mid loser-filter ridge/logistic as a hard gate** (2023 holdout E does not lift; expanding-WF spike is 2024–26 only); copying daily beyond-0.25 onto 15m L3; daily `--intraday-fill 15m` hybrid until it beats wait-6/RSI-50 without fill-bar close; L3 shakeout rebuy (5/10/15 bars); L3 in-channel / RSI 50 on H2 breakouts; **15m H2 resist-break as a replacement for wait-12 L3 on the full IB 15m set**; **15m L3 min-wait 1 (textbook first-tag) vs wait-12 on realistic fill**; 15m unique H1–H4 (tight-break / RS floor / ATR ceiling / quiet-day); logistic-on-H5 as a hard gate; wiring any 15m book into nightly; H2 shakeout-breakout **hard-stop-only** and **min_inside=5** vs any-closed min_inside=1; **1d `open-cross` / `next-open` / daily tick-above (`--touch-error-pct 0`) vs the B12 close-fill keeper**; **1d `--intraday-trigger hot-cross` buy-now vs nightly clip**; **choppy-regime RSI-2 MOC / 3-day pullback** (occ-15 fric0.25 PF 0.96; pullback 2022-23 PF 0.77; not a last-15m overlay); **QS 15m@15:45 as an early MOC decision** (bar spans into the close; use 1m 15:57, not 5m@15:55).
- **Next (optional):** Beat `current_best/` 1d last-15m **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01) with a live-executable causal fill **and** realistic sells (do not grade vs clip or clip-exit hot-cross n=4079 E −0.19). Honest next-mid unique is **n=1986 E +2.24 PF 1.89** (still a post-EOD overlay). Optional live cap of 2 unique names/day ranked by `wait_bars` lost its +3.01 / 2.25 lift on the rebuild (honest n=1395 E +2.29 PF 1.91) — do not wire. Greedy max concurrent ~8 is the equal-dollar capital book (unconstrained med 18 / max 85). Run `backfill_ib_15m_universe.py` with Gateway up; then dry-run `channel_touch_15m.py`. Bootstrap / max-open on the 15m H5 book; ADV-tiered friction.
- **Ops cadence:** Daily EOD scan on stored bars. 15m: armed watchlist + Alpaca last price + `/hot` dashboard; do not stream the IB 15m universe.

---

## Layout

```
docs/status_log/
  current_status.md          <- this file
  daily/                     <- day-by-day digests
  edge_hunt/                 <- SPY-beat portfolio phases
    channel_touch/           <- ascending-channel research + nightly
  weekly_bigvol/             <- Weekly BigVol + TTM Squeeze
```
