# Research & data — current status

Last updated: **2026-09-11**

Working notes live under `docs/status_log/` (`edge_hunt/`, `edge_hunt/channel_touch/`, `weekly_bigvol/`, `daily/`).

---

## Stock market data (TimescaleDB `market_data`)

Stocks only; options excluded. Queried via `utils.db.timescaledb_client.get_timescaledb_client()`.

| Timeframe | Symbols | Bars (approx) | Date range / freshness |
|-----------|---------|---------------|------------------------|
| Daily (`1d`) | **~2,217** | ~7.6M+ | ALPACA primary; gap-fill 2026-08-23 + nightly multi-symbol refresh. Nightly 2026-08-25 ~23:00: **2113 saved / 104 failed** (95.3%), `as_of` scan bar **2026-08-24**. |
| 15-minute (`15m`) | **1,478** | ~58.7M (EXPLAIN est.) | IB primary, ~2018-01-02 → **2025-12-02** (stale vs 2026-08-30). Catch-up now runs **16:30 ET** via `after_rth_ib_backfill` (`backfill_ib_15m_universe.py`, client 8822) then 5m. ALPACA 15m unused aside from leftover `AEO`. |
| 5-minute (`5m`) | starting (A, AAL full; rest pending) | — | IB, same symbol set as 15m. Ingest `backfill_ib_5m_universe.py` (client 8823) **year-first**: 2025-through-now all symbols, then 2024..2020. Yields to RTH (stop 09:15 ET; resume 16:30 ET after 15m catch-up; weekends continuous). Hourly watchdog restarts 5m if the after-RTH job died. Playbook: [ib_5m_backfill](../features/ib_5m_backfill.md). |
| Weekly | **0** | — | Not stored — resample from daily |

### Data sources

| Source | Role | Persist? |
|--------|------|----------|
| **Alpaca** | Historical OHLCV (primary daily) via `utils/data/fetch_data.py` / `update_universe_data.py`; IEX default; nightly `--multi-symbol` batches | Yes — provider `ALPACA` |
| **IBKR** | Historical OHLCV (primary 15m; 5m ingest starting 2026-09-04); Gateway ~4001 | Yes — provider `IB` |
| **TradingView** | Chart / Pine verify only (workspace MCP, CDP 9222) | No |

### Ops notes

- Full-table heavy `COUNT`/`GROUP BY` on `market_data` is slow; prefer per-timeframe/provider queries and `EXPLAIN` estimates.
- `ticker_universe` ~6,072 tickers; daily and 15m sets are subsets.
- For 15m universe backtests prefer the IB 15m symbol set over the full daily universe.
- ALPACA daily panel coverage for many names starts ~2020–2022; compute SPY SMA on full IB SPY history then align to the panel index.
- Gap-fill detail: [daily/2026-08-23](daily/2026-08-23.md) (and archived plan in that folder).

---

## Research status (2026-09-06)

### Edge hunt (SPY-beating portfolio sleeves)

Status: **`docs/status_log/edge_hunt/`**

| Phase | Verdict |
|-------|---------|
| 4 | **KEEP** `Blend_SPY70_BV30` |
| 5 | No promote (XS mom + swing MR) |
| 6 | No full-sample promote; vol-target near-miss |
| 6b | Near-KEEP `Blend_VT60_BV40` (VT SPY + BigVol) — prefer freeze/productionize over more Sharpe>1 fishing |
| 6c | No promote / no lift vs plain vol-target (VIX curve) |

Promotion gate (vs SPY): Sharpe > 1.0, Sharpe ≥ SPY, MDD better than SPY (CAGR reported, not ranking objective).

### Weekly BigVol + TTM Squeeze

Status: **`docs/status_log/weekly_bigvol/`** — Phase 1 TV verify + Phase 2 edge smoke done. Research execution default: no fixed TP + ~10% stop (+ MA exit). Scanner `squeeze` is TTM zero-cross only.

### Channel-touch / ascending channels (active production track)

Status: **`docs/status_log/edge_hunt/channel_touch/`**

**Latest (2026-09-11):** Rebuilt last-15m HTML after the LAUR detector fix. Unique signal-close span365 **n=2310 E +2.50 PF 2.03**. Last-15m + 15m N+1 mid **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01). LAUR 2021 fill is 2021-09-10 @ 16.82 (Mar-4 rails); Oct 28 gone. HTML: `…165800.html`. **No promote.** See [LAUR false shakeout](edge_hunt/channel_touch/2026-09-11_laur_false_shakeout.md).

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
| 1d hot-cross buy-now (2026-09-06) | `--intraday-trigger hot-cross` `--hot-cross-fill lerp85`: first 15m high >= daily rail after wait, no EOD close. Unique span365 + shakeout **n=4079 E −0.19 PF 0.94**. Gap splits ~PF 0.92–0.95. RDWR Jun-13 absent (in trade from Jun-5 poke). **No promote.** **`current_best/` 1d slot** (`1d_hot_cross.html`) — the number to beat. See [hot-cross](edge_hunt/channel_touch/2026-09-06_channel_touch_1d_15m_trigger.md) and [current_best copy](edge_hunt/channel_touch/2026-09-06_channel_touch_current_best_hot_cross.md) |
| 1d open-cross fill (2026-09-05) | First IB 15m open above resist, fill at that close. Unique span365 + shakeout **n=2845 E +2.06 PF 1.775** vs next-mid HTML n=1239 E +2.27 PF 1.92 (pre-fix). **No promote.** Nightly stays close fills. RDWR **25.71 @ 09:45 ET**. See [open-cross](edge_hunt/channel_touch/2026-09-05_channel_touch_open_cross.md) |
| 1d next-open A/B (2026-09-05) | `--touch-error-pct` 1.2 vs 0 and same-bar close vs `--realistic-fill-mode next-open` (next session MOO, no 15m). Unique span365 + shakeout: B12 close **n=4558 E +2.62 PF 2.10** (keeper); tick-above **n=5649 E +1.38 PF 1.49**; next-open 1.2 **n=4894 E +0.45 PF 1.14** (2020-21 E −0.07). **No promote.** See [next-open A/B](edge_hunt/channel_touch/2026-09-05_channel_touch_next_open_ab.md) |
| 1d next-mid session date (2026-09-05 / 2026-09-06) | Daily midnight UTC must stay the calendar date for the 15m join. Pre-fix HTML n=1239 used prior-session 15m (RDWR 24.64). **Rebuilt:** unique **n=1986 E +2.24 PF 1.89**. Not `current_best`. See [session date](edge_hunt/channel_touch/2026-09-05_channel_touch_session_date.md) and [rebuild](edge_hunt/channel_touch/2026-09-06_channel_touch_next_mid_session_date.md) |
| H2 resist-break (2026-08-29) | Fill close-above-resistance after H2. Sleeve span365 **n=2253 E +2.80 PF 2.21**. **Nightly trigger as of 2026-08-29** (`--h2-resist-break-only`, no RSI/in-channel/beyond). HTML: `1d_unrealistic/1d_channel_touch.html` (unrealistic rail clip; not `current_best`). 15m 300-name analog span<=10 + RS top1 **n=1664 E +0.46 PF 1.87** vs 15m L3 +0.31 / 1.57. **Full IB 15m reverses:** H2 span10+RS n=1766 E +0.30 PF 1.48 vs L3 RS n=1775 E +0.34 PF 1.61 (research only; do not promote 15m H2). See [nightly H2](edge_hunt/channel_touch/2026-08-29_channel_touch_nightly_h2_break.md) and [15m full](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_full_h2_break.md) |
| Unique-symbol/day (2026-08-29) | RS top1 was a 1-name/day **capacity cap**, not a quality filter. Live nightly takes every distinct symbol (earliest fill per ticker). Daily H2 span365 **n=2253 E +2.80 PF 2.21** (med 2 names/day, max 30) vs RS top1 n=981 E +3.11 PF 2.34. Full IB 15m H2 span10 unique **n=65561 E +0.30 PF 1.55** vs RS top1 n=1766 E +0.30 PF 1.48. Optional `--max-entries-per-day 1` restores RS top1. See [unique symbol](edge_hunt/channel_touch/2026-08-29_channel_touch_unique_symbol_day.md) |
| 15m unique filters (2026-08-29) | Five pre-registered gates on unique-symbol 15m H2 (prior-bar only, expanding-year OOS). Only **H5 `volume_rel_20 >= 1`** lifts: **n=39442 E +0.34 PF 1.63**. Tight-break, RS floor, ATR ceiling, and quiet-day hurt or flat. Post-hoc wide overshoot (train p80) n=12779 E +0.61 PF 2.43. Do not use full-day name counts. See [unique filters](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_unique_filters.md) |
| 15m H5 refine (2026-08-30) | Train-year overshoot + stricter volume on H5. **H5 + overshoot p80 + vol>=2:** **n=6306 WR 48.8% E +0.93 PF 3.61** (drop-top-1% PF 3.22 WR 48.3; all year buckets +; ~2.5 fills/RTH day). H5 + overshoot p50 keeps ~20k (WR 39 PF 2.18). Volume-alone / narrower width / logistic-on-H5 did not beat the overshoot cap. **Research only — not nightly.** See [H5 refine](edge_hunt/channel_touch/2026-08-30_channel_touch_15m_h5_refine.md) |
| Rejected | Hard ADV/ATR floors; SPY SMA50 alone; structure-exit bundle; H3 lower-40% geometry; `entry_mode=reclaim` until look-ahead fixed; ridge P&L confidence sizing on RS-top1; **1d unique next-mid loser-filter ridge/logistic**; L3 shakeout rebuy (5/10/15 bars); in-channel / RSI 50 on H2 breakouts; 15m unique H1–H4 (RS floor, tight-break, ATR ceiling, quiet-day); full-day 15m name counts; wiring 15m H5 into nightly; **1d `open-cross` and `next-open` as a replacement for nightly close fills**; **daily `--touch-error-pct 0` (tick-above)**; **1d `--intraday-trigger hot-cross` buy-now (n=4079 E −0.19 PF 0.94)** |
| Long-history window | 2018-11-01 → 2026-08-23; density still mostly **2024–2026** |
| Interactive report | **`current_best/`** live-executable: `1d_hot_cross.html` / `1d_channel_touch.html` (buy-now lerp85 **n=4079 E −0.19 PF 0.94**), `15m_channel_touch.html` (L3 wait-12 signal-close **n=1744 E +0.13 PF 1.22**), plus realistic 15m H2/H5 (losing/flat). **Clip 1d:** `reports/ascending_channels/1d_unrealistic/` (`1d_h2_resist_break.html` nightly H2 + shakeout, `1d_l3_touch.html` retired L3). Fill rules: [realistic purchasing](../features/realistic_purchasing.md). |
| Walk-replay (2026-09-02) | Feed detector bars `0..t` only. AAPL 1d live H2 book: **batch=walk n=3** after `causal_h2` + L3 tags excluded from occupancy. CLI: `scripts/research/replay_channel_touch_walk.py --symbol AAPL`. See [walk-replay](edge_hunt/channel_touch/2026-09-02_channel_touch_walk_replay.md) |
| Causal full-universe (2026-09-03) | Replaced `current_best/`. Live 1d H2 unique **n=3364 E +2.40 PF 2.02** (leaky 2253 / +2.80 / 2.21). 15m H5 p80+vol2 **n=8212 E +0.92 PF 3.57**. Full 15m H2 RS **n=1770 E +0.21 PF 1.33** — do not promote. See [causal rescan](edge_hunt/channel_touch/2026-09-03_channel_touch_causal_full_universe.md) |
| Live monitor | **Daily nightly:** Alpaca 1d refresh → H2 resist-break → Telegram (`channel_touch_nightly.py`). **15m research loop:** IB 15m backfill → armed H2 watchlist in TimescaleDB → minute Alpaca last → `/hot` dashboard → bar-close H5 (`channel_touch_15m.py`). Do not stream 1478 via IB. Optional `--max-entries-per-day 1` restores RS top1. Rollback L3: `--no-h2-resist-break --require-in-channel --max-rsi 50 --max-beyond-width 0.25`. |
| TV draw caveat | Left-endpoint time-snap can float long rails — verify vs `watchlist_channels_draw.json` ([playbook](../features/tv_channel_trendline_alert.md)) |
| 15m hunt | **Research, not live.** IB 15m + native SPY RS, 2018-11-01→2025-12-02. Pivot: n=1764 E **+0.15%** PF **1.24**. **2026-08-29 l3_touch loops (300 names):** L3 + min_wait 12, **no** daily beyond-0.25: **n=1751 E +0.32% PF 1.58** (all year buckets +). Daily `--max-beyond-width 0.25` hurts 15m L3. L4 (`--entry-touch 4`) has **fewer** signals (n=1312 E +0.20 PF 1.36), not more. Entry MLP/logistic and `min_close_loc` wick filter not promoted (same-bar close leak). **Same-bar leak fix:** features at `fill_i-1`; 15m lagged rescan **n=1751 E +0.31 PF 1.57**. Honest lagged `min_close_loc` 0.6 dies (E +0.06). Daily `--intraday-fill 15m` hybrid on the same 300: **n=260 E +1.31 PF 1.47** vs leaky daily wick **n=314 E +1.73 PF 1.63** — does not beat wait-6/RSI-50; **nightly stays daily H2 resist-break** (not 15m hybrid). **15m H2 resist-break (300 names)** wait-12 span<=10 + RS top1 **n=1664 E +0.46 PF 1.87** vs L3 wait-12 +0.31 / 1.57. **Full IB 15m (1177 loaded, 3064s):** H2 span10+RS **n=1766 E +0.30 PF 1.48** loses to L3 RS **n=1775 E +0.34 PF 1.61** — do not promote 15m H2 over wait-12 L3. Unique-symbol / H5 / tight overshoot+vol rows above. See [15m opt](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_opt_loops.md), [same-bar leak](edge_hunt/channel_touch/2026-08-29_channel_touch_samebar_leak.md), [nightly H2](edge_hunt/channel_touch/2026-08-29_channel_touch_nightly_h2_break.md), [15m full](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_full_h2_break.md), [unique filters](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_unique_filters.md), and [H5 refine](edge_hunt/channel_touch/2026-08-30_channel_touch_15m_h5_refine.md) |

### What is frozen vs in motion

- **Frozen / prefer ship:** Phase 6b near-KEEP blend; channel-touch **H2 resist-break** nightly (min-wait 6 / span365 / unique-symbol/day / `--shakeout-breakout`; no RSI/in-channel/beyond; **unrealistic** same-bar rail clip, not the realistic purchaser).
- **Not fishing further (same windows):** Edge-hunt Sharpe>1 on Phase 5/6/6c overlays; reclaim until look-ahead fixed; SPY SMA hard filter; lower-40% geometry; l3_touch max-age/max-wait caps; bb%B 0.2; 15m `min_close_loc` wick filter (leaky or lagged); 15m entry MLP as a hard gate; ridge P&L confidence sizing (RS-top1 OOS); **1d unique next-mid loser-filter ridge/logistic as a hard gate** (2023 holdout E does not lift; expanding-WF spike is 2024–26 only); copying daily beyond-0.25 onto 15m L3; daily `--intraday-fill 15m` hybrid until it beats wait-6/RSI-50 without fill-bar close; L3 shakeout rebuy (5/10/15 bars); L3 in-channel / RSI 50 on H2 breakouts; **15m H2 resist-break as a replacement for wait-12 L3 on the full IB 15m set**; **15m L3 min-wait 1 (textbook first-tag) vs wait-12 on realistic fill**; 15m unique H1–H4 (tight-break / RS floor / ATR ceiling / quiet-day); logistic-on-H5 as a hard gate; wiring any 15m book into nightly; H2 shakeout-breakout **hard-stop-only** and **min_inside=5** vs any-closed min_inside=1; **1d `open-cross` / `next-open` / daily tick-above (`--touch-error-pct 0`) vs the B12 close-fill keeper**; **1d `--intraday-trigger hot-cross` buy-now vs nightly clip**.
- **Next (optional):** Beat `current_best/` 1d hot-cross **n=4079 E −0.19 PF 0.94** with a live-executable causal fill (do not grade vs clip). Honest next-mid unique is **n=1986 E +2.24 PF 1.89** (still a post-EOD overlay). Optional live cap of 2 unique names/day ranked by `wait_bars` lost its +3.01 / 2.25 lift on the rebuild (honest n=1395 E +2.29 PF 1.91) — do not wire. Greedy max concurrent ~8 is the equal-dollar capital book (unconstrained med 18 / max 85). Run `backfill_ib_15m_universe.py` with Gateway up; then dry-run `channel_touch_15m.py`. Bootstrap / max-open on the 15m H5 book; ADV-tiered friction.
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
