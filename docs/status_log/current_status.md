# Research & data — current status

Last updated: **2026-08-29**

Working notes live under `docs/status_log/` (`edge_hunt/`, `edge_hunt/channel_touch/`, `weekly_bigvol/`, `daily/`).

---

## Stock market data (TimescaleDB `market_data`)

Stocks only; options excluded. Queried via `utils.db.timescaledb_client.get_timescaledb_client()`.

| Timeframe | Symbols | Bars (approx) | Date range / freshness |
|-----------|---------|---------------|------------------------|
| Daily (`1d`) | **~2,217** | ~7.6M+ | ALPACA primary; gap-fill 2026-08-23 + nightly multi-symbol refresh. Nightly 2026-08-25 ~23:00: **2113 saved / 104 failed** (95.3%), `as_of` scan bar **2026-08-24**. |
| 15-minute (`15m`) | **1,478** | ~58.7M (EXPLAIN est.) | IB primary, ~2018-01-02 → 2025-12-02; ALPACA 15m effectively unused aside from leftover `AEO` |
| Weekly | **0** | — | Not stored — resample from daily |

### Data sources

| Source | Role | Persist? |
|--------|------|----------|
| **Alpaca** | Historical OHLCV (primary daily) via `utils/data/fetch_data.py` / `update_universe_data.py`; IEX default; nightly `--multi-symbol` batches | Yes — provider `ALPACA` |
| **IBKR** | Historical OHLCV (primary 15m); Gateway ~4001 | Yes — provider `IB` |
| **TradingView** | Chart / Pine verify only (workspace MCP, CDP 9222) | No |

### Ops notes

- Full-table heavy `COUNT`/`GROUP BY` on `market_data` is slow; prefer per-timeframe/provider queries and `EXPLAIN` estimates.
- `ticker_universe` ~6,072 tickers; daily and 15m sets are subsets.
- For 15m universe backtests prefer the IB 15m symbol set over the full daily universe.
- ALPACA daily panel coverage for many names starts ~2020–2022; compute SPY SMA on full IB SPY history then align to the panel index.
- Gap-fill detail: [daily/2026-08-23](daily/2026-08-23.md) (and archived plan in that folder).

---

## Research status (2026-08-26)

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

| Item | State |
|------|--------|
| Detector | Classical Edwards/Magee; Pine `indicators/pine/ascending_channel_3touch.pine` (**v1 unchanged** — quality via post-filters, not rewrite) |
| Backtest keepers | RS vs SPY **top1**; squeeze trail 10%/18%; **ATR hard-stop k=2.0** (clamp 1.5%–6%); friction 0.25% |
| Soft promote | `--require-in-channel` + `--max-channel-span-days 365` — ~22% of RS-top1 entries were already above resist (drag); filtered live ATR stack **n=374 E +2.66% PF 1.96** vs unfiltered **n=496 E +2.29% PF 1.82** |
| Beyond-width (2026-08-28) | `--max-beyond-width 0.25` on IB-windowed ATR keeper: **n=700 E +1.14% PF 1.40** vs off **n=1001 E +0.97% PF 1.32**. `0.0` too few; `0.5`/`1.0` worse. SRCE 2026 pierce was 0.19 (kept). Entry-feature CSV + univariate mining — do not promote RSI/calendar/ADV from raw quintiles. See [entry features](edge_hunt/channel_touch/2026-08-28_channel_touch_entry_features.md) |
| L3 rail-touch entry (2026-08-28) | Bounce-from-above `l3_touch` then opt loops: abort if tagged <6 bars after H2, `--max-rsi 50`, cancel close-above-resist. **n=1024 E +1.77% PF 1.62** (all year buckets +). Beats pivot n=700 E +1.14 PF 1.40 on this window. **Nightly uses this stack as of 2026-08-29.** Report: `channel_touch_tv_report_interactive_rs_top1_default_fric0.25_l3_touch_minwait6_rsi50_beyond025_inchannel_span365_20260829_105855.html`. See [opt loops](edge_hunt/channel_touch/2026-08-28_channel_touch_l3_opt_loops.md) and [nightly l3](edge_hunt/channel_touch/2026-08-29_channel_touch_nightly_l3.md) |
| History gap | Research window starts 2018-11 but **kept trades from 2023+** (Alpaca IEX per-symbol starts often ~2020–22; RS top1). **IB 1d prefix backfill 2026-08-26:** ~1900+ symbols filled via `backfill_ib_daily_prefix.py` (one-shot + 0.25s pacing); remainder mostly recent IPOs with **no prefix gap** (IB history starts at/after Alpaca) — tracked in `logs/data/ib_prefix_no_gap_symbols.txt`. |
| IB-fallback windowed (2026-08-28) | Default 504/252 window + IB prefix: **n=1001 E +0.97% PF 1.32**; **106 pre-2020 buys (2019)** vs 0 non-windowed. Edge weaker — more history, lower quality density. CSV: `channel_touch_trades_20260828_000726.csv` |
| Fixed 2% stop (2026-08-28) | Same windowed IB-fallback stack, `--stop-pct 0.02` (no ATR): **n=1014 E +0.84% PF 1.45** WR 16.7% hold 16.3d; 80% hard-stop exits. PF up vs ATR k=2, E down — **do not replace keepers**. CSV: `channel_touch_trades_20260828_023127.csv`. See [stop 2%](edge_hunt/channel_touch/2026-08-28_channel_touch_stop2pct.md) |
| Robustness (014435) | Drop BETR: E **+1.70** PF **1.61**; top-3 tail **24.8%**; bootstrap P(mean&lt;0) **0.3%**; **median trade −3.05%** (fat-tail); drop top-5% winners kills PF; concurrent opens max **29** / med **11** — see [robustness](edge_hunt/channel_touch/2026-08-25_channel_touch_robustness.md) |
| Confidence size (2026-08-29) | Ridge on `gain_pct_net`, expanding WF + purge/embargo 21d, size 0.25x–2.0x from train pred percentiles. **1d RS-top1 OOS:** scale-all E **+2.29 vs +2.46** equal (spearman **-0.05**). **15m unranked** scale-all lifts E/PF but **RS-top1 reverses** (E +0.44 vs +0.46; MDD 41 vs 26). Skip-neg not useful. **No promote; nightly stays equal-dollar.** See [confidence size](edge_hunt/channel_touch/2026-08-29_channel_touch_confidence_size.md) |
| Shakeout rebuy (2026-08-29) | After L3, rebuy if close-through-support then from-above reclaim within N bars (not hold-through). **10 bars:** shakeout-only n=73 E **+0.89** PF **1.29** (2020-21 E **-3.19** PF 0.17); combined+RS n=1064 E **+1.66** PF **1.57** vs keeper **+1.77 / 1.62**. CSTM 2019-06-06 would have been **+27%**. **No promote.** See [shakeout rebuy](edge_hunt/channel_touch/2026-08-29_channel_touch_shakeout_rebuy.md) |
| Rejected | Hard ADV/ATR floors; SPY SMA50 alone; structure-exit bundle; H3 lower-40% geometry; `entry_mode=reclaim` until look-ahead fixed; ridge P&L confidence sizing on RS-top1; L3 shakeout rebuy (5/10/15 bars)
| Long-history window | 2018-11-01 → 2026-08-23; density still mostly **2024–2026** |
| Interactive report | Stable links: `reports/ascending_channels/current_best/` (`1d_channel_touch.html`, `15m_channel_touch.html`). Daily l3_touch (nightly source): `..._l3_touch_minwait6_rsi50_beyond025_inchannel_span365_20260829_195015.html` (HTML max/day=1 matches Python RS; missing RS last/stable — old 105855 page showed PF 1.48 from a broken JS sort). 15m research: `..._15m_l3_touch_minwait12_priorbar_20260829_112614.html`. Pivot+beyond0.25: `..._beyond025_inchannel_span365_ib_fallback_20260828_165613.html` |
| Live monitor | Nightly: Alpaca multi-symbol 1d refresh → **l3_touch** scan (min-wait 6, RSI<=50, in-channel, span365, beyond 0.25, RS top1, ATR k=2, window 504/252) → Telegram (`utils/notify/telegram_pinger.py`). Pivot-confirm retired as the default (`--entry-mode pivot` to revert). **2026-08-25 23:00** (old pivot) run: **KRYS** @ 341.57 (as_of 2026-08-24; stop −6%; RS126 +21.3%; channel pos 0.35) |
| TV draw caveat | Left-endpoint time-snap can float long rails — verify vs `watchlist_channels_draw.json` ([playbook](../features/tv_channel_trendline_alert.md)) |
| 15m hunt | **300-name research, not live.** IB 15m + native SPY RS, 2018-11-01→2025-12-02. Pivot: n=1764 E **+0.15%** PF **1.24**. **2026-08-29 l3_touch loops:** L3 + min_wait 12, **no** daily beyond-0.25: **n=1751 E +0.32% PF 1.58** (all year buckets +). Daily `--max-beyond-width 0.25` hurts 15m L3. L4 (`--entry-touch 4`) has **fewer** signals (n=1312 E +0.20 PF 1.36), not more. Entry MLP/logistic and `min_close_loc` wick filter not promoted (same-bar close leak). **Same-bar leak fix:** features at `fill_i-1`; 15m lagged rescan **n=1751 E +0.31 PF 1.57**. Honest lagged `min_close_loc` 0.6 dies (E +0.06). Daily `--intraday-fill 15m` hybrid on the same 300: **n=260 E +1.31 PF 1.47** vs leaky daily wick **n=314 E +1.73 PF 1.63** — does not beat wait-6/RSI-50; **nightly stays daily l3_touch wick fills** (not 15m hybrid). See [15m opt](edge_hunt/channel_touch/2026-08-29_channel_touch_15m_opt_loops.md) and [same-bar leak](edge_hunt/channel_touch/2026-08-29_channel_touch_samebar_leak.md) |

### What is frozen vs in motion

- **Frozen / prefer ship:** Phase 6b near-KEEP blend; channel-touch l3_touch nightly (min-wait 6 / RSI 50 / in-channel / span365 / beyond 0.25).
- **Not fishing further (same windows):** Edge-hunt Sharpe>1 on Phase 5/6/6c overlays; reclaim until look-ahead fixed; SPY SMA hard filter; lower-40% geometry; l3_touch max-age/max-wait caps; bb%B 0.2; 15m `min_close_loc` wick filter (leaky or lagged); 15m entry MLP as a hard gate; ridge P&L confidence sizing (RS-top1 OOS); copying daily beyond-0.25 onto 15m L3; daily `--intraday-fill 15m` hybrid until it beats wait-6/RSI-50 without fill-bar close; L3 shakeout rebuy (5/10/15 bars).
- **Next (optional):** Channel-touch 15m robustness (drop-top-N/bootstrap) on the 2026-08-29 wait12 keeper before expanding past 300 names; portfolio max-open in live sizing; ADV-tiered friction.
- **Ops cadence:** Post-close EOD scan on stored bars — do not stream full universe via IB.

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
