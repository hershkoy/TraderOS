# Research & data — current status

Last updated: **2026-08-25**

Working notes live under `docs/status_log/` (`edge_hunt/`, `edge_hunt/channel_touch/`, `weekly_bigvol/`, `daily/`).

---

## Stock market data (TimescaleDB `market_data`)

Stocks only; options excluded. Queried via `utils.db.timescaledb_client.get_timescaledb_client()`.

| Timeframe | Symbols | Bars (approx) | Date range / freshness |
|-----------|---------|---------------|------------------------|
| Daily (`1d`) | **~2,217** | ~7.6M+ | ALPACA primary; after 2026-08-23 gap-fill most names reach **~2026-08-20/21** (~2109/2217 fresh). Nightly 2026-08-25 update: 2113 saved / 104 failed (multi-symbol batches). |
| 15-minute (`15m`) | **1,478** | ~58.7M (EXPLAIN est.) | IB primary, ~2018-01-02 → 2025-12-02; ALPACA 15m effectively unused aside from leftover `AEO` |
| Weekly | **0** | — | Not stored — resample from daily |

### Data sources

| Source | Role | Persist? |
|--------|------|----------|
| **Alpaca** | Historical OHLCV (primary daily) via `utils/data/fetch_data.py` / `update_universe_data.py`; IEX default | Yes — provider `ALPACA` |
| **IBKR** | Historical OHLCV (primary 15m); Gateway ~4001 | Yes — provider `IB` |
| **TradingView** | Chart / Pine verify only (workspace MCP, CDP 9222) | No |

### Ops notes

- Full-table heavy `COUNT`/`GROUP BY` on `market_data` is slow; prefer per-timeframe/provider queries and `EXPLAIN` estimates.
- `ticker_universe` ~6,072 tickers; daily and 15m sets are subsets.
- For 15m universe backtests prefer the IB 15m symbol set over the full daily universe.
- ALPACA daily panel coverage for many names starts ~2020–2022; compute SPY SMA on full IB SPY history then align to the panel index.
- Gap-fill detail: [daily/2026-08-23](daily/2026-08-23.md) (and archived plan in that folder).

---

## Research status (2026-08-25)

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
| Detector | Classical Edwards/Magee rules; Pine `indicators/pine/ascending_channel_3touch.pine` (v1 unchanged) |
| Backtest keepers | Same-day RS vs SPY **top1**; TTM squeeze-adaptive trail (10%/18%); **ATR hard-stop k≈2.0** (clamped 1.5%–6%); friction 0.25% |
| Soft promote (2026-08-25) | `--require-in-channel` + `--max-channel-span-days 365` — drops ~22% above-resist entries; live ATR stack **n=374 E +2.66% PF 1.96** vs unfiltered **n=496 E +2.29% PF 1.82** |
| Rejected | Hard ADV/ATR floors; SPY SMA50 alone; structure-exit bundle; H3 lower-40% geometry; `entry_mode=reclaim` until pivot look-ahead fixed |
| Long-history (2018-11 → 2026-08-23) | H0 E +1.63% / PF 1.72 (n=496); best ATR k=2.0 E **+2.29%** / PF **1.82**; density still mostly **2024–2026** |
| Live monitor | Nightly Windows task: Alpaca multi-symbol 1d refresh → pivot-confirm scan → Telegram. First bat run 2026-08-25: **HOOD** trigger (as_of 2026-08-21) |
| Reports | `reports/ascending_channels/`; TV watchlist / trendline alerts per `docs/features/tv_channel_trendline_alert.md` (verify left-endpoint times — TV snap bug) |

### What is frozen vs in motion

- **Frozen / prefer ship:** Phase 6b near-KEEP blend; channel-touch edge-v2 keepers + nightly cron.
- **Not fishing further (same windows):** Edge-hunt Sharpe>1 on Phase 5/6/6c stock overlays; reclaim entry until look-ahead fixed.
- **Ops cadence:** Post-close EOD scan on stored bars for large universes — do not stream full universe via IB.

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
