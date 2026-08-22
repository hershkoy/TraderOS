## Learned User Preferences

- Prefer workspace-scoped MCP config (project `.cursor/mcp.json`) over global always-on Cursor MCP for tools like TradingView
- When validating a strategy, prefer finding historical examples in TimescaleDB before live signal hunting or full-universe backtests
- Persist coverage inventories, strategy plans, and phase status under `docs/status_log/` (Weekly BigVol under `docs/status_log/weekly_bigvol/`; edge hunt under `docs/status_log/edge_hunt/`)
- For market-data coverage questions, focus on stocks and exclude options unless options are requested
- For long multi-symbol hunts or backtests, record wall-clock timings in the relevant status log
- For rare-signal strategies, assess with PF/expectancy/payoff/MDD per `docs/assessing_strategies.md`; do not treat win rate as the primary gate
- Edge-hunt promotion vs SPY is risk-adjusted: Sharpe > 1.0, Sharpe >= SPY, and MDD better than SPY (CAGR is reported, not the ranking objective)
- After Phase 5 stock-level XS momentum and swing mean-reversion failed those gates, prefer portfolio overlays and vol-targeting over further single-stock signal fishing

## Learned Workspace Facts

- Stock OHLCV lives in TimescaleDB `market_data`; use `utils.db.timescaledb_client.get_timescaledb_client()`
- Daily (`1d`) coverage is about 2,203 symbols (~7.63M bars), mainly ALPACA, roughly 2017-11-29 to 2025-11-26; no weekly timeframe is stored (resample from daily)
- 15m coverage is about 1,478 symbols (~58.7M bars estimated), mainly IB, roughly 2018-01-02 to 2025-12-02; ALPACA 15m is effectively unused aside from leftover `AEO`
- Heavy full-table `COUNT`/`GROUP BY` on `market_data` is slow and can hit shared-memory or disk pressure; prefer per-timeframe/provider queries and `EXPLAIN` estimates
- TradingView MCP (`tradesdontlie/tradingview-mcp`) is enabled only for this workspace via `.cursor/mcp.json`, with the server under the user home `tradingview-mcp` folder and CDP on port 9222
- Weekly BigVol + TTM Squeeze lives in `strategies/weekly_bigvol_ttm_squeeze.py` / `weekly_bigvol_components.py`; example hunter is `scripts/data/find_weekly_bigvol_examples.py`; multi-symbol loads use `utils/data/ohlcv_loader.py`; default research execution is no fixed TP + ~10% stop (plus MA exit); `scanner_runner.py --scanner squeeze` is TTM zero-cross only
- Weekly confirm dates from W-FRI resample + squeeze heuristics can lag TradingView LazyBear TTM flips by weeks; split-unadjusted forward returns can badly distort example outcomes
- For 15m universe backtests, prefer the IB 15m symbol set over the full daily universe to avoid empty loads
- Edge-hunt vectorized research lives in `utils/research/` and `scripts/research/` (not `strategies/`); status under `docs/status_log/edge_hunt/`; Phase 4 KEEP is `Blend_SPY70_BV30`; Phase 5 XS mom + swing MR failed; Phase 6 vol-target is a near-miss, CTA weak, VIX overlay skipped (no VIX data)
- ALPACA daily panel coverage for many names starts ~2020-2022 (wide panel early years mostly empty); compute SPY SMA on full IB SPY history then align to the panel index
- Macro ETFs for overlays (TLT/GLD/USO/UUP/IEF/DBC) were Alpaca-ingested for Phase 6 with history often from ~2020-07; TimescaleDB has no VIX/VXV or VIX futures
- `ticker_universe` holds about 6,072 tickers; current daily and 15m symbols are subsets of that universe
