## Learned User Preferences

- Prefer workspace-scoped MCP config (project `.cursor/mcp.json`) over global always-on Cursor MCP for tools like TradingView
- When validating a strategy, prefer finding historical examples in TimescaleDB before live signal hunting or full-universe backtests
- Persist coverage inventories, strategy plans, and phase status under `docs/status_log/` (Weekly BigVol under `docs/status_log/weekly_bigvol/`)
- For market-data coverage questions, focus on stocks and exclude options unless options are requested
- For long multi-symbol hunts or backtests, record wall-clock timings in the relevant status log

## Learned Workspace Facts

- Stock OHLCV lives in TimescaleDB `market_data`; use `utils.db.timescaledb_client.get_timescaledb_client()`
- Daily (`1d`) coverage is about 2,203 symbols (~7.63M bars), mainly ALPACA, roughly 2017-11-29 to 2025-11-26; no weekly timeframe is stored (resample from daily)
- 15m coverage is about 1,478 symbols (~58.7M bars estimated), mainly IB, roughly 2018-01-02 to 2025-12-02; ALPACA 15m is effectively unused aside from leftover `AEO`
- `ticker_universe` holds about 6,072 tickers; current daily and 15m symbols are subsets of that universe
- Heavy full-table `COUNT`/`GROUP BY` on `market_data` is slow and can hit shared-memory or disk pressure; prefer per-timeframe/provider queries and `EXPLAIN` estimates
- TradingView MCP (`tradesdontlie/tradingview-mcp`) is enabled only for this workspace via `.cursor/mcp.json`, with the server under the user home `tradingview-mcp` folder and CDP on port 9222
- Weekly BigVol + TTM Squeeze is implemented in `strategies/weekly_bigvol_ttm_squeeze.py` and `strategies/weekly_bigvol_components.py`; incomplete weekly bars can create false TTM signals versus TradingView
- `scanner_runner.py --scanner squeeze` finds TTM zero-cross only, not the full ignition + trend strategy setup
- Weekly BigVol example hunter is `scripts/data/find_weekly_bigvol_examples.py`; multi-symbol OHLCV loads should use `utils/data/ohlcv_loader.py` (batch SQL, parquet cache, workers)
- Weekly confirm dates from W-FRI resample + squeeze heuristics can lag TradingView LazyBear TTM flips by weeks; split-unadjusted forward returns can badly distort example outcomes
- For 15m universe backtests, prefer the IB 15m symbol set over the full daily universe to avoid empty loads
