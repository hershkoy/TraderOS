## Learned User Preferences

- Prefer workspace-scoped MCP config (project `.cursor/mcp.json`) over global always-on Cursor MCP for tools like TradingView
- When validating a strategy, prefer finding historical examples in TimescaleDB before live signal hunting or full-universe backtests
- Persist coverage inventories, strategy plans, and phase status under `docs/status_log/` (Weekly BigVol under `docs/status_log/weekly_bigvol/`; edge hunt under `docs/status_log/edge_hunt/`)
- For market-data coverage questions, focus on stocks and exclude options unless options are requested
- For long multi-symbol hunts or backtests, record wall-clock timings in the relevant status log
- For rare-signal strategies, assess with PF/expectancy/payoff/MDD per `docs/assessing_strategies.md`; do not treat win rate as the primary gate
- Edge-hunt promotion vs SPY is risk-adjusted: Sharpe > 1.0, Sharpe >= SPY, and MDD better than SPY (CAGR is reported, not the ranking objective)
- After Phase 5 stock-level signals and Phase 6c VIX overlays failed to clear or lift the gate, prefer freezing and productionizing Phase 6b near-KEEP `Blend_VT60_BV40` over more Sharpe>1 fishing on the same window
- Validate the BigVol backtester visually via TradingView entry/exit order markers on discrete sleeve trades before trusting vectorized fills
- Classical chart-pattern detectors must follow Edwards & Magee / Murphy / Bulkowski rules: ascending channel ≠ ascending triangle (flat top); require distinct swings with significant intervening moves, ≥2 confirmed opposite-boundary touches, and reject repeated support/resistance violations
- Prefer original Pine/Python pattern implementations over copying proprietary or CC BY-NC-SA chart-pattern libraries (e.g. Trendoscope ACP)

## Learned Workspace Facts

- Stock OHLCV lives in TimescaleDB `market_data`; use `utils.db.timescaledb_client.get_timescaledb_client()`
- Daily (`1d`) coverage is about 2,203 symbols (~7.63M bars), mainly ALPACA, roughly 2017-11-29 to 2025-11-26; no weekly timeframe is stored (resample from daily)
- 15m coverage is about 1,478 symbols (~58.7M bars estimated), mainly IB, roughly 2018-01-02 to 2025-12-02; ALPACA 15m is effectively unused aside from leftover `AEO`; for 15m universe backtests prefer the IB 15m symbol set over the full daily universe to avoid empty loads
- Heavy full-table `COUNT`/`GROUP BY` on `market_data` is slow and can hit shared-memory or disk pressure; prefer per-timeframe/provider queries and `EXPLAIN` estimates
- TradingView MCP (`tradesdontlie/tradingview-mcp`) is enabled only for this workspace via `.cursor/mcp.json`, with the server under the user home `tradingview-mcp` folder and CDP on port 9222; it exposes price/quote/indicator/chart control, not historical fundamentals/financials; Polygon Stocks Basic also excludes Financials & Ratios (paid add-on)
- Weekly BigVol + TTM Squeeze lives in `strategies/weekly_bigvol_ttm_squeeze.py` / `weekly_bigvol_components.py`; example hunter is `scripts/data/find_weekly_bigvol_examples.py`; multi-symbol loads use `utils/data/ohlcv_loader.py`; default research execution is no fixed TP + ~10% stop (plus MA exit); `scanner_runner.py --scanner squeeze` is TTM zero-cross only; W-FRI resample + squeeze heuristics can lag TradingView LazyBear flips by weeks, and split-unadjusted forward returns can distort outcomes
- Edge-hunt vectorized research lives in `utils/research/` and `scripts/research/` (not `strategies/`); status under `docs/status_log/edge_hunt/`; Phase 4 KEEP is `Blend_SPY70_BV30`; Phase 5 XS mom + swing MR failed; Phase 6 vol-target near-miss and CTA weak; Phase 6b near-KEEP is `Blend_VT60_BV40` (vol-target SPY + BigVol); Phase 6c VIX curve ran with no promote / no lift vs plain vol-target; BigVol sleeve trade export and TradingView marker plans live in `scripts/research/export_bigvol_trades.py` with CSVs/plans under `reports/edge_hunt_phase6b/trades/`
- ALPACA daily panel coverage for many names starts ~2020-2022 (wide panel early years mostly empty); compute SPY SMA on full IB SPY history then align to the panel index
- Macro ETFs for overlays (TLT/GLD/USO/UUP/IEF/DBC) were Alpaca-ingested for Phase 6 with history often from ~2020-07; IB can supply CBOE `VIX`/`VIX3M`/`VVIX` when Gateway farms (especially `ushmds`) are connected; Alpaca VIX ETF proxies (VIXY/VXX/etc.) are also available; VX futures history from IB was too thin to use
- `ticker_universe` holds about 6,072 tickers; current daily and 15m symbols are subsets of that universe
- Ascending-channel research: classical detector in `scripts/research/find_ascending_channels.py` with Pine `indicators/pine/ascending_channel_3touch.pine`; reports under `reports/ascending_channels/`
- Largest-breakouts research report: `scripts/research/largest_breakouts_report.py`; outputs under `reports/largest_breakouts/`
