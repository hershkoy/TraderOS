# Stock market data coverage — 2026-08-21

Source: TimescaleDB `market_data` (stocks only; options excluded).

## Summary

| Timeframe | Symbols | Bars (approx) | Date range |
|-----------|---------|---------------|------------|
| Daily (`1d`) | **2,203** | ~7.63M | 2017-11-29 → 2025-11-26 |
| 15-minute (`15m`) | **1,478** | ~58.7M (EXPLAIN est.) | 2018-01-02 → 2025-12-02 |
| Weekly | **0** | — | Not stored |

## Data sources / API connections

| Source | Role | How we connect | Persisted in `market_data`? |
|--------|------|----------------|-----------------------------|
| **Alpaca** | Historical OHLCV fetch (primary for daily) | `utils/data/fetch_data.py` via `alpaca-py` `StockHistoricalDataClient`; credentials from env `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET`; default feed **IEX** (SIP tried in some paths) | Yes — provider `ALPACA` |
| **Interactive Brokers (IBKR)** | Historical OHLCV fetch (primary for 15m) | `ib_insync` shared connection via `get_ib_connection()`; host `127.0.0.1`; port from `IB_PORT`, auto-detect, or Gateway default **4001**; client ID random/configurable | Yes — provider `IB` |
| **TradingView** | Live chart / indicator verification (not bulk ingest) | Workspace MCP `tradesdontlie/tradingview-mcp` in `.cursor/mcp.json`; node server under `C:\Users\Hezi\tradingview-mcp`; CDP on port **9222** (`TV_MCP_PORT`) | No — chart/API read only |

### Ingest vs verify

- **Ingest path:** `utils/data/fetch_data.py` and `utils/data/update_universe_data.py` (`--provider alpaca|ib`) write bars into TimescaleDB.
- **TradingView path:** MCP tools (`data_get_ohlcv`, `chart_*`, screenshots, Pine) for cross-checking signals and setups against live charts; does not replace Alpaca/IB as the store of record.

## Details

- **Providers (daily):** ALPACA 2,203 symbols (~7.63M bars); IB 1 symbol (~3.8K bars)
- **Providers (15m):** IB 1,478 symbols (2018-01-02 → 2025-12-02); ALPACA 1 symbol (`AEO`, ~1.9K bars, 2020-07-27 → 2020-09-15 only)
- **`ticker_universe`:** 6,072 tickers; all 2,203 daily and all 1,478 15m symbols are in the universe
- **15m notes:** Broad equity set (not just liquid ETFs); sample checks found `AAPL`/`TSLA` (~51.5K IB bars each) but no `SPY`/`QQQ`/`NVDA`/`IWM` 15m bars
- **Other stock timeframes present:** `1m`, `1h` (not inventoried in detail)
- **Weekly:** no `1w` / weekly timeframe in `market_data`; weekly would need resampling from daily or separate ingest

## Notes

- Full-table `COUNT(DISTINCT …)` / heavy `GROUP BY` on this hypertable is slow and can hit shared-memory / disk pressure; prefer per-`timeframe` (and per-`provider`) distinct counts, `MIN`/`MAX(ts)`, and `EXPLAIN` row estimates for bar totals.
- Queried via `utils.db.timescaledb_client.get_timescaledb_client()`.
