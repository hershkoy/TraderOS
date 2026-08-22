# Edge hunt Phase 1 - scorecard vs SPY buy-and-hold

Date: 2026-08-22

## Window / data

- Evaluation window: `2018-11-01` -> `2025-11-25`
- SPY ALPACA 1d bars used: 2018-11-01 -> 2025-11-25 (n=1343). Earlier than ~2018-11 may be unavailable on current Alpaca feed.
- Edge gate: Sharpe >= SPY and/or better MDD with CAGR within ~2pp of SPY (or higher CAGR).
- Cross-sectional results use current SPX membership (survivorship bias) - optimistic.

## Scorecard

| name | CAGR | Sharpe | MaxDD | Calmar | Vol | TotRet | Invested% | xsCAGR | xsSharpe | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY_buy_hold | 13.75% | 0.73 | -25.38% | 0.54 | 18.89% | 148.53% | 100% | +0.00% | +0.00 | BENCHMARK |
| SPY_SMA200_abs_mom | 3.60% | 0.33 | -25.88% | 0.14 | 11.05% | 28.43% | 65% | -10.15% | -0.40 | FAIL |
| XS_mom_12_1_gross | 6.16% | 0.34 | -29.03% | 0.21 | 18.37% | 52.55% | 43% | -7.59% | -0.39 | FAIL |
| XS_mom_12_1_net_10bps | 6.02% | 0.33 | -29.09% | 0.21 | 18.37% | 51.18% | 43% | -7.73% | -0.40 | FAIL |
| WeeklyBigVol_portfolio | 8.82% | 0.73 | -12.51% | 0.71 | 12.07% | 81.77% | 48% | -4.93% | +0.00 | FAIL |

## Notes per candidate

- **SPY_buy_hold**: 100% SPY
- **SPY_SMA200_abs_mom**: Month-end close>SMA200; cash=0%; next-day position
- **XS_mom_12_1_gross**: universe=501 (SPX_intersect_ALPACA_1d); survivorship bias (current SPX list); top_n=20; cost=10bps RT; rebalances=29 [gross]
- **XS_mom_12_1_net_10bps**: universe=501 (SPX_intersect_ALPACA_1d); survivorship bias (current SPX list); top_n=20; cost=10bps RT; rebalances=29 [net]
- **WeeklyBigVol_portfolio**: setups=1519 symbols=1068 alloc=10% max_pos=15 stop=15% + MA10 exit; source=weekly_bigvol_full_setups_20260822_104401.csv; closed_trades~=173 stops=5 ma_exits=164

## Timings

- load_spy: 0.0s
- bh: 0.0s
- abs: 0.0s
- xs: 14.1s
- bigvol: 4.4s
- total: 18.5s

## Promotion

**No Phase-1 candidate cleared gates.**

Near-miss (much better MDD, but CAGR too far below SPY):
- `WeeklyBigVol_portfolio`: CAGR=8.82% Sharpe=0.73 MDD=-12.51% (invested 48%)

### Weekly BigVol IB 15m execution (500-sym, no fixed TP)

Completed same day: `reports/weekly_bigvol_ttm_squeeze_universe_backtest_20260822_130007/` (21m 54s, workers=4).

| Metric (traded symbols) | Value |
|-------------------------|-------|
| Symbols / traded / trades | 500 / 333 / **333** |
| Mean / median return | +3.18% / +0.88% |
| Mean WR | ~65.5% |
| Gross $ PF proxy | ~7.66 |
| Mean max DD% | ~4.16% |
| Mean without ABVX | +2.90% |
| % traded > +10% / +20% / +30% | 7.8% / 2.7% / 1.8% |

Per-symbol no-TP execution clears sample-size and PF-proxy gates vs the rejected fixed-TP path, but **portfolio equity vs SPY still fails CAGR gate** (see scorecard above). Do not promote as a SPY-beater on this window.

Next data to add for Phase 2 dual momentum: daily **SHY** (or BIL) and **EFA**/**VXUS**, plus ideally longer SPY history pre-2018 if available. Also fetch **QQQ** daily if testing risk-on Nasdaq sleeve.

## Data ingest update (2026-08-22, IB Gateway)

Pulled via `utils/data/fetch_data.py --provider ib --timeframe 1d --bars max --since 2010-01-01` (Gateway `127.0.0.1:4001`).

| Symbol | Provider | Bars | Range |
|--------|----------|------|-------|
| SPY | IB (+ prior ALPACA) | IB ~4184 | 2010-01-04 -> 2026-08-21 |
| QQQ | IB | ~4021 | ~2010-08 -> 2026-08-21 |
| EFA | IB | ~4184 | 2010-01-04 -> 2026-08-21 |
| SHY | IB | ~2274 | **2017-08-04** -> 2026-08-21 (IB HMDS empty before that) |
| BIL | IB | 4021 | 2010-08-26 -> 2026-08-21 (T-bill ETF; better depth than SHY) |

**Issue:** Python process often exits with Windows access-violation (`0xC0000005` / `-1073741819`) *after* successful DB commit — likely `ib_insync` disconnect teardown. Data in TimescaleDB is fine; treat exit code as noisy.

**SHY caveat:** only back to 2017-08 on this IB subscription; dual momentum absolute filter on SHY is limited before that (use BIL or cash=0% for earlier years).

### TradingView Desktop dump (same day)

Looped the loaded main-series bars via CDP (TV MCP `data_get_ohlcv` only returns the *latest* 500 of the series; paging older bars requires index walks on the same API). After scrolling the chart to ~2008 to force history load:

| Symbol | Provider | Bars | Range |
|--------|----------|------|-------|
| SHY | TRADINGVIEW | ~5300 | **2005-07-28** -> 2026-08-21 |

Artifact: `temp/shy_tv/shy_all_bars.json` + ingest `temp/ingest_shy_tv.py`.
