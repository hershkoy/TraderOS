# Phase 2 — Edge smoke (setup-level + IB 15m execution)

Date: 2026-08-22

## Run timings (wall clock)

| Run | Scope | Elapsed |
|-----|-------|---------|
| Weekly completeness sample | 10 symbols | ~1.4 min (86.7s) baseline; **50.0s** cold batch / **0.1s** warm cache |
| Setup hunt (300 ALPACA daily) | 300 symbols | ~21.5 min (1292s) baseline (pre-opt) |
| Setup hunt smoke | 10 symbols | **15.1s** cold / **6.2s** warm cache |
| IB 15m Backtrader smoke | 20 symbols | ~3.0 min (182s) baseline; **48.6s** with `--workers 4` + quiet weekly |

Going forward, scripts print timing in the summary:
- `find_weekly_bigvol_examples.py` - total / load / analyze / avg per symbol
- `score_weekly_bigvol_examples.py` - total
- `check_weekly_completeness_sample.py` - total
- `backtrader_runner_yaml.py --universe` - total, avg/symbol, per-symbol elapsed, `timing.txt` + `elapsed_seconds` column in CSV

## Runtime optimizations (2026-08-22)

Implemented end-to-end speedups:

1. **Batch OHLCV** - `TimescaleDBClient.get_ohlcv_batch()` (`ANY(%s)` chunks)
2. **Parquet cache + thread workers** - `utils/data/ohlcv_loader.py` (`data/cache/ohlcv/`)
3. **Hunter / completeness** wired to loader (`--no-cache`, `--workers`, `--chunk-size`)
4. **Quiet weekly resample** - skip O(weeks*bars) incomplete-week scan in universe mode
5. **Universe `--workers N`** - process pool over symbols (`run_one_universe_symbol`)

| Before -> After | Result |
|-----------------|--------|
| Completeness 10-sym | 86.7s -> 50s cold / 0.1s cached |
| Hunt 10-sym | (was ~4s/sym load-ish) -> 7.1s load cold / 0.1s cached |
| IB universe 20 | 182s -> **48.6s** (~3.7x) at workers=4 |

Fast commands:

```bat
python scripts\data\find_weekly_bigvol_examples.py --no-liquid-only --max-symbols 300 --workers 4
python backtrader_runner_yaml.py --strategy weekly_bigvol_ttm_squeeze --universe --max-symbols 20 --provider IB --timeframe 15m --fromdate 2022-01-01 --todate 2025-11-26 --quiet --workers 4
```

## Hygiene (quick)

Sample of Phase 1 symbols (UBER, MSFT, NVDA, WMT, …): ~18% of W-FRI weeks have &lt;5 sessions (mostly holidays). Almost no weeks with &lt;4 sessions. Weekly resample quality is acceptable for hunting; holiday incompletes still can shift TTM slightly vs TV.

## A) Setup-level expectancy (ALPACA daily, 300 symbols)

Tool: `scripts/data/find_weekly_bigvol_examples.py --no-liquid-only --max-symbols 300`  
CSV: `reports/examples/universe300/weekly_bigvol_full_setups_20260822_013329.csv`  
Scorer: `scripts/data/score_weekly_bigvol_examples.py`

| Metric | Value |
|--------|-------|
| Symbols with data | 300 / 300 (alphabetical A… slice) |
| Ignitions | 2280 |
| Full setups (raw rows) | 387 |
| Unique symbol+confirm | 212 (146 symbols) |
| Clean unique (split filter) | 198 |
| Median delay ignition→confirm | 16 weeks |
| Clean fwd_4w median / win% | +2.26% / 57.6% |
| Clean fwd_13w median / win% | +5.49% / 64.6% |
| Clean mean 13w | +6.51% |

Confirms by year: 2023=69, 2024=73, 2025=70 (stable count).

**Caveat:** forward returns are naive hold-from-confirm, not MA10/30/stop exits. Split outliers filtered at |fwd_4w|&lt;45%. Slice is A-tickers only (selection bias).

## B) Execution smoke (IB 15m Backtrader, 20 symbols)

```bat
python backtrader_runner_yaml.py --strategy weekly_bigvol_ttm_squeeze --universe --max-symbols 20 --provider IB --timeframe 15m --fromdate 2022-01-01 --todate 2025-11-26 --quiet
```

Results: `reports/weekly_bigvol_ttm_squeeze_universe_backtest_20260822_011440/universe_results.csv`

| Metric | Value |
|--------|-------|
| Symbols | 20 (all succeeded) |
| Symbols with trades | 12 |
| Total trades | 13 |
| Avg return (all) | +1.67% |
| Avg return (traded) | +2.78% |
| Mean win rate (traded) | 50% |
| Median PF (finite) | 0.0 (many single-trade losers) |
| % symbols profitable | 30% |

Skew: **ABVX +44%** dominates the average; without it the smoke is roughly flat-to-slightly-negative.

Strategy requires **15m base** (intraday POC entry + 15% stop / 30% target). ALPACA daily cannot drive the full Backtrader path.

## Gate read vs assessing_strategies.md

| Gate | Setup-level (A) | Execution smoke (B) |
|------|-----------------|---------------------|
| Sample size | ~200 unique confirms — approaching useful | **13 trades — noise** |
| Expectancy | Positive at 13w | Unclear / skewed |
| Win rate | ~58–65% (hold) | ~50% (system exits) |
| PF | n/a (no R exits) | Fragile on this slice |

## Decision

**Continue**, do not kill.

1. Setup-level signal on 300 names looks **directionally positive** and year-stable.
2. Execution layer needs a **much larger IB 15m universe** (100–500+) before judging PF/MDD — 20 names / 13 trades is not enough.
3. Before trusting portfolio metrics: expand beyond A-tickers; keep split filters; optionally add daily-only “signal backtest” with strategy exits without 15m entry.

## Next actions

1. Expand setup hunt to full ALPACA daily (~2200) for &gt;300 unique confirms.
2. Expand IB 15m Backtrader to `--max-symbols 100` (then 300) with same date window.
3. Only then: robustness grid / walk-forward.
