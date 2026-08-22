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

## Decision (after 20-sym smoke)

**Continue**, do not kill.

1. Setup-level signal on 300 names looks **directionally positive** and year-stable.
2. Execution layer needs a **much larger IB 15m universe** (100–500+) before judging PF/MDD — 20 names / 13 trades is not enough.
3. Before trusting portfolio metrics: expand beyond A-tickers; keep split filters; optionally add daily-only “signal backtest” with strategy exits without 15m entry.

## Expansion runs (same day, post-opt)

### C) Full ALPACA daily setup hunt (~2203)

```bat
python scripts\data\find_weekly_bigvol_examples.py --no-liquid-only --workers 4 --outdir reports\examples\universe_full
python scripts\data\score_weekly_bigvol_examples.py reports\examples\universe_full\weekly_bigvol_full_setups_20260822_104401.csv
```

Timing: **6m 29s** total (load 5m 48s, analyze 26s) vs ~21.5 min for 300 pre-opt.

| Metric | 300 A-slice | Full ~2203 |
|--------|-------------|------------|
| Symbols with data | 300 | 2203 |
| Ignitions | 2280 | 16273 |
| Full setups (raw) | 387 | 2946 |
| Unique symbol+confirm | 212 | 1568 |
| Clean unique (|fwd_4w|&lt;45%) | 198 | **1451** |
| Clean fwd_4w median / win% | +2.26% / 57.6% | +0.68% / 54.1% |
| Clean fwd_13w median / win% | +5.49% / 64.6% | **+1.77% / 55.3%** |
| Clean mean 13w | +6.51% | +2.76% |
| Confirms by year | 69/73/70 | 514/609/445 (2023/24/25) |

Read: sample size now **clears the ~300 setup bar**. Edge stays **positive but smaller** once A-ticker bias is removed — still directionally supportive at setup level (naive hold, not strategy exits).

### D) IB 15m Backtrader — 100 symbols

```bat
python backtrader_runner_yaml.py --strategy weekly_bigvol_ttm_squeeze --universe --max-symbols 100 --provider IB --timeframe 15m --fromdate 2022-01-01 --todate 2025-11-26 --quiet --workers 4
```

Results: `reports/weekly_bigvol_ttm_squeeze_universe_backtest_20260822_104158/`

| Metric | 20-sym smoke | 100-sym |
|--------|--------------|---------|
| Timing | ~3.0 min / **48.6s** opt | **4m 26s** (workers=4) |
| Successful symbols | 20 | 100 |
| Symbols with trades | 12 | 66 |
| Total trades | 13 | **88** |
| Avg return (all) | +1.67% | +0.27% |
| Avg return (traded) | +2.78% | +0.42% |
| Median return (traded) | n/a | **-0.24%** |
| Mean win rate (traded) | ~50% | ~47% |
| % symbols profitable | 30% | 32% |
| Without ABVX top outlier | flat/slightly neg | traded avg **-0.26%** (87 trades) |

Read: trade count still **below ~300** for law-of-large-numbers. Execution expectancy looks **fragile / near-flat** once the ABVX outlier is removed. Setup-level hold edge does **not** automatically translate to the 15m POC / 15% stop / 30% target path on this A… slice.

### E) IB 15m Backtrader — 300 symbols

```bat
python backtrader_runner_yaml.py --strategy weekly_bigvol_ttm_squeeze --universe --max-symbols 300 --provider IB --timeframe 15m --fromdate 2022-01-01 --todate 2025-11-26 --quiet --workers 4
```

Results: `reports/weekly_bigvol_ttm_squeeze_universe_backtest_20260822_105753/`

| Metric | 100-sym | 300-sym |
|--------|---------|---------|
| Timing | 4m 26s | **13m 38s** (workers=4) |
| Successful / failed | 100 / 0 | **254 / 46** (missing IB 15m) |
| Symbols with trades | 66 | 162 |
| Total trades | 88 | **236** |
| Avg return (all) | +0.27% | +0.20% |
| Avg return (traded) | +0.42% | +0.31% |
| Median return (traded) | -0.24% | **+0.13%** |
| Mean win rate (traded) | ~47% | ~49% |
| % traded profitable | 48.5% | **51.9%** |
| Without ABVX outlier | traded avg -0.26% | traded avg **+0.04%** (235 trades) |

Notes:
- Universe came from **combined ticker list**, not IB-15m coverage — 46/300 failed (no IB 15m / only 1h). Fixed afterward: `get_available_symbols` WHERE-clause spacing bug + universe now filters by provider+timeframe.
- Trade count (236) still shy of ~300; with IB-15m-only universe (~1478) the next run should clear that bar.

### F) IB 15m Backtrader — 500 symbols (provider-filtered)

```bat
python backtrader_runner_yaml.py --strategy weekly_bigvol_ttm_squeeze --universe --max-symbols 500 --provider IB --timeframe 15m --fromdate 2022-01-01 --todate 2025-11-26 --quiet --workers 4
```

Results: `reports/weekly_bigvol_ttm_squeeze_universe_backtest_20260822_112103/`

| Metric | 300-sym (mixed univ) | 500-sym (IB 15m only) |
|--------|----------------------|------------------------|
| Timing | 13m 38s | **20m 52s** (workers=4) |
| Successful / failed | 254 / 46 | **500 / 0** |
| Symbols with trades | 162 | 333 |
| Total trades | 236 | **495** (clears ~300 bar) |
| Avg return (all) | +0.20% | +0.16% |
| Avg return (traded) | +0.31% | +0.24% |
| Median return (traded) | +0.13% | **+0.04%** |
| Mean win rate (traded) | ~49% | ~47% |
| % traded profitable | 51.9% | **50.2%** |
| Mean PF (finite) | ~0.58 | ~0.57 |
| Without ABVX outlier | traded avg +0.04% | traded avg **+0.11%** (494 trades) |

Read: with **495 trades**, sample size is finally usable. Execution expectancy stays **near zero** (median ~flat, ~50% symbols profitable, PF &lt; 1). ABVX still skews the mean but does not change the flat median story. Setup-level hold edge is **not** surviving the 15m POC + stop/target path at portfolio scale.

## Assessing-strategies scorecard (not win-rate-first)

For a **rare-signal / fat-right-tail** system, gates follow `docs/assessing_strategies.md`: trade count, PF, expectancy, payoff (avg win / |avg loss|), MDD — win rate is secondary (30–40% can be fine if winners are large).

### Setup-level naive hold (clean unique confirms, n=1451)

| Metric | fwd 4w | fwd 13w |
|--------|--------|---------|
| Win rate | 54.1% | 55.3% |
| Median / mean | +0.68% / +1.03% | +1.77% / +2.76% |
| Avg win / avg loss | +7.34% / -6.40% | +14.21% / -11.39% |
| Payoff (AvgWin/\|AvgLoss\|) | 1.15 | **1.25** |
| Expectancy / setup | +1.03% | **+2.76%** |
| PF proxy | 1.35 | **1.54** |
| p90 / p95 | +12.0% / +16.3% | **+23.7% / +33.4%** |
| Share &gt; +20% / +30% | 2.8% / 1.0% | **13.2% / 6.3%** |

Year expectancy (13w): 2023 +1.40% (PF~1.25), 2024 +4.14% (PF~1.92), 2025 +2.38% (PF~1.45).

**Setup read:** Positive expectancy and PF&gt;1.5 at 13w, with a real right tail (p95 ~+33%). Not “lottery ticket every time,” but **asymmetric enough** that cutting winners early would destroy the edge.

### Execution BT (500 IB 15m, fixed −15% / +30%)

| Metric | Value | Gate |
|--------|-------|------|
| Trade count | 495 | OK (~300+) |
| Approx trade WR | ~46% | OK for trend style |
| Median / mean PF (symbol) | **0.00 / 0.57** | Fail (&lt;1.5) |
| Mean / median symbol return | +0.24% / +0.04% | Flat |
| Symbol AvgWin / \|AvgLoss\| | 2.12% / 1.64% (~1.29x) | Weak asymmetry |
| Expectancy proxy $/trade | ~$164 (~$75 w/o ABVX) | Tiny vs capital |
| Symbols &gt; +20% | **0.2%** | Right tail almost gone |
| MDD | not in prior CSV | need enriched runs |

**Gap diagnosis:** Implementation uses **fixed +30% take-profit** (and −15% stop) plus 10w MA exit. That **caps** the setup’s p95 (~+33% at 13w hold) and truncates the rare massive winners this strategy is supposed to harvest. README describes trend exits without fixed TP; code defaults disagree.

**Code fix for ablation:** `take_profit_pct &lt;= 0` now disables fixed TP (`profit_target=None`). Universe CSV now stores `avg_win`, `avg_loss`, `max_drawdown_pct`. CLI: `--take-profit-pct 0 --stop-loss-pct 0.15`.

## Updated decision

**Continue research on the setup; do not promote the current fixed-TP execution path.**

| Layer | Verdict |
|-------|---------|
| Setup-level (full ALPACA) | Keep — PF~1.54, exp +2.76%/13w, right tail present |
| Execution (IB 15m, −15%/+30%) | Fail PF/payoff gates; TP likely kills the rare-winner thesis |

## Next actions

1. Ablation: IB 15m BT with **`--take-profit-pct 0`** (stop + 10w MA only) — score PF, expectancy, payoff, MDD.
2. If still flat: try wider stop and/or daily confirm → next-open (no 15m POC).
3. Only after a positive execution variant: robustness grid / walk-forward.

## Ablation G — no fixed take-profit (stop 15% + 10w MA)

```bat
python backtrader_runner_yaml.py --strategy weekly_bigvol_ttm_squeeze --universe --max-symbols 200 --provider IB --timeframe 15m --fromdate 2022-01-01 --todate 2025-11-26 --quiet --workers 4 --take-profit-pct 0 --stop-loss-pct 0.15
```

Results: `reports/weekly_bigvol_ttm_squeeze_universe_backtest_20260822_121354/` (same first-200 IB 15m names as baseline overlap).

| Metric (traded symbols) | Baseline −15%/+30% | **No fixed TP** |
|-------------------------|--------------------|-----------------|
| Trades | 188 | 131 |
| Mean / median return | +0.30% / −0.03% | **+4.24% / +0.85%** |
| Mean WR | ~47% | ~66% |
| Symbol AvgWin / \|AvgLoss\| | 2.43% / 1.80% (1.35x) | **7.17% / 1.56% (4.61x)** |
| Gross $ PF proxy (sum avg_win / sum \|avg_loss\|) | weak | **~9.15** |
| % traded &gt; +10% / +20% / +30% | 0.8% / 0.8% / 0.8% | **11.5% / 5.3% / 3.8%** |
| Best symbol | ABVX +44% | ABVX **+96%** |
| Mean return without top | −0.04% | **+3.53%** |
| Mean max DD% | n/a in old CSV | **4.64%** (median 3.3%) |

Note: many single-winner symbols report `profit_factor=inf` (87/131) or `0` (all losers); use **dollar gross PF proxy** and payoff ratio, not the broken median PF field.

**Ablation read:** Removing the +30% cap restores the rare-winner shape the setup scorecard predicted. Payoff and right-tail mass jump; mean stays positive without ABVX. Sample still small (131 trades) — expand to 500 (running).

### G2 — 500-symbol no-TP confirmation

```bat
python backtrader_runner_yaml.py --strategy weekly_bigvol_ttm_squeeze --universe --max-symbols 500 --provider IB --timeframe 15m --fromdate 2022-01-01 --todate 2025-11-26 --quiet --workers 4 --take-profit-pct 0 --stop-loss-pct 0.15
```

Results: `reports/weekly_bigvol_ttm_squeeze_universe_backtest_20260822_123554/` (20m 35s).

| Metric | Baseline 500 (−15%/+30%) | **No-TP 500** |
|--------|--------------------------|---------------|
| Trades | 495 | **333** (≥300 OK) |
| Mean / median return (traded) | +0.24% / +0.04% | **+3.18% / +0.88%** |
| Mean WR (traded) | ~47% | ~65% |
| Symbol payoff AvgWin/\|AvgLoss\| | ~1.29x | **~4.02x** |
| Gross $ PF proxy | ~0.57 mean PF | **~7.66** |
| % traded &gt; +10% / +20% / +30% | ~1.5% / 0.2% / 0.2% | **7.8% / 2.7% / 1.8%** |
| Mean return without top | +0.11% | **+2.91%** |
| Mean / median max DD% | n/a | **4.16% / 3.13%** |
| Expectancy proxy $/trade | ~$75–164 | **~$3.2k** (risk-sized per symbol run) |

**Gate read vs assessing_strategies:** trade count OK; expectancy and payoff strongly positive; gross PF proxy healthy; MDD mild on this per-symbol framing. Fixed +30% TP remains **rejected**. Default execution path for further work: **stop + MA exit, no fixed TP**.

### Updated next actions

1. Optional stop-width sweep (e.g. 10%/15%/20%) with TP off — watch payoff vs MDD.
2. Then robustness / walk-forward on the no-TP path.
3. Do not reintroduce a tight fixed TP unless payoff stays ≥~2x after costs.
