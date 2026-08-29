# Same-bar close leak: prior-bar features + daily 15m hybrid (2026-08-29)

Wick limit fills cannot use the fill bar's close (RSI, %B, `close_loc`, range, SMA distance). The rule is **only completed bars with timestamp &lt; fill time**. On daily that is IB 15m of the same session up to (not including) the first 15m support tag — not "yesterday's daily close" only.

Detector v1 and `find_h2_l3_setups` are unchanged. Nightly stays **pivot**.

## What shipped

- `--feature-asof auto|prior-bar|entry-bar`. Auto = prior-bar for `l3_touch` on 15m native and `--intraday-fill 15m`.
- `--intraday-fill 15m` on daily `l3_touch`: first IB 15m from-above tag that session; skip names with no 15m (no prior-day-only fallback). Features at 15m `i-1`. ATR/ADV from the prior **daily** bar. RS/SPY from last completed 15m (or prior daily session if the panel is daily).
- 15m native: same lag (`entry_i - 1`). Fills unchanged vs Loop 5; feature/`rs_spy_*` as-of the prior 15m bar.
- `min_close_loc` now means the **previous completed 15m bar** closed in the upper half, not "this tag bar will close strong."

## Unit tests

Tag on the 4th RTH 15m bar; stock features match bar 3, not bar 4. 15m `l3_touch` snapshot is `entry_i - 1`. Missing 15m panel returns no trades (no leaky daily-close fallback). Daily SPY enrichment with `buy_time` does not use the fill session's close.

## A/B (300 IB 15m names, 2018-11-01 → 15m end 2025-12-02)

Gate: PF/E, year splits. Friction 0.10 on 15m, 0.25 on daily. Wall-clock: 15m lagged **356s** (cache-hit load); daily hybrid **581s** (15m load 257s + scan 161s); same-300 daily leaky **9s** (ALPACA_IB cache).

| stack | n | E% | PF | notes |
|-------|---|-----|-----|--------|
| Loop 5 15m L3 wait12, leaky fill-bar features | 1751 | +0.32 | 1.58 | prior session; all years + |
| **15m L3 wait12, lagged `prior-bar`** | **1751** | **+0.31** | **1.57** | fills same; RS as-of prior 15m. Years: 2018-19 +0.05, 2020-21 +0.24, 2022-23 +0.29, 2024-26 +0.55 |
| leaky `min_close_loc` 0.6 (Loop 1 raw, wait6) | 1261 | +0.41 | 1.85 | **same-bar leak** |
| lagged `min_close_loc` 0.6 (wait12 raw, beyond 0.25 keeper) | 900 | +0.06 | 1.12 | 2024-26 **negative** |
| logistic on lagged features then RS (holdout 2023+) | 216 | +0.41 | 1.86 | WF 2022 kept E +0.16 PF 1.26; not a hard gate |
| same-300 daily leaky wick + **entry-day close** wait6/RSI50/in+span365/b0.25 | 314 | +1.73 | 1.63 | 2022-23 +0.07 |
| **same-300 daily hybrid 15m fill, prior-bar features** | **260** | **+1.31** | **1.47** | 2022-23 **+0.02**. Smoke 10 names: 9/11 had 15m (CAT skipped) |

Hybrid does **not** beat the same-universe daily wait-6 / RSI-50 keeper. Do not wire `--intraday-fill` into nightly.

15m Loop 5 research stack is unchanged by the lag (E/PF within noise). Honest `min_close_loc` does not reproduce the leaky wick filter. Logistic on lagged `close_loc`/RSI is a mild raw lift, not a promote.

## Artifacts

- 15m lagged: `reports/ascending_channels/channel_touch_15m_trades_20260829_105511.csv` (raw `..._raw_20260829_105511.csv`)
- Hybrid: `reports/ascending_channels/channel_touch_hybrid_trades_20260829_110642.csv`
- Same-300 daily leaky: `reports/ascending_channels/channel_touch_trades_20260829_111156.csv`

## CLI

```bat
python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --entry-mode l3_touch --min-l3-wait-bars 12 --workers 4 --load-workers 8
python scripts\research\backtest_channel_touch_trades.py --n-symbols 300 --entry-mode l3_touch --intraday-fill 15m --min-l3-wait-bars 6 --max-rsi 50 --require-in-channel --max-channel-span-days 365 --max-beyond-width 0.25 --atr-stop-mult 2.0 --squeeze-adaptive --max-entries-per-day 1 --friction-pct 0.25 --fallback-provider IB --workers 4 --load-workers 8
```

`--feature-asof entry-bar` reproduces the leak for A/B. Default auto lags 15m/`--intraday-fill` `l3_touch` only.
