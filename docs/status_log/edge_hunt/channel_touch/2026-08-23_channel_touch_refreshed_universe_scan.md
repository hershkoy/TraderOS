# Channel-touch refreshed-universe scan + TV watchlist (2026-08-23)



## Context

ALPACA `1d` gap-fill completed same evening (see `../../daily/2026-08-23_alpaca_1d_refresh_monday_plan.md`). Ran full-universe channel-touch with data through **2026-08-23** (last bar mostly **2026-08-21**).



## Command

```bat

python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --trail-pct-wide 0.18 --workers 4 --load-workers 8 --end 2026-08-23 --max-entries-per-day 1 --friction-pct 0.25

```

Log: `logs/channel_touch_test_scan.log`



## Wall-clock

- Cold OHLCV load (2217 symbols, cache hits=0 for new end date) + scan: **~328s (~5.5 min)**



## Results (RS top1/day, 0.25% friction)

| Metric | Value |

|--------|-------|

| Trades | 496 |

| Symbols with trades | 453 |

| Win rate | 26.0% |

| Expectancy | **+1.63%** |

| Profit factor | **1.72** |

| Avg win / avg loss | +14.98% / -3.06% |

| Avg hold | 21.4 days |

| Hard stop / trail / trail_wide / eod | 326 / 151 / **2** / 17 |



Artifacts:

- `reports/ascending_channels/channel_touch_trades_20260823_225856.csv`

- `reports/ascending_channels/channel_touch_trades_summary_20260823_225856.txt`

- `reports/ascending_channels/channel_touch_tv_report_interactive_rs_top1_default_fric0.25_20260823_230143.html`



## Vs prior research window (end=2025-11-26)

| Run | End | Filter | Trades | E | PF |

|-----|-----|--------|--------|---|-----|

| Squeeze-adaptive full (status 2026-08-23) | 2025-11-26 | none | 2259 | +1.72% | 1.82 |

| This test | 2026-08-23 | RS top1 + 0.25% fric | 496 | +1.63% | 1.72 |



Not an apples-to-apples compare (RS-top1 + friction cut trade count). Edge still clears the +1% net expectancy bar used in `2026-08-22_channel_touch_edge_improve.md`.



## Recent buys (~last 14d before last bar)

| Symbol | Buy | Outcome (as of 2026-08-21) |

|--------|-----|------------------------------|

| UNIT | 2026-08-20 | still open (eod) |

| SMTC | 2026-08-19 | still open |

| APH | 2026-08-18 | hard_stop |

| IDCC | 2026-08-17 | hard_stop |

| TKO | 2026-08-14 | hard_stop |

| PRKS | 2026-08-13 | still open |

| BFC | 2026-08-12 | hard_stop |

| MSCI | 2026-08-11 | still open |



## Still open through last bar (2026-08-21)

UNIT, SMTC, PRKS, MSCI, QRVO, HCA (+7.1%), EXR, IRM, CRGY (+34%), NSC, RUSHB, APGE (+52%), SRCE, AEE



## TradingView: channels + support alerts

Drew classical channel (support teal / resistance red) for all **18** unique names above on the active layout; added **cross** price alerts at **projected support** (`support_last`).



| Symbol | Alert px | Symbol | Alert px |

|--------|----------|--------|----------|

| UNIT | 9.28 | QRVO | 82.63 |

| SMTC | 107.38 | HCA | 357.77 |

| APH | 144.26 | EXR | 145.90 |

| IDCC | 258.79 | IRM | 122.60 |

| TKO | 180.65 | CRGY | 9.39 |

| PRKS | 46.97 | NSC | 311.92 |

| BFC | 144.79 | RUSHB | 69.58 |

| MSCI | 558.42 | APGE | 90.10 |

| | | SRCE | 77.24 |

| | | AEE | 115.81 |



Geometry JSON: `reports/ascending_channels/watchlist_channels_draw.json`



### Alerts (updated 2026-08-24)

Replaced flat price alerts with **open-ended drawing alerts** on each teal support trendline:

- Type: `drawing` — Price **Crossing** trendline
- Resolution: `1D`
- Frequency via API: `on_bar_close` (REST rejects `once_per_bar`; edit in UI to "Once per bar" if preferred)
- Expiration: `null` (open-ended)
- Bound to `drawing_id` + layout `WSlUWqyb` so the alert follows the sloping line

Always verify `chart_get_state` symbol before creating alerts (race can attach to previous ticker).

## Monitoring stance (unchanged)

Post-close EOD scan on stored bars for ~2.2k symbols; IB only for execution / open positions — do not stream the full universe.

