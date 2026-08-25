# Channel-touch geometry quality audit (2026-08-25)

## Goal

Check whether "bad" detector usage (entries already above resistance, multi-year stale channels like IDCC) is poisoning the edge — without rewriting `find_channels`. Use post-hoc / CLI filters only; freeze detector v1.

## Setup

- Same detector: `find_ascending_channels.find_channels`
- Keeper stack: RS top1/day, squeeze trail 10%/18%, ATR hard stop k=2.0 (clamped), friction 0.25%
- Window: 2018-11-01 → 2026-08-23, ALPACA 1d, warm parquet cache
- Wall-clock: unit tests + ATR baseline scan ~47s; filtered rescan ~38s (cache hit)

## Separate bug: TV draw snap (not detector)

Long multi-year rails can have left `t1` snapped forward on TradingView while `p1` stays at the old low — lines float under recent candles (IDCC visual). Documented in `docs/features/tv_channel_trendline_alert.md`. Fix by verifying `draw_get_properties` vs `watchlist_channels_draw.json`.

## How common is "bad" geometry? (ATR keeper, n=496 RS-top1)

| Issue | Rate |
|-------|------|
| Entry above resist (`channel_pos>1`) | **22.4%** (111) |
| Entry lower 40% of channel | 38.7% |
| Channel span > 365d | 36.7% |
| Channel span > 730d | 5.4% |
| Median span / age at buy | 302d / 298d |

IDCC: buy 348.43 on 2026-08-17, `channel_pos=3.58`, span 1001d — classic above-resist loser (hard/eod), not a silent winner.

## Bucket expectancy (ATR keeper, net of 0.25%)

| Bucket | n | E% | PF |
|--------|---|----|----|
| **in_channel** (pos<=1) | 385 | **+2.76** | **2.02** |
| above_resist (pos>1) | 111 | +0.65 | 1.21 |
| mid 0.4–1.0 | 193 | **+3.52** | **2.24** |
| lower40 | 192 | +1.99 | 1.77 |
| span<=365d | 314 | +2.57 | 1.91 |
| span>365d | 182 | +1.81 | 1.66 |

**Key:** above-resist is a **drag**, not the secret sauce. Mid-channel beats lower-40%. Strict classical "must buy near support" (`geometry_h3` / lower40+width/slope) **hurts**.

## A/B filters (post-hoc on ATR RS-top1 set)

| scenario | n | E% | PF | vs baseline |
|----------|---|----|----|-------------|
| baseline ATR | 496 | 2.29 | 1.82 | — |
| in_channel | 385 | 2.76 | 2.02 | lift |
| span_le_365 | 314 | 2.57 | 1.91 | lift |
| **in_channel+span365** | 251 | **3.26** | **2.21** | best post-hoc |
| geometry_h3 | 127 | 1.54 | 1.64 | **reject** |

## Live CLI confirmation (filter **before** RS top1)

Correct promotion path: quality-filter raw trades, then same-day RS.

```bat
python scripts\research\backtest_channel_touch_trades.py --all-symbols --squeeze-adaptive --atr-stop-mult 2.0 --require-in-channel --max-channel-span-days 365 --max-entries-per-day 1 --friction-pct 0.25 --workers 4 --load-workers 8 --start 2018-11-01 --end 2026-08-23
```

| Stack | n | E% | PF |
|-------|---|----|----|
| ATR keeper (no geom filter) | 496 | 2.29 | 1.82 |
| **+ require-in-channel + span<=365** | **374** | **2.66** | **1.96** |

(n higher than post-hoc 251 because filtering before RS frees same-day slots for other names.)

## Verdict

1. **Do not rewrite** `find_channels` swing logic yet — the edge improves with **entry quality gates**, not stricter pivots.
2. **Soft promote:** `--require-in-channel` and preferably `--max-channel-span-days 365` on the live stack.
3. **Reject** promoting H3 lower-40% geometry bundle (confirmed again under ATR).
4. IDCC-style charts are expected under unfiltered detector; filters drop them.
5. Mid-channel entries (pos 0.4–1.0) are the sweet spot after pivot lag — tightening to support-only would cut the best bucket.

## Code / artifacts

- Audit: `scripts/research/audit_channel_touch_quality.py`
- Flags on `backtest_channel_touch_trades.py`: `--require-in-channel`, `--max-channel-span-days`, `--max-channel-age-days`; geometry cols now in `REPORT_COLS`
- Tests: `tests/unit/test_backtest_channel_touch_trades.py` (4 passed)
- Trades: `reports/ascending_channels/channel_touch_trades_20260825_014349.csv` (ATR baseline), `..._014435.csv` (filtered)
- A/B CSV: `reports/ascending_channels/channel_touch_quality_ab_20260825_014359.csv`
- Interactive TV report (filtered best stack): `reports/ascending_channels/channel_touch_tv_report_interactive_rs_top1_default_fric0.25_atr_k2_inchannel_span365_20260825_210115.html`
