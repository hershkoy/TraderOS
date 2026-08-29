# Channel-touch H2 resistance-break — 2026-08-29

## Why MGNI 2019-11-15 was not a trade

Live entry is **post-H2 L3** (arm at H2, then first from-above support tag). On MGNI:

- TV "L3" ~2019-11-15 is **L2** on the detector (`L1=2018-12-24 L2=2019-11-18 H2=2020-01-09`), or an interior low on `L1=2019-05-30 L2=2019-10-10`. It prints **before H2**, so `l3_touch` does not buy it.
- The taken 2020-03-06 Touch 3 is the first post-H2 support tag (`L1=2019-05-30 H2=2020-02-20`). Same-day ATR-ceiling hard stop into the COVID crash — that stop is the correct one.

## Idea

Today a **close above resistance after H2 cancels** the L3 wait ("breakout, not pullback"). Invert that: **fill the breakout** at the rail (+slip), still cancel if support has already closed through. Rebuy/hold-through unchanged. Detector v1 unchanged. Nightly default **off**.

MGNI example: H2 2020-08-11, first close>resist **2020-10-13** @ 8.35 (user's ~10-10). Trail **+4.3% in 2 days** — not the Nov–Dec melt-up (10% trail from a small peak). Span **560d** so it fails `--max-channel-span-days 365`. RSI 85.

## Setup

Rescan unique names from `channel_touch_trades_raw_20260828_194314.csv` (2099 + SPY), same L3 stack plus `--h2-resist-break`. Wall-clock **62s**.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_h2_break.py
```

Breakouts skip in-channel / RSI<=50 (they are defined as above the rail, often overbought). Span 365 is applied as a quality gate on the sleeve.

## Results (0.25% friction)

| book | n | E | PF | notes |
|------|---|---|----|--------|
| L3 keeper CSV (RS top1) | 1024 | +1.77 | 1.62 | frozen |
| Resist-break, no span cap | 4284 | +2.95 | 2.26 | all year buckets + |
| Resist-break, span<=365 | 2253 | +2.80 | 2.21 | all year buckets + |
| **Keeper + break span365, re-RS top1** | **1467** | **+2.26** | **1.86** | 771 breakouts kept; all years + |

2022-23 combined E +0.91 still +. Median still negative (fat tail). Fill is close-through then limit at the rail (same class of wick fill as L3).

## Gate

Breakout-only PF+E beat the L3 keeper on the full window and every year bucket. Adding the span365 sleeve to the frozen keeper and re-ranking RS also lifts E and PF.

**Soft-promote as a second book, not a nightly replacement.** L3 nightly stays support-tag. `--h2-resist-break` is research/optional. Do not apply `--require-in-channel` or `--max-rsi 50` to breakouts. Do not treat MGNI Oct-13 as a moonshot example.

## Code

- [`scripts/research/backtest_channel_touch_trades.py`](../../../scripts/research/backtest_channel_touch_trades.py): `--h2-resist-break`
- [`scripts/research/backtest_channel_touch_h2_break.py`](../../../scripts/research/backtest_channel_touch_h2_break.py)
- Tests: `tests/unit/test_l3_touch_entry.py`

## Artifacts

- Raw breakouts (no span cap): `reports/ascending_channels/channel_touch_h2_resist_break_20260829_204905.csv`
- Span≤365 sleeve: `reports/ascending_channels/channel_touch_h2_break_span365.csv`
- Keeper + sleeve (pre RS-cap, 3277 rows): `reports/ascending_channels/channel_touch_h2_break_keeper_plus_span365.csv`
- Interactive HTML (same tester as L3 current-best):
  - Sleeve: `reports/ascending_channels/current_best/1d_h2_resist_break.html` (`..._fric0.25_h2_break_span365_20260829_205717.html`) — default max/day=all, n=2253 E +2.80 PF 2.21
  - Combined: `reports/ascending_channels/current_best/1d_keeper_plus_h2_resist_break.html` (`..._keeper_plus_h2_break_span365_20260829_205726.html`) — default max/day=1, n=1467 E +2.26 PF 1.86
- Rebuild books: `python scripts\research\export_h2_break_report_books.py`
