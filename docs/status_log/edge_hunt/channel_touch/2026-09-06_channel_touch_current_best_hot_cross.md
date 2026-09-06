# hot-cross into `current_best/` — 2026-09-06

`current_best/` is the **live-executable, causal** gate, even if the book loses. That is the number the next 1d idea has to beat.

## What went in

Buy-now `--intraday-trigger hot-cross` `--hot-cross-fill lerp85` unique span365 + shakeout from [the 15m-trigger scan](2026-09-06_channel_touch_1d_15m_trigger.md):

| Book | n | E | PF |
|------|---|----|----|
| Unique span365 | **4079** | **−0.19** | **0.94** |

HTML generated from `channel_touch_full_h2_break_span365_unique_20260906_055000.csv` (friction 0.25%, max/day=all, IB-prefix SPY from first trade 2019-04-24). Stamp: `...interactive_fric0.25_1d_hot_cross_lerp85_20260906_212818.html`.

Copied (not hard-linked) to:

- `reports/ascending_channels/current_best/1d_hot_cross.html`
- `reports/ascending_channels/current_best/1d_channel_touch.html` (same bytes)

Stable trades: `reports/ascending_channels/channel_touch_1d_hot_cross.csv`.

15m L3 wait-12 signal-close **n=1744 E +0.13 PF 1.22** stays the 15m keeper. Clip 1d HTML stays in `1d_unrealistic/`.

## What this is not

- **Not a promote.** E < 0, PF < 1, 2022-23 fails, drop-top-3 worse.
- **Not the nightly fill.** Nightly still uses the unrealistic same-bar rail clip.
- **Not next-mid** (n=1986 E +2.24 PF 1.89) — that overlay still keys off rail X after the EOD close.

Grade the next live-executable 1d scan against **n=4079 E −0.19 PF 0.94**, not against clip n=4558 E +2.62 PF 2.10.
