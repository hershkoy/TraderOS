# Channel-touch `entry_mode=l3_touch` (2026-08-28)

## Why

Pivot-confirm entry buys `L3 + pivot_len` **close**, which can sit mid-channel. EYE Touch 3 in the 0.25 report was **2025-05-01 @ 12.885** (May 1 close), while support had already been tagged in early April. Live plan: draw rails when **H2** prints (L1–H1–L2–H2), fill the first support tag (wick OK) plus slippage, stop from that fill.

Detector **v1 `find_channels` is unchanged**. New setup path: `find_h2_l3_setups` / `find_h2_l3_setups_windowed`.

## Fill rules

- Fit support on two higher lows (L1, L2) with an intervening rally (H1 as a swing; **H1 need not tag the parallel** — EYE Nov 2024 peak is inside a tighter width than Feb/Mar).
- Arm when two resistance touches exist and the completing high (**H2**) is after L2.
- From the next bar, buy the first bar whose range tags support; limit = support × (1 + `--entry-slip-pct`, default 0.1%).
- If a bar makes a **higher high than H2** before the tag, cancel (H2 invalidated).
- Same-bar hard stop still applies if the wick continues through the stop.

Alpaca EYE 2025: H2 **2025-03-25**, first tag **2025-04-04** low 10.505 vs support ~10.44 (`channel_pos` 0.019). Apr 25 low was **12.02** and did **not** tag the L1–L2 rail (TV “L3 Apr 25 ~$10.50” matches the **Apr 4–9** lows, not Apr 25). May 1 @ 12.89 is the old pivot_len close. Rail fill then **trailed out 2025-04-07** (+2.4%) — the old +78% ride was an artifact of the late mid-channel fill.

## Keeper A/B (same stack as beyond-width 0.25)

`--all-symbols --squeeze-adaptive --atr-stop-mult 2.0 --require-in-channel --max-channel-span-days 365 --max-entries-per-day 1 --friction-pct 0.25 --max-beyond-width 0.25 --fallback-provider IB --merge-mode prefix --start 2018-11-01 --end 2026-08-27`

Windowed 504/252. Raw **18215** after bounce-from-above fill (was 21087).

| entry | n | E% | PF | median% | WR% |
|-------|---|-----|-----|---------|-----|
| pivot + beyond 0.25 (prior) | 700 | +1.14 | 1.397 | -3.35 | — |
| l3_touch + beyond 0.25 (gap-through fills) | 1231 | +0.99 | 1.312 | -3.72 | 33.9 |
| **l3_touch + bounce tag** (2026-08-28 eve) | **1164** | **+1.32** | **1.441** | -3.44 | 35.4 |

l3_touch beyond-width A/B after bounce filter: 0.25 still the only useful cap (off n=1678 E +0.10 PF 1.03).

**WTFC 2019-07-16:** not the Aug 2019–Jan 2020 3L/2H on TV. The fill used a stale L1 2018-12-26 / L2 2019-03-25 / H2 2019-04-18 rail. Earnings 2019-07-16 gapped **entirely under** support ($67.14): H 66.49 / L 63.77 / C 65.07. That is a break, not a from-above wick tag. Fill now requires high ≥ support, close not broken, and cancels if a prior bar closed below support.

Nightly scanner stays on pivot-confirm until explicitly switched.

## Code

- `scripts/research/find_ascending_channels.py`: `find_h2_l3_setups` (no `last_end` greedy skip)
- `scripts/research/backtest_channel_touch_trades.py`: `--entry-mode l3_touch`, `_l3_rail_touch`
- Tests: `tests/unit/test_l3_touch_entry.py`

## Artifacts

- `reports/ascending_channels/channel_touch_trades_20260828_192557.csv`
- `reports/ascending_channels/channel_touch_trades_raw_20260828_192557.csv`
- `reports/ascending_channels/channel_touch_beyond_width_ab_20260828_192557.csv`
- `reports/ascending_channels/channel_touch_tv_report_interactive_rs_top1_default_fric0.25_l3_touch_beyond025_inchannel_span365_ib_fallback_20260828_192617.html`
