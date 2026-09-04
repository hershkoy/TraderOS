# 15m L3 first-tag (min-wait 1) vs wait-12, realistic fill

Textbook Edwards/Magee L3 is the first from-above support tag after H2 (`--min-l3-wait-bars 1`). Wait-12 was an empirical overlay. This rescan asks whether dropping it helps on the honest 15m purchaser.

Same stack as frozen `current_best/15m_channel_touch.html` except min-wait: `--preset 15m --n-symbols 300 --entry-mode l3_touch --min-l3-wait-bars 1 --realistic-fill --workers 4 --load-workers 8`. Causal H2, in-channel, span≤10d, RS top1, friction 0.10, 2018-11-01 → 2025-12-02. Do **not** copy onto `current_best/15m_channel_touch.html`.

## Wall-clock

Cache hit 300/300 IB 15m (1.1s). Scan+trade **241s** (raw 114211). RS 61s. Elapsed **442s**.

## Results (RS top1, 0.10% friction)

| Book | n | WR% | median% | E% | PF | 2018-19 |
|------|---|-----|---------|-----|-----|---------|
| **Wait-12 realistic (frozen)** | **1746** | — | — | **+0.09** | **1.16** | all + on optimistic; kept |
| **Wait-1 realistic (this run)** | **1770** | 30.3 | -0.77 | **+0.02** | **1.03** | E −0.13 PF 0.80 |
| Wait-1, tags in bars 1–11 | 1298 | 29.2 | -0.82 | **−0.04** | **0.95** | — |
| Wait-1, tags at bar ≥12 | 472 | 33.5 | -0.66 | +0.17 | 1.29 | occupancy leftover, **not** the wait-12 book |

Year net E wait-1: 2018-19 **−0.13**, 2020-21 +0.06, 2022-23 +0.07, 2024-26 +0.02. Median `wait_bars` **6** (p10=1, p75=12). Median `channel_pos` 0.089; 177 trades with pos>0.35 (wait-12 Loop 5 had 0).

The late-only slice of wait-1 is **not** wait-12: early fills take RS/occupancy, so n=472 leftovers. Wait-12 **aborts** those setups and lets a later tag (or a later channel) compete.

Beyond-width A/B on this wait-1 raw (still worse than wait-12): off n=1770 E +0.02 PF 1.03; 0.25 n=1734 E +0.07 PF 1.11. Do not promote copying daily 0.25 onto 15m from this one table.

## Verdict

**Keep wait-12.** First-tag is closer to the textbook rail-buy, but on 15m + next-bar mid the extra 1–11 bar dips lose money (PF 0.95) and 2018-19 fails. Wait-12 stays the only 15m L3 book that is net-positive after realistic fill.

HTML: `reports/ascending_channels/channel_touch_tv_report_interactive_rs_top1_default_fric0.10_15m_l3_wait1_realistic_20260904_114015.html` (quality-filtered embed 61163; UI default RS top1).

Trades: `reports/ascending_channels/channel_touch_15m_trades_20260904_113653.csv` / raw `_raw_`.
