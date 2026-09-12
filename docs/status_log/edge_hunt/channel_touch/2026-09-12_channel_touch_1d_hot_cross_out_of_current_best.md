# 1d hot-cross clip-exits out of `current_best/` — 2026-09-12

Buy-now lerp85 is a live **entry**, but `current_best/1d_hot_cross.html` still **kept daily occupancy sells** (same-bar ATR clip). Honest 15m N+1 mid on that list is **n=4079 E −0.83 PF 0.77**. That is not the 1d number to beat.

Moved `1d_hot_cross.html` to `reports/ascending_channels/1d_unrealistic/`. Clip nightly HTML (`1d_channel_touch.html` / `1d_h2_resist_break.html`) was already there.

## What replaced the 1d slot

Best **live-executable** 1d book we have with a realistic **buy and** a realistic **sell** (occupancy not re-walked): last RTH 15m mid if that bar opened above the daily rail, then ATR k=2 + 10% trail on RTH 15m, fill N+1 mid.

| Book | n | E gross | PF | E net0.25 | PF net | Why it wins the slot |
|------|---|---------|----|-----------|--------|----------------------|
| **Last-15m + 15m N+1 mid (LAUR-fixed)** | **2297** | **+0.27** | **1.09** | **+0.02** | **1.01** | EOD-contemporaneous buy; 15m sells. Best 1d E/PF among honest books. |
| 1d H2 close-confirm + form 0.25 | 1947 | +0.24 | 1.07 | — | — | Causal 15m close-confirm **buy**; occupancy **sells** still daily clip. Not eligible. |
| Hot-cross lerp85 + 15m N+1 mid | 4079 | −0.58 | 0.83 | −0.83 | 0.77 | `/hot` analog. Losing. |
| 15m L3 wait-12 signal-close | 1744 | +0.13 | 1.22 | — | — | Stays the **15m** keeper (already in `current_best/`). Not a 1d book. |

Copied (not hard-linked) to:

- `reports/ascending_channels/current_best/1d_last_15m.html`
- `reports/ascending_channels/current_best/1d_channel_touch.html` (same bytes)

Stamp: `...interactive_fric0.25_1d_h2_last_15m_open_mid_sell_15m_next_mid_span365_20260911_165800.html`. Stable trades: `reports/ascending_channels/channel_touch_1d_last_15m.csv`.

2022-23 still fails on last-15m H2 (2022 n=117 E −2.37 PF 0.41). **Not a promote.** Nightly stays the clip. `/hot` stays buy-now last-cross, not last-15m.

See [realistic sells](2026-09-12_channel_touch_hot_cross_realistic_sells.md), [LAUR](2026-09-11_laur_false_shakeout.md), [realistic purchasing](../../../features/realistic_purchasing.md).
