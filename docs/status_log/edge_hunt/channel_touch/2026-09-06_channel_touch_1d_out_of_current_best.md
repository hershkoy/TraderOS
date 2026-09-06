# 1d HTML out of `current_best/` — 2026-09-06

Daily H2/L3 reports rely on an **unrealistic** entry: after an EOD close-above-resist, the fill is rail + slip clipped to that same daily bar. You cannot work that rail/wick live. Nightly still uses this path.

Moved out of `reports/ascending_channels/current_best/` into `reports/ascending_channels/1d_unrealistic/`:

- `1d_channel_touch.html` / `1d_h2_resist_break.html`
- `1d_l3_touch.html`
- `1d_keeper_plus_h2_resist_break.html`

Clip / next-mid overlay HTML stays here. 15m L3 wait-12 signal-close remains the 15m keeper. 1d `--realistic-fill` (next-mid / open-cross / next-open) stays a research overlay — next-mid n=1239 is still pre-`_as_session_date` and still keys off the rail clip.

**Update (same evening):** the live-executable 1d book (hot-cross unique **n=4079 E −0.19 PF 0.94**) was copied into `current_best/` as the number to beat even though it loses. Clip HTML stays here. See [hot-cross in current_best](2026-09-06_channel_touch_current_best_hot_cross.md).

Write-up: [realistic purchasing](../../../features/realistic_purchasing.md).
