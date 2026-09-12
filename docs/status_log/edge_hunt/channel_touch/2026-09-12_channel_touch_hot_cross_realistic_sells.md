# current_best 1d hot-cross: realistic sells (2026-09-12)

`current_best/1d_hot_cross.html` is buy-now lerp85 (live-executable **entry**) but still **kept daily occupancy sells**. That sell is the same-bar ATR clip: occupancy walks daily bars, skips the fill day (`skip_entry_bar_stop`), then fills the stop on the next daily bar whose low tagged it.

No prior overlay existed. This is the last-15m realistic-sell walk on the published unique span365 book (`reports/ascending_channels/channel_touch_1d_hot_cross.csv`). Occupancy **not** re-walked. Hard-stop is daily ATR k=2 clamp 1.5%–6% + 10% trail.

**HCC 2019-05-02** (15:30 ET lerp85 **33.4326**): kept sell next day **31.4267** (−6.00%, the 6% ceiling). 15m N+1 mid **28.805 @ 09:45 ET** (−13.84%). EOD then next 09:30 mid **28.155 @ 2019-05-06** (−15.79%).

Unlike last-15m (buy at 15:45 ET), hot-cross can fill mid-session. Daily occupancy never same-day stops (kept `hold_days==0` is 0). 15m-next-mid does: **261** calendar-same-day sells.

## Results (occupancy not re-walked)

| Book | Decision | Fill | n | E gross | PF gross | E net0.25 | PF net0.25 | geo year PF (gross / net) |
|------|----------|------|---|---------|----------|-----------|------------|---------------------------|
| Kept daily occupancy (`current_best`) | Next daily low vs stop | Stop on that daily bar | 4079 | +0.06 | 1.02 | **−0.19** | **0.94** | 0.96 / (README net ~0.89) |
| **15m-next-mid** | RTH 15m N low vs stop | N+1 15m mid (overnight OK) | 4079 | −0.58 | 0.83 | **−0.83** | **0.77** | 0.80 / 0.74 |
| **daily-close-next-open-mid** | Session OHLC vs stop at 16:00 | Next 09:30 ET 15m mid | 4078 | −0.40 | 0.89 | **−0.65** | **0.83** | 0.85 / 0.79 |

One EOD skip: VRTX 2026-08-19 (`no_next_bar`; kept sold 2026-08-25).

Year buckets after 0.25% friction (equal-weight gate; all books fail 2022-23):

| Bucket | Kept (README net) | 15m-next-mid E / PF | EOD-next-open-mid E / PF |
|--------|-------------------|---------------------|--------------------------|
| 2018-2019 | −0.55 / 0.82 | **−0.93 / 0.72** | **−1.02 / 0.71** |
| 2020-2021 | +0.02 / 1.01 | **−0.31 / 0.91** | +0.06 / 1.02 |
| 2022-2023 | **−0.90 / 0.75** | **−1.53 / 0.60** | **−1.36 / 0.66** |
| 2024-2026 | +0.04 / 1.01 | **−0.83 / 0.78** | **−0.72 / 0.81** |

Winner $ (equal 1u, gross): kept +13377 / −13142; 15m +11771 / −14133; EOD +12959 / −14583. Realistic fills clip winners **and** deepen losers. 15m vs kept: 2088 trades worse, 1991 better, mean Δ −0.64 pts.

Hard-stop share stays ~60% (kept 2430; 15m 2438; EOD 2426). Same-day 15m stops are the extra hurt vs last-15m (that book bought 15:45 so the next bar was already tomorrow).

**No promote.** Honest 15m exits make the already-losing hot-cross book worse, not better. Clip-exit HTML moved to `1d_unrealistic/1d_hot_cross.html` on 2026-09-12. The 1d number to beat is last-15m + 15m N+1 mid **n=2297 E +0.27 PF 1.09**. `/hot` Bought already tracks last vs ATR k=2 + 10% trail.

Wall-clock **1146s** (~19 min; IB 15m TimescaleDB load 1175/1175 in 576s after a Docker cold start, cache miss).

## HTML

- 15m N+1: `reports/ascending_channels/2026-09-12/channel_touch_tv_report_interactive_fric0.25_1d_hot_cross_sell_15m_next_mid_span365_20260912_100737.html`
- EOD next-open mid: `reports/ascending_channels/2026-09-12/channel_touch_tv_report_interactive_fric0.25_1d_hot_cross_sell_eod_next_open_mid_span365_20260912_100742.html`
- Compare CSV: `reports/ascending_channels/2026-09-12/hot_cross_realistic_sells_compare.csv`

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\compare_1d_hot_cross_realistic_sells.py --workers 8
```

Code: `utils/research/realistic_exits.py`, `scripts/research/compare_1d_hot_cross_realistic_sells.py`. See [realistic purchasing](../../../features/realistic_purchasing.md).
