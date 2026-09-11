# Last-15m-open-mid: realistic sells (2026-09-11)

The last-RTH 15m mid **buy** is EOD-contemporaneous (open of 15:45 ET known at 15:45; mid known at 16:00 with the daily close). The first HTML still **kept daily occupancy sells**. That sell is the same-bar ATR clip.

**WVE 2023-12-06** (CTF from `…last_15m_open_mid_span365_20260911_132918.html`): buy last-15m mid **6.85**; sell **2023-12-07 @ 6.0254** (`hard_stop`). 6.0254 is the original signal-close book's 6% ATR ceiling from a cheaper ~6.41 fill, clipped to the next daily bar — not a price you could work after 16:00.

## Two delayed fills (occupancy not re-walked)

Same 2779 last-15m entries. Hard-stop is daily ATR k=2 clamp 1.5%–6% from the last-15m entry + 10% trail. Entry bar/session skipped.

| Book | Decision | Fill | n | E | PF | fric0.25 E | fric0.25 PF |
|------|----------|------|---|---|----|------------|-------------|
| Kept daily occupancy | Next daily low vs stop | Stop on that daily bar | 2779 | +0.96 | 1.29 | +0.71 | 1.21 |
| **15m-next-mid** | RTH 15m N low vs stop | N+1 15m mid (overnight OK) | 2779 | +0.35 | 1.11 | +0.10 | 1.03 |
| **daily-close-next-open-mid** | Session OHLC vs stop at 16:00 | Next 09:30 ET 15m mid | 2778 | +0.47 | 1.14 | +0.22 | 1.06 |

WVE: 15m N+1 mid **4.80 @ 09:45 ET** (−29.9%); EOD then next 09:30 mid **4.65** (−32.0%). One EOD skip (AUB 2026-06-24, no next 15m).

Year buckets after 0.25% friction (equal-weight gate; 2022-23 fails):

| Bucket | 15m-next-mid E / PF | EOD-next-open-mid E / PF |
|--------|---------------------|--------------------------|
| 2018-2019 | +1.12 / 1.43 | +0.84 / 1.30 |
| 2020-2021 | +0.13 / 1.04 | +0.49 / 1.15 |
| 2022-2023 | **−0.83 / 0.78** | **−0.54 / 0.86** |
| 2024-2026 | +0.31 / 1.09 | +0.26 / 1.08 |

Hard-stop share rises 981 → ~1370: 15m tracking tags the stop intra-day instead of waiting for the daily bar.

**No promote.** Does not beat hot-cross **n=4079 E −0.19 PF 0.94** on the live-executable gate, and 2022-23 is negative. Nightly stays the clip. Wall-clock ~16 min (IB 15m load 1013 symbols 2019–2026 from TimescaleDB, cache miss).

## HTML

- 15m N+1: `reports/ascending_channels/2026-09-11/channel_touch_tv_report_interactive_fric0.25_1d_h2_last_15m_open_mid_sell_15m_next_mid_span365_20260911_142523.html`
- EOD next-open mid: `reports/ascending_channels/2026-09-11/channel_touch_tv_report_interactive_fric0.25_1d_h2_last_15m_open_mid_sell_eod_next_open_mid_span365_20260911_142526.html`
- Compare CSV: `reports/ascending_channels/1d_unrealistic/last_15m_realistic_sells_compare.csv`

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\compare_1d_last_15m_realistic_sells.py --workers 8
```

Code: `utils/research/realistic_exits.py`. See [realistic purchasing](../../../features/realistic_purchasing.md).
