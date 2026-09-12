# Shakeout then confirm only (skip first chase) — 2026-09-12

Safer 1d H2: arm on the first close above resist, **do not buy it**, fill only after ≥1 close at/below the rail then a second close above (support holds). Last-15m-open-mid + 15m N+1 mid is the live-clock grade. Overlay C’s n=13 delayed sleeve is not this test.

**No promote.** No nightly / `/hot` / `current_best` swap. Candle/vol occupancy-walk skipped (2022 still red).

## Diagnostic (same-list, occupancy-surviving extras)

Parent CSV: `reports/ascending_channels/2026-09-11/channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv` (last-15m N+1 **n=2297 E +0.27 PF 1.09**, geo 1.095).

| Sleeve | n | E | PF | geo year PF | 2022-23 E/PF |
|--------|---|---|----|-------------|--------------|
| All | 2297 | +0.27 | 1.086 | 1.095 | −0.90 / 0.73 |
| Parent first-break | 1552 | +0.12 | 1.037 | 1.024 | −0.82 / 0.75 |
| Extras | 745 | +0.58 | 1.185 | 1.038 | **−1.08 / 0.70** |

2018-19 extras n=19 (geo skips). Skip-parent winner $ 4991 vs loser $ saved 4812 (dollar_ratio 1.04, not blunt). Occupancy-surviving extras only — a lower bound on skip-first.

## Occupancy walk (clip unique, then last-15m)

`--shakeout-confirm-only` on `_h2_rail_tag_fills`: first close above arms; emit only `_shakeout_breakout_fill`. Unique span365, ATR k=2, friction 0.25. Then last-15m open-mid + 15m N+1 sells (occupancy not re-walked on the 15m overlays).

Smoke AMPL/VST/TARS (clip unique): AMPL 2024-02-09 skipped; VST 2021-02-18 skipped; TARS 2023-07-20 skipped. Later clip fills: VST 2022-04-18, TARS 2023-07-27, VST 2023-12-07.

| Book | n | E | PF | 2022-23 | 2022 |
|------|---|---|----|---------|------|
| Clip unique confirm-only | 1942 | +2.61 | 2.11 | **+2.26 / 1.90** | (clip — not the grade) |
| Last-15m + kept clip sells | 1191 | +0.70 | — | — | lookback sells |
| **Last-15m + 15m N+1 mid** | **1191** | **+0.36** | **1.115** | **−0.40 / 0.88** | **n=67 E −3.73 PF 0.11** |
| vs last-15m current_best | 2297 | +0.27 | 1.086 | −0.90 / 0.73 | n=117 E −2.37 PF 0.41 |

Fric0.25 on confirm-only N+1: E +0.11 PF 1.03. Geo year PF 1.22 (2018-19 n=51). Hard-stop 50%. Median still negative.

2023 last-15m confirm-only n=143 E +1.16 PF 1.41 (baseline 2023 E −0.33 PF 0.90). The pooled bump is 2023/2018-19, not 2022.

Live last-15m smoke: TARS 2023-07-27 mid 22.78 **−3.73%**; VST 2022-04-18 **−1.55%**; VST 2023-12-07 **+89%**. AMPL never prints.

Stop: 2022 itself is redder than the first-chase book. Do not stack morning-star / engulf / buy_pct occupancy-walks. Do not grade the clip unique E +2.61.

## Code

- `--shakeout-confirm-only` on `backtest_channel_touch_trades.py` / `backtest_channel_touch_h2_break.py`
- Overlay: `scripts/research/overlay_last_15m_shakeout_confirm.py`
- Candles: `utils/research/morning_doji_star.py` (entry helpers; not occupancy-walked)

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_shakeout_confirm.py --diag-extras
python scripts\research\backtest_channel_touch_h2_break.py --symbols AMPL,VST,TARS --shakeout-confirm-only --workers 1
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-confirm-only --workers 4 --load-workers 8
python scripts\research\compare_1d_last_15m_mid.py --before reports\ascending_channels\2026-09-12\channel_touch_full_h2_break_span365_unique_20260912_202409.csv --outdir reports\ascending_channels\2026-09-12 --out-name last_15m_confirm_only_compare.csv --workers 8
python scripts\research\export_last_15m_mid_trades.py --compare reports\ascending_channels\2026-09-12\last_15m_confirm_only_compare.csv --before reports\ascending_channels\2026-09-12\channel_touch_full_h2_break_span365_unique_20260912_202409.csv --outdir reports\ascending_channels\2026-09-12
python scripts\research\compare_1d_last_15m_realistic_sells.py --trades reports\ascending_channels\2026-09-12\channel_touch_h2_last_15m_open_mid_span365.csv --compare-in reports\ascending_channels\2026-09-12\last_15m_confirm_only_compare.csv --outdir reports\ascending_channels\2026-09-12 --mode 15m-next-mid --workers 8
```

Wall-clock: unique scan **116s** (daily cache hit) + last-15m 15m load **641s** + N+1 sells load **517s**.

Artifacts: `reports/ascending_channels/2026-09-12/channel_touch_full_h2_break_span365_unique_20260912_202409.csv`, `last_15m_confirm_only_compare.csv`, `channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv`, `last_15m_shakeout_confirm_diag.csv`.
