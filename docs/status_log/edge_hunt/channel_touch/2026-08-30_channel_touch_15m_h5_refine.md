# 15m H5 stack — lift WR and PF — 2026-08-30

H5-only (unique-symbol + prior-bar `volume_rel_20 >= 1`) is still ~39k trades
in 7 years (WR 37% PF 1.63). Stack leak-safe, train-year-only cutoffs on top.
**Research only. Not nightly.**

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\refine_15m_h5_filters.py
```

## Expanding-year OOS (train years < Y)

| Book | n | WR | E | PF | Drop-top-1% PF |
|------|---|----|---|-----|----------------|
| H5 only | 39442 | 37.0 | +0.34 | 1.63 | 1.41 |
| H5 + overshoot train p50 | 20024 | 39.4 | +0.51 | 2.18 | 1.90 |
| H5 then logistic | 20474 | 41.1 | +0.48 | 1.90 | 1.66 |
| H5 + overshoot train p80 | 8631 | 45.8 | +0.79 | 3.06 | 2.71 |
| **H5 + overshoot p80 + vol>=2** | **6306** | **48.8** | **+0.93** | **3.61** | **3.22** |

Overshoot is `channel_pos - 1` at the prior completed bar. Volume is prior-bar
`volume_rel_20`. No fill-bar close, no RS rank, no full-day name count.

## Year buckets (tight stack)

H5 + overshoot >= train p80 + vol >= 2:

| Years | n | WR | E | PF |
|-------|---|----|---|-----|
| 2018-2019 | 687 | 39.6 | +0.56 | 2.45 |
| 2020-2021 | 1709 | 52.8 | +1.12 | 4.32 |
| 2022-2023 | 1839 | 48.1 | +0.80 | 3.18 |
| 2024-2026 | 2071 | 49.1 | +1.02 | 3.86 |
| FULL OOS | 6306 | 48.8 | +0.93 | 3.61 |

Drop-top-5 winners: PF 3.56. Drop-top-1%: PF 3.22 WR 48.3. Not a one-trade tail.

~2.5 fills per RTH day vs ~15 on H5-only vs 1 on the old RS top1 cap.

## Did not help (on H5)

Stricter volume **alone** (p60 or >=2) barely moves WR. Narrower width **cuts**
WR. Fast wait is a small lift (WR 38.6 PF 1.77). Logistic on H5 loses to a
simple overshoot cap.

## Gate

If the goal is **higher WR and PF** and 39k is too many: use **H5 + overshoot
>= train p80 + volume_rel >= 2** (n=6306). If you still want ~20k trades: H5
+ overshoot p50 (WR 39 PF 2.18).

Do **not** wire 15m into nightly. Stress is drop-top-N only, not a full
bootstrap/max-open pass.

## Artifacts

- `scripts/research/refine_15m_h5_filters.py`
- `reports/ascending_channels/channel_touch_15m_h5_refine_20260830_015525.csv`
- `reports/ascending_channels/channel_touch_15m_h5_refine_stress_20260830_015525.csv`
