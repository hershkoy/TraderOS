# Causal VX2/VX1 slope < 4 skip on last-15m (2026-09-14)

Same-list skip on the LAUR-fixed last-15m-open-mid + 15m N+1 mid book **n=2297 E +0.27 PF 1.09** (geo 1.095). Prior-session `completed_asof` only (fill-day CFE close is illegal at 15:45 ET). Occupancy not re-walked. **No promote.**

Pine rule: `((VX2 / VX1) - 1) * 100`; skip if < 4. **Ratio is computed in Python** from separate TradingView daily dumps (`CBOE_DLY:VX1!` n=2463 2016-11-27→2026-09-13; `CBOE_DLY:VX2!` n=3439 2013-01-13→2026-09-13). Chart paged via `chart_set_visible_range` + `dump_tv_bars.mjs`. Not a Pine plot, not cash VIX/VIX3M.

Runner: `scripts/research/overlay_last_15m_vx_curve.py --vx1 …vx1_daily.json --vx2 …vx2_daily.json`.

## Results (true futures)

| Overlay | n | E | PF | geo | skipped | 2022 n / E / PF | 2023 E / PF | 2022-23 E / PF |
|---------|---|---|----|-----|---------|-----------------|-------------|----------------|
| Baseline | 2297 | +0.27 | 1.09 | 1.095 | — | 117 / **−2.37 / 0.41** | −0.33 / 0.90 | −0.90 / 0.73 |
| **VX12_lt4** | 1645 | **−0.00** | **1.00** | **1.00** | 652 | 62 / **−2.45 / 0.39** | **−0.58 / 0.82** | −0.93 / 0.72 |

VX12 asof on fills: median **6.40%**, 28.4% below 4. Pine bands: >=8 n=769, 5–8 n=701, 0–5 n=599, <0 n=228. **2022 median 4.78**, 47% skip; leftover 2022 is worse. 2023 leftover also worse (−0.58 vs −0.33).

## Winner-$ (2020-21 / 2024-26)

| Overlay | 2020-21 win$ / lose$ | blunt | 2024-26 win$ / lose$ | blunt |
|---------|----------------------|-------|----------------------|-------|
| VX12_lt4 | 779 / −453 | **yes** | 1546 / −1273 | no |

2020-21 kept **E −0.02 PF 0.99** vs baseline +0.46 / 1.15. Year-bucket geo falls 1.095 → 1.00.

## Why

Same failure as cash VIX backwardation / SPY SMA / TLT. The 4% contango gate fires on high-vol trend legs and stays on through the 2022 bear rallies that still close above the daily rail. Skipping 47% of 2022 does not make the leftover 53% green; it clips 2020-21 winners.

A first cash analog (linear 60d VIX/VIX3M) had already failed (VX60_lt4 n=1714 E +0.06). The real VX2/VX1 book is **worse** (flat expectancy). Do not threshold-fish the Pine 5%/8% hlines.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_vx_curve.py --vx1 reports\ascending_channels\2026-09-14\tv_cboe_vx1_daily.json --vx2 reports\ascending_channels\2026-09-14\tv_cboe_vx2_daily.json
```

Artifacts: `reports/ascending_channels/2026-09-14/last_15m_vx_curve_skip_summary.csv`, `tv_cboe_vx1_daily.json`, `tv_cboe_vx2_daily.json`.
