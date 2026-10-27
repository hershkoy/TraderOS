# Edge hunt - status log

Working notes for finding strategies that beat SPY buy-and-hold.

| Date | Doc |
|------|-----|
| 2026-10-26 | [Tachun long book — 20-year daily window and backtest todo](2026-10-26_tachun_long_book_todo.md) |
| 2026-10-24 | [QQQ 0-DTE 9:45 short put (claim replicates on mids)](2026-10-24_qqq_0dte_put_claimcheck.md) |
| 2026-09-25 | [Cup & Handle swing claim-check (no promote)](2026-09-25_cup_handle.md) |
| 2026-09-22 | [SPY 7d ROC rebound claim-check](2026-09-22_roc_rebound_spy.md) |
| 2026-09-22 | [ROC rebound 50-name ETF/large/small sample](2026-09-22_roc_rebound_universe.md) |
| 2026-09-22 | [ROC rebound ETF overlay grid (A1/A3 soft-interesting)](2026-09-22_roc_rebound_etf_grid.md) |
| 2026-09-22 | [ROC rebound expanding walk-forward (SOFT_PASS fragile years)](2026-09-22_roc_rebound_walkforward.md) |
| 2026-09-22 | [ROC A3 single-slot vs SPY (FAIL holdout gates)](2026-09-22_roc_rebound_occupancy.md) |
| 2026-09-22 | [ROC capital fixes (time-stop/Rf soft improve, no gate)](2026-09-22_roc_rebound_capital_fix.md) |
| 2026-09-19 | [Animesh/Dhan HA + 55 EMA + MACD swing](2026-09-19_ha_ema_macd_swing.md) |
| 2026-09-14 | [Causal VX-curve slope < 4 skip on last-15m](channel_touch/2026-09-14_last_15m_vx_curve_skip.md) |
| 2026-09-13 | [SMB LOD / 9 EMA x VWAP reclaim](2026-09-13_vwap_ema_reclaim.md) |
| 2026-09-13 | [VWAP reclaim 100x filters](2026-09-13_vwap_ema_reclaim_filters.md) |
| 2026-09-13 | [Choppy-regime RSI-2 MOC + 3-day pullback](2026-09-13_choppy_regime.md) |
| 2026-09-13 | [Daily Bollinger mean reversion in 2022](2026-09-13_mean_reversion_2022.md) |
| 2026-09-13 | [Causal VCP AVWAP breakout](2026-09-13_vcp_avwap.md) |
| 2026-09-13 | [Last-15m AND Deepvue take-4](channel_touch/2026-09-13_last_15m_deepvue_and.md) |
| 2026-09-12 | [Causal Deepvue weekend scan](2026-09-12_weekly_deepvue.md) |
| 2026-08-22 | [Phase 1 scorecard](2026-08-22_edge_hunt_phase1_scorecard.md) |
| 2026-08-22 | [Phase 2 dual momentum](2026-08-22_edge_hunt_phase2_scorecard.md) |
| 2026-08-22 | [Phase 3 SPY+BigVol blend](2026-08-22_edge_hunt_phase3_scorecard.md) |
| 2026-08-22 | [Phase 4 blend robustness](2026-08-22_edge_hunt_phase4_scorecard.md) |
| 2026-08-22 | [Phase 5 XS mom delta + swing MR](2026-08-22_edge_hunt_phase5_scorecard.md) |
| 2026-08-22 | [Phase 6 portfolio overlays](2026-08-22_edge_hunt_phase6_scorecard.md) |
| 2026-08-22 | [Phase 6b VolTarget+BigVol](2026-08-22_edge_hunt_phase6b_voltarget_bigvol.md) |
| 2026-08-22 | [Phase 6c VIX curve overlay](2026-08-22_edge_hunt_phase6c_vix_curve.md) |

Related: [current status](../current_status.md), [channel-touch](channel_touch/README.md), [Weekly BigVol](../weekly_bigvol/README.md), [daily digests](../daily/README.md), [realistic purchasing](../../features/realistic_purchasing.md)

## After Phase 6 (frozen 2026-08-22)

Phases 5 / 6 / 6c closed. Prefer freeze/productionize Phase 6b `Blend_VT60_BV40` over more Sharpe>1 fishing on the same window.

Active production track is **channel-touch** (daily H2 resist-break nightly + 15m `/hot` research loop). Same-bar rail clip is **unrealistic**, not optimistic. `current_best/` 1d slot is last-15m-open-mid + 15m N+1 mid **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01; 2022-23 fails; not a promote). Clip-exit hot-cross n=4079 E −0.19 is in `1d_unrealistic/`. Status: [channel-touch](channel_touch/README.md). Latest digest: [2026-09-05](../daily/2026-09-05.md).

## Harness

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\spy_benchmark_screen.py
python scripts\research\run_dual_momentum_phase2.py
python scripts\research\run_spy_bigvol_blend.py
python scripts\research\run_spy_bigvol_blend_robustness.py
python scripts\research\run_xs_mom_delta.py
python scripts\research\run_swing_mr_screen.py
python scripts\research\run_mean_reversion_2022.py
python scripts\research\run_choppy_regime.py
python scripts\research\run_ha_ema_macd_swing.py --weekly
python scripts\research\backtest_vwap_ema_reclaim.py --n-symbols 300
python scripts\research\overlay_vwap_ema_filters.py
python scripts\research\run_phase6_overlays.py
python scripts\research\run_voltarget_bigvol_blend.py
python scripts\research\run_vix_curve_overlay.py
python scripts\research\backtest_cup_handle.py --n-symbols 300 --skip-keeper
```

Outputs: `docs/status_log/edge_hunt/`, `reports/edge_hunt/`, `reports/edge_hunt_phase2/`, `reports/edge_hunt_phase3/`, `reports/edge_hunt_phase4/`, `reports/edge_hunt_phase5/`, `reports/edge_hunt_phase6/`, `reports/edge_hunt_phase6b/`, `reports/edge_hunt_phase6c/`.

## Verdict (2026-08-22)

Phase 4: **KEEP** `Blend_SPY70_BV30` (costs=ok, OOS=OOS_PASS (Sharpe>=SPY), stress20=ok).

Phase 5: **No promote.** PIT 12-1 + SMA200 failed; swing MR best Sharpe 0.28. Stock-signal families closed.

Phase 6: **No full-sample promote** (Sharpe>1.0 gate). Best near-miss: `VolTarget_12pct_20d_cap1` Sharpe 0.79 / MDD -15.9% vs SPY 0.63 / -34.1%. OOS that variant clears the gate vs OOS SPY (Sharpe 1.42, MDD -13.2%). CTA sleeve alone failed. Incumbent remains Phase 4 blend; vol-target is the strongest Phase 6 lead.

Phase 6b: **No full-sample promote.** Best: `Blend_VT60_BV40` Sharpe **0.93** / MDD **-13.0%** (vs SPY 0.63 / -34%). OOS clears hard (Sharpe 1.58). Closest yet to Sharpe>1.0 without clearing full-sample gate.

Phase 6c: **No full-sample promote.** IB VIX+VIX3M ingested; best `VIXCurve_soft_x_VT12` Sharpe **0.80** / MDD **-16.0%** — no lift vs plain vol-target. Binary curve alone ~Sharpe 0.73 / MDD -30%. Do not threshold-fish further.
