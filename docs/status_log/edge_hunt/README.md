# Edge hunt - status log

Working notes for finding strategies that beat SPY buy-and-hold.

| Date | Doc |
|------|-----|
| 2026-08-22 | [Phase 1 scorecard](2026-08-22_edge_hunt_phase1_scorecard.md) |
| 2026-08-22 | [Phase 2 dual momentum](2026-08-22_edge_hunt_phase2_scorecard.md) |
| 2026-08-22 | [Phase 3 SPY+BigVol blend](2026-08-22_edge_hunt_phase3_scorecard.md) |
| 2026-08-22 | [Phase 4 blend robustness](2026-08-22_edge_hunt_phase4_scorecard.md) |
| 2026-08-22 | [Phase 5 XS mom delta + swing MR](2026-08-22_edge_hunt_phase5_scorecard.md) |
| 2026-08-22 | [Phase 6 portfolio overlays](2026-08-22_edge_hunt_phase6_scorecard.md) |

Related: [stock market data coverage](../2026-08-21_stock_market_data.md), [Weekly BigVol](../weekly_bigvol/README.md)

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
python scripts\research\run_phase6_overlays.py
```

Outputs: `docs/status_log/edge_hunt/`, `reports/edge_hunt/`, `reports/edge_hunt_phase2/`, `reports/edge_hunt_phase3/`, `reports/edge_hunt_phase4/`, `reports/edge_hunt_phase5/`, `reports/edge_hunt_phase6/`.

## Verdict (2026-08-22)

Phase 4: **KEEP** `Blend_SPY70_BV30` (costs=ok, OOS=OOS_PASS (Sharpe>=SPY), stress20=ok).

Phase 5: **No promote.** PIT 12-1 + SMA200 failed; swing MR best Sharpe 0.28. Stock-signal families closed.

Phase 6: **No full-sample promote** (Sharpe>1.0 gate). Best near-miss: `VolTarget_12pct_20d_cap1` Sharpe 0.79 / MDD -15.9% vs SPY 0.63 / -34.1%. OOS that variant clears the gate vs OOS SPY (Sharpe 1.42, MDD -13.2%). CTA sleeve alone failed; VIX overlay skipped (no VIX data). Incumbent remains Phase 4 blend; vol-target is the strongest Phase 6 lead.
