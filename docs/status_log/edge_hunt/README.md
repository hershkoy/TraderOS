# Edge hunt - status log

Working notes for finding strategies that beat SPY buy-and-hold.

| Date | Doc |
|------|-----|
| 2026-08-22 | [Phase 1 scorecard](2026-08-22_edge_hunt_phase1_scorecard.md) |
| 2026-08-22 | [Phase 2 dual momentum](2026-08-22_edge_hunt_phase2_scorecard.md) |
| 2026-08-22 | [Phase 3 SPY+BigVol blend](2026-08-22_edge_hunt_phase3_scorecard.md) |

Related: [stock market data coverage](../2026-08-21_stock_market_data.md), [Weekly BigVol](../weekly_bigvol/README.md)

## Harness

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\spy_benchmark_screen.py
python scripts\research\run_dual_momentum_phase2.py
python scripts\research\run_spy_bigvol_blend.py
```

Outputs: `docs/status_log/edge_hunt/`, `reports/edge_hunt/`, `reports/edge_hunt_phase2/`, `reports/edge_hunt_phase3/`.

## Verdict (2026-08-22)

Phase 3 **promoted** `Blend_SPY70_BV30` (CAGR 12.32%, Sharpe 0.79, MDD -25.3% vs SPY 13.64% / 0.68 / -34.1%). 80/20 and 60/40 also PASS; pure BigVol sleeve alone still fails CAGR.
