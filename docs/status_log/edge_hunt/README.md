# Edge hunt - status log

Working notes for finding strategies that beat SPY buy-and-hold.

| Date | Doc |
|------|-----|
| 2026-08-22 | [Phase 1 scorecard](2026-08-22_edge_hunt_phase1_scorecard.md) |
| 2026-08-22 | [Phase 2 dual momentum](2026-08-22_edge_hunt_phase2_scorecard.md) |

Related: [stock market data coverage](../2026-08-21_stock_market_data.md), [Weekly BigVol](../weekly_bigvol/README.md)

## Harness

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\spy_benchmark_screen.py
python scripts\research\run_dual_momentum_phase2.py
```

Outputs: `docs/status_log/edge_hunt/`, `reports/edge_hunt/`, `reports/edge_hunt_phase2/`.

## Verdict (2026-08-22)

No Phase 1/2 candidate cleared gates vs SPY B&H. Best near-miss: Weekly BigVol portfolio (similar Sharpe, much lower MDD, lower CAGR).
