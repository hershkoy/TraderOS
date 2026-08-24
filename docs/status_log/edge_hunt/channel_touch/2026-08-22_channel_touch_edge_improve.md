# Channel-touch edge improvement (2026-08-22)

## Setup
- Script: `scripts/research/backtest_channel_touch_trades.py --all-symbols --edge-improve`
- Universe: ALPACA `1d`, 2,217 symbols (SPY loaded for RS)
- Rules: classical channel bottom touch ≥3, 3% hard stop, 10% trail
- Features: `adv_20`, `atr_pct` (ATR14/close), `rs_spy_63d` / `rs_spy_126d`
- Outputs:
  - `reports/ascending_channels/channel_touch_trades_20260822_231105.csv`
  - `reports/ascending_channels/channel_touch_edge_scenarios_20260822_231105.csv`

## Wall-clock (warm cache)
- Symbol list ~12s, load ~13s, scan+trade ~1.2s, RS enrich ~0.9s → ~30s total

## Verdict
| Hypothesis | Result |
|---|---|
| ADV floor lifts winner magnitude / expectancy | **Rejected** on this sample — expectancy fell (1.48 → 1.19 at $20M) |
| Higher ATR% filters sluggish names | **Rejected** — expectancy fell; avg win did not rise enough |
| Same-day RS vs SPY prioritization | **Supported** — `rs_top1_only` best: E=+1.58% gross, PF 1.76 |
| Net E stays ≥ +1.0% after 0.10–0.25% friction | **Pass for baseline and RS-top1**; fail for hard ADV/ATR stacks |

## Recommended next cut
Prefer **same-day RS top-1 vs SPY (126d)** without hard ADV/ATR floors. Soft ADV floors for operational tradability are still reasonable, but they are not an edge lift here.

## Friction
- Baseline net: +1.38% @ 0.10%, +1.23% @ 0.25%
- RS top1 net: +1.48% @ 0.10%, +1.33% @ 0.25%
