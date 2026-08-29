# Channel-touch confidence sizing (ridge on P&L) — 2026-08-29

## Goal

Size entries from a **ridge regression** that predicts `gain_pct_net` from the point-in-time entry snapshot. Not a new filter. Expanding walk-forward with **purge + 21d embargo**. Train on quality-filtered fills; score two OOS views: all quality-filtered, and RS top1 (live-like).

Detector v1 unchanged. Nightly unchanged.

## Setup

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_confidence_size.py --stack 1d
python scripts\research\backtest_channel_touch_confidence_size.py --stack 15m
```

- Ridge (numpy closed form, L2=1), train y winsorized 1/99 and |y|<=15; eval on uncapped P&L
- Features z-scored on the train fold only; size map from train pred 10th–90th pct to 0.25x–2.0x
- **1d:** `NO_SAMEBAR_CLOSE_FEATURES` (fill-bar RSI/close/%B/range/SMA dist dropped). Quality: in-channel, span365, beyond 0.25, wait>=6, RSI<=50, friction 0.25%
- **15m:** lagged `MODEL_FEATURES` (prior-bar CSV). Quality: in-channel, span 10d, wait>=12, **no** beyond-0.25, friction 0.10%
- First OOS fold starts after **60%** of buys (by date); then calendar years. Skip fold if n_train<80 or n_test<30
- Wall-clock **~4s** (CSV only; no rescan)

## 1d (live stack)

Quality-filtered n=2627 from `channel_touch_trades_raw_20260828_194314.csv`. Density is late: first WF fold starts **2024-03-25**. Stitched OOS n=1052 (RS-top1 n=397).

| view | variant | n | E | PF | MDD | mean size | spearman |
|------|---------|---|----|----|-----|-----------|----------|
| quality | equal | 1052 | +2.21 | 1.80 | 83.5 | 1.00 | — |
| quality | **scale-all** | 1052 | **+2.06** | 1.80 | 89.1 | 0.93 | **-0.047** |
| quality | skip-neg | 318 active | +1.20 | 1.90 | 84.6 | 0.49 | — |
| RS top1 | equal | 397 | +2.46 | 1.90 | 105.7 | 1.00 | — |
| RS top1 | scale-all | 397 | +2.29 | 1.93 | 110.8 | 0.91 | -0.038 |

Folds mixed: 2024 scale E ~tied, **2025 scale E worse** (+1.48 vs +2.26), 2026 scale E better. Drop-top-3 still PF>1 but equal-dollar already is. Pred quintiles are not monotone (middle bucket worst).

**Verdict: no promote.** Scale-all does not beat equal-dollar on expectancy. Rank correlation is negative. Skip-neg is the old hard gate with worse E.

## 15m (research stack)

Quality-filtered n=21646 from `channel_touch_15m_trades_raw_20260829_105511.csv`. First WF fold **2023-04-05**. Stitched OOS n=8657 (RS-top1 n=662).

| view | variant | n | E | PF | MDD | mean size | spearman |
|------|---------|---|----|----|-----|-----------|----------|
| quality | equal | 8657 | +0.181 | 1.34 | 73.2 | 1.00 | — |
| quality | scale-all | 8657 | +0.219 | 1.41 | 73.1 | 1.05 | **+0.091** |
| quality | skip-neg | 7783 active | +0.216 | 1.41 | 71.0 | 1.02 | — |
| quality | scale mean-norm | 8657 | +0.208 | 1.41 | 70.5 | 1.00 | — |
| RS top1 | equal | 662 | **+0.463** | **1.87** | **26.2** | 1.00 | — |
| RS top1 | scale-all | 662 | +0.436 | 1.79 | 41.2 | 1.04 | +0.017 |

Unranked quality-filtered: scale-all beats equal on E and PF in **all three** WF folds; drop-top-3 PF 1.38; mean-norm still a small lift (not just 1.05x leverage). **RS top1 (the 15m keeper) reverses the lift** and **worsens MDD** (41 vs 26). Skip-neg does not beat scale-all.

**Verdict: no promote** for live-like sizing. The unranked lift is not usable after same-day RS, which is how the stack is traded.

## Gate

Promote-to-research (not nightly) required stitched OOS scale-all to beat equal on **PF and E**, more than one fold, and drop-top-3 PF>1. Skip-neg had to beat both equal and scale-all.

Neither stack clears that on **RS top1**. Do **not** wire into `scripts/scanners/channel_touch_nightly.py`.

## Code

- [`utils/research/channel_touch_entry_model.py`](../../../utils/research/channel_touch_entry_model.py): `RidgePnlScorer`, purge/embargo, expanding WF, train-only size map
- [`scripts/research/backtest_channel_touch_confidence_size.py`](../../../scripts/research/backtest_channel_touch_confidence_size.py)
- Tests: `tests/unit/test_channel_touch_entry_model.py`

## Artifacts

- `reports/ascending_channels/channel_touch_confidence_size_1d_20260829_115859_*.csv`
- `reports/ascending_channels/channel_touch_confidence_size_15m_20260829_115900_*.csv`
