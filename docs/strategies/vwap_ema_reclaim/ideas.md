Implemented 2026-09-13 as causal research (not a promote): `utils/research/vwap_ema_reclaim.py`, runner `scripts/research/backtest_vwap_ema_reclaim.py`, overlay `scripts/research/overlay_vwap_ema_filters.py`. Write-up: [VWAP EMA reclaim](../../status_log/edge_hunt/2026-09-13_vwap_ema_reclaim.md), [filters](../../status_log/edge_hunt/2026-09-13_vwap_ema_reclaim_filters.md).

SMB Capital short "the trade I wait for every day":

1. Stock turns off the low of the day (sells off, LOD in, stops going down).
2. Momentum rebuilds toward session VWAP.
3. Trigger: 9 EMA crosses VWAP.
4. Enter at the cross (do not wait extra bars). Research fill is 15m next-mid after the completed cross; intra-bar line-cross needs tick/5m.
5. Measured move: unit = LOD to the cross. Target = +1 unit. Stop = 1/3 of the way back toward the LOD (~3:1 on paper).
