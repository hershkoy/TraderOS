# Detect-once stop / ATR exit sweep (2026-09-06)

Tooling, not a promote. `--stop-pct-sweep` and `--atr-stop-mult-sweep` load OHLCV once, detect H2/L3 setups and fills once per symbol, then occupancy-walk `_simulate_trade` for each stop (or ATR k).

A tighter stop can free the symbol earlier (`busy_until`) and take a later fill the looser stop never sees, so this is **not** one trade list with extra P&L columns.

`--stop-pct-sweep` uses fractions (`0.02,0.03,0.04` = 2/3/4%) and **disables ATR** for those variants (`stop_pct` is unused when `atr_stop_mult` is set). `--atr-stop-mult-sweep 1,1.5,2` keeps CLI `--stop-pct` as the ATR-off fallback. Mutually exclusive with each other and with `--edge-v2` / `--peak-trail-sweep`. Other CLI filters (in-channel, span, unique/day, friction) still apply.

Outputs: `reports/ascending_channels/YYYY-MM-DD/channel_touch_{stop_pct|atr_k}_sweep_*.csv` plus years + summary txt.

This is not a replacement for the ATR k=2 keeper. Do not promote a fixed 2% stop from a sweep without E/PF/payoff/MDD vs that keeper.
