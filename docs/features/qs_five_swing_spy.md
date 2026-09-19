# QS Five-Swing SPY (live)

Strategy summary (rules, claim-check, MOC realism):
[`docs/strategies/qs_five_swing/README.md`](../strategies/qs_five_swing/README.md).

## Live on `/hot`

| Piece | Detail |
|-------|--------|
| Timeframe key | `qs` (Detector filter: QS) |
| Writer | `scripts/scanners/qs_five_swing_spy.py` |
| Cron | `qs_five_swing_spy` ~5m RTH; every minute 15:45–16:00 ET |
| Data | TimescaleDB SPY `1d` + optional `1m` developing H/L + Alpaca last |
| Resist column | Sleeve MOC close ceiling (priority S1→S5) |
| Support column | Prior day high (QS exit level) |
| Wait column | Sleeve ordinal 1–5 when armed |
| Status | `waiting` / `armed` / `filled` / `exit_signal` |

**Telegram / browser approaching:** from **15:45** ET when developing combined arms. Sticky row refreshes Alpaca last every minute; `/hot` **Last** shows `current / target` (green = buy-ok). **BUY NOW** from **15:57** only while last ≤ target. EXIT when last > prior high after Bought.

Same notify matrix pattern as DP (QS Approaching / Armed / Sell × TG|Br).
