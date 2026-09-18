# Deep Pullback SPY (live)

Strategy summary (rules, claimed stats, rebuild, links):
[`docs/strategies/deep_pullback/README.md`](../strategies/deep_pullback/README.md).

Source video: [I Backtested a Simple Pullback Strategy for 30 Years](https://www.youtube.com/watch?v=eV8YhMUpk3Q).

Mean-reversion sleeve from Quantified Strategies: buy SPY at the close after an
unusually deep short-term pullback while still above the 200-day MA; exit on the
QS rule (close > prior day's high).

Research HTML: `reports/research/deep_pullback/deep_pullback_spy.html`
(includes futures-style 1x/2x/3x leverage tab).

## Live on `/hot`

| Piece | Detail |
|-------|--------|
| Timeframe key | `dp` (Detector filter: DP) |
| Writer | `scripts/scanners/deep_pullback_spy.py` |
| Cron | `deep_pullback_spy` ~5m RTH; every minute 15:40–16:00 ET (`crons/crontab.yaml`) |
| Data | TimescaleDB SPY `1d` + Alpaca last as developing close |
| Resist column | MOC target `moc_max` = min(15d-low cap, worst-1d ret cap) |
| Support column | SMA200 floor |
| Wait column | score 0–3 |
| Status | `waiting` = approaching; `armed` = full 3 conditions (buy MOC); `filled` = Bought / in trade; `exit_signal` = QS |

**Approaching (early arm):** above SMA200 and at least two of {near 15d low within 0.5%, sharp day in worst-3 of 10, exact cond1, exact cond2}. Shown on `/hot` anytime during RTH when score warrants it.

**Telegram approaching:** only in the last **20 minutes** before cash close (15:40–16:00 ET). Message includes the MOC target: if SPY finishes at or below `moc_max` (and still above SMA200), send MOC buy. ARMED / EXIT (and browser alerts) are gated per detector on `/hot` notify matrix (DP Armed / Approaching / Sell × TG|Br). Mark **Bought** after you fill so the scanner tracks QS exit.

Fill at the **close auction / last minutes** — same as the research book. Prefer MES/ES for leverage; options are a poor fit for ~0.8% average moves.
