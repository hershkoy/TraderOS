# Deep Pullback Strategy (SPY)

Short-term mean reversion on **SPY**: buy unusually deep pullbacks while the long-term trend stays bullish, then exit quickly on a rebound.

**Source (for reference):** [I Backtested a Simple Pullback Strategy for 30 Years](https://www.youtube.com/watch?v=eV8YhMUpk3Q) — Quantified Strategies (Håkan Samuelsson / Oddmund Groette).

Related article: [Deep Pullback Strategy (Only 14% Drawdown)](https://www.quantifiedstrategies.com/deep-pullback-strategy-only-14-drawdown/).

---

## Idea

Combine **short-term weakness** with a **long-term bull filter**:

- Want a close that is rare on a ~3-week window and a sharp 1-day drop.
- Still require price above the 200-day moving average (stay out of major bear regimes).
- Do not hold for a new trend — clip the bounce with a fast exit.

---

## Rules

### Entry (buy at the close)

All three must be true on the same session:

1. Today’s close is the **lowest close of the last 15** trading days.
2. Today’s 1-day % return is the **lowest 1-day return of the last 10** trading days.
3. Close is still **above the 200-day SMA**.

### Exit (QS exit)

Sell at the close when **today’s close > yesterday’s high**.

Typical hold is only a few days (~3–4 in our rebuild).

### Live fill

Match the backtest with **MOC / last minutes of RTH**, not next-open. Developing RTH “armed” uses Alpaca last as a proxy for today’s close; the research fill is still the closing print.

---

## Claimed performance (video, ~30y SPY)

| Metric | Claimed |
|--------|---------|
| Trades | 128 |
| Win rate | 78% |
| Profit factor | 3.0 |
| Avg gain / trade | 0.8% |
| CAGR | 3% |
| Time in market | ~5% |
| Risk-adj (CAGR / exposure) | ~52% |
| Max drawdown | 14% |

Absolute CAGR is modest; the point is **selective exposure** (~95% cash) so capital can sit in other sleeves.

---

## Independent rebuild (this repo)

Yahoo SPY daily ~1993–2026, same rules, zero friction:

| Metric | Ours |
|--------|------|
| Trades | 127 |
| Win rate | 75.6% |
| Profit factor | 2.99 |
| Avg gain | 0.74% |
| CAGR | 2.82% |
| Exposure | 5.9% |
| Max DD (MTM) | 14.0% |

Claim largely **replicates**. Friction 0.25% round-turn softens PF to ~2.1. Futures-style leverage (2×/3×) scales expectancy and drawdown together — prefer **MES/ES** over options for this ~0.8% average move.

HTML report (claim check + friction + leverage tabs):

`reports/research/deep_pullback/deep_pullback_spy.html`

Generator:

`scripts/research/generate_deep_pullback_report.py`

### Multi-symbol TimescaleDB scan

`scripts/research/run_deep_pullback_universe.py` — indexes (SPY/QQQ from DB; IWM/DIA/VTI/MDY via Yahoo when missing) + 150 top-ADV + 150 random from the ALPACA daily universe. Status: `docs/status_log/edge_hunt/2026-09-17_deep_pullback_universe.md`.

---

## Live monitor

| Piece | Path / detail |
|-------|----------------|
| Feature playbook | [`docs/features/deep_pullback_spy.md`](../features/deep_pullback_spy.md) |
| Scanner | `scripts/scanners/deep_pullback_spy.py` |
| Logic | `utils/scanning/deep_pullback_spy.py` |
| Cron | `deep_pullback_spy` — ~5m RTH; every minute 15:45–16:00 ET |
| Dashboard | `/hot` Detector **DP** (`timeframe=dp`) |

Status on `/hot`:

- **waiting** — approaching (early arm: above SMA200 and getting close to the 15d-low / sharp-day stack)
- **armed** — all three entry conditions on live last → buy MOC
- **filled** — after you mark Bought
- **exit_signal** — QS exit (close / last > prior day high)

`/hot` **Resist** = MOC target (`moc_max` = min of 15d-low cap and worst-1d ret cap); **Support** = SMA200 floor.

Telegram:

- **Approaching** only in the last **15 minutes** before 16:00 ET, with an explicit target: *If SPY stays at or below X (still above SMA200), send MOC buy at ~15:57.* After that, `/hot` shows **current / target** (green while still buy-ok).
- **BUY NOW** from **15:57** only while last is still at/below the MOC target.
- **EXIT** uses SELL NOW toggle.

---

## Notes / caveats

- Historical backtest, not a forecast; video results exclude commissions/slippage.
- Not graded against edge-hunt promote gates (Sharpe vs SPY, etc.) — treated as a **small selective sleeve**.
- Sit-out below SMA200 explains flat stretches in 2001–02, 2008, 2022.
- One large loser (e.g. 2011) drives much of the ~14% unlevered MDD.
