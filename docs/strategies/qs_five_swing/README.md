# QS Five Swing Strategies (SPY)

Five short-term mean-reversion sleeves on **SPY**, each buying weakness and exiting on the first close above the prior day's high. Combined as a **single-slot** book (priority S1→S5 when flat).

**Source:** [5 Swing Trading Strategies (Backtested + Trading Rules)](https://www.quantifiedstrategies.com/swing-trading-strategies-for-beginners/) — Quantified Strategies (video [KuXV0LRfJx8](https://www.youtube.com/watch?v=KuXV0LRfJx8)).

Related sleeve in this repo: [Deep Pullback](../deep_pullback/README.md) (same QS exit).

---

## Rules

Shared **exit:** sell at close when `close > prior day's high`.

| ID | Entry (buy at close) |
|----|----------------------|
| S1 Band+IBS | `close < (10d high - 2.5 * 25d avg(high-low))` and `IBS < 0.3` |
| S2 Turnaround Tuesday | Monday and close down two sessions in a row |
| S3 5-Day Low | `close < min(low of prior 5 days)` |
| S4 Vol + ADX | today's range `< min(range of prior 6)` and `ADX(5) > 40` |
| S5 10d High Reject | `high > max(high of prior 10)` and `IBS < 0.15` |

`IBS = (close - low) / (high - low)` (0 when range is 0).

**Combined:** allocate 100% to one position; ignore new buys while in a trade; on a flat day with multiple signals, take the first in priority **S1→S5**.

---

## Claimed performance (article / video)

Article backtests include ~**0.03% cost per trade**. Combined claim: **n=975**, avg **+0.38%**/trade, **CAGR 11.1%**, exposure **41%**, MDD **23%**, $100k→~$3.2M; losing years **2008 (−1%)** and **2018 (−13%)**.

Per-strategy highlights: S1 ~8%/18%/23% MDD; S2 ~7.2%/11%/16% MDD (~271 trades); S3 ~8%/21%; S5 video ~3.3%/8% MDD (article S5 block looks copy-pasted from S2 — prefer video numbers).

---

## Independent rebuild (this repo)

Yahoo SPY daily **1993-01-29 → 2026-09-16** (8465 bars). Wall-clock ~13s for frictions 0 / 0.03 / 0.25.

### Friction 0.03% (matches article cost assumption)

| Book | n | E % | PF | CAGR % | Exp % | MDD % | $100k final |
|------|---|-----|----|--------|-------|-------|-------------|
| S1 | 407 | 0.71 | 2.54 | 8.62 | 18.2 | 23.8 | 1.60M |
| S2 | 310 | 0.82 | 3.69 | 7.61 | 12.1 | 16.7 | 1.17M |
| S3 | 478 | 0.56 | 2.14 | 7.90 | 22.1 | 23.8 | 1.28M |
| S4 | 388 | 0.36 | 2.08 | 4.07 | 13.9 | 14.4 | 0.38M |
| S5 | 174 | 0.57 | 3.59 | 2.98 | 7.2 | 8.9 | 0.27M |
| **Combined** | **958** | **0.39** | **1.93** | **11.31** | **40.3** | **23.8** | **3.62M** |

Combined vs claim: n 958 vs 975, CAGR 11.3 vs 11.1, exp 40 vs 41, MDD 24 vs 23, E 0.39 vs 0.38 — **largely replicates**. Only calendar losing year in our fric-0.03 trade-sum book is **2018 (−14.4%)**; 2008 is not a net loser on that metric.

Friction 0 (no costs) combined: n=958, CAGR 12.3%, final ~$4.8M. Friction 0.25% RT stress: combined CAGR ~4.5%, PF ~1.36 — still positive but much softer.

HTML report: `reports/research/qs_five_swing/qs_five_swing_spy.html`

Generator:

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\research\generate_qs_five_swing_report.py
```

Logic: `utils/research/qs_five_swing.py`

---

## Fill realism / MOC clocks (2018–2026)

Oracle claim-check assumes **buy at the settled close (MOC)**. Live Arca MOC locks at **15:59 ET**. Tested with IB **1m** developing OHLC at **15:57** and **15:58**, plus a **15m@15:45 proximity → 1m decide** funnel (same live path as DP). Fill stays settled close; exit unchanged.

**Verdict:** **15:57 ≈ 15:58** (trivial gap). Both recover ~10.4–10.6% CAGR vs oracle **10.86%**. Prefer **15:57** for ~2 min headroom before lock (matches DP buy-now); 15:58 is a hair better on paper only. Honest **5m@15:55** is much weaker (~7.5%) — do not treat 15m@15:45 as an early decision (that bar spans into the close).

| Book | n | E % | PF | CAGR % | MDD % | Prec / Recall |
|------|---|-----|----|--------|-------|----------------|
| **oracle_moc** | 256 | 0.36 | 1.78 | **10.86** | -21.4 | — |
| **proxy 15:58** (1m) | 253 | 0.355 | 1.78 | **10.56** | -16.0 | 0.93 / 0.92 |
| **proxy 15:57** (1m) | 254 | 0.349 | 1.77 | **10.42** | -16.0 | 0.93 / 0.92 |
| **prox15m∧1m 15:58** | 252 | 0.352 | 1.77 | **10.44** | -16.0 | 0.94 / 0.92 |
| **prox15m∧1m 15:57** | 253 | 0.349 | 1.77 | **10.36** | -16.0 | 0.94 / 0.91 |
| next_open | 249 | 0.346 | 1.67 | 10.04 | **-29.1** | — |
| proxy 15:55 (5m) | 257 | 0.255 | 1.51 | 7.47 | -17.8 | 0.88 / 0.93 |

Window: Yahoo 1d + IB 1m **2018-01-19 → 2026-09-16** (candidate-day 1m backfill). Prec/recall = arm agreement vs settled oracle day.

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\research\run_qs_five_swing_realism.py --intraday-tf 1m --start 2018-01-01 --clocks 15:57,15:58 --with-5m-baseline
```

HTML: `reports/research/qs_five_swing_realism/qs_five_swing_realism_1m_moc.html`  
Status: `docs/status_log/edge_hunt/2026-09-18_qs_five_swing_1m_moc.md` (5m/15m: `2026-09-18_qs_five_swing_realism.md`)

---

## Live on `/hot`

| Piece | Detail |
|-------|--------|
| Detector | **QS** (`timeframe=qs`) |
| Cron | `qs_five_swing_spy` — ~5m RTH; every minute **15:45–16:00** ET |
| Data | TimescaleDB SPY `1d` + optional `1m` H/L + Alpaca last |
| Resist | Active sleeve MOC close ceiling (priority S1→S5) |
| Last | `current / target` — **green** while buy-ok, **red** if above target / signal gone |
| Alerts | **Approaching** from 15:45; **BUY NOW** from **15:57** if still ≤ target; EXIT on QS exit |

Enable QS Approaching / Armed / Sell in the `/hot` notify matrix. Mark **Bought** after MOC fill so EXIT tracks.

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\scanners\qs_five_swing_spy.py --dry-run
```

---

## Notes / caveats

- Historical claim-check, not a forecast. Treated as **selective sleeves** (same posture as Deep Pullback) — not graded on edge-hunt Sharpe>1 promote gates.
- Strategy 1 uses the article **2.5×** range multiplier (easy to miss in short video summaries).
- Combined fill mix (fric 0): S4 294, S1 252, S3 148, S5 140, S2 124 — priority and overlap change which sleeve “owns” a day.
- Live path: `utils/scanning/qs_five_swing_spy.py` / CronRunner `qs_five_swing_spy` (same MOC clocks as DP).
- Status (claim-check): `docs/status_log/edge_hunt/2026-09-18_qs_five_swing.md`

