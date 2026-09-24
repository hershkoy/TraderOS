# ROC Rebound Strategy (SPY)

Short-term mean reversion on **SPY** using 7-day Rate of Change: buy after a sharp decline, hold until ROC recovers above a positive threshold.

**Source (for reference):** [I Tested a Simple ROC Strategy on 33 Years of SPY Data](https://www.youtube.com/watch?v=SF1OtXR5crQ) — Quantified Strategies (Håkan Samuelsson / Oddmund Groette), 2026-09-22.

---

## Idea

ROC is usually a momentum gauge. Here it is inverted into a rebound rule:

- A 7-day drop below −3% marks short-term weakness.
- Stay long until 7-day ROC closes above +3%.
- No stop, no profit target — one long at a time.

---

## Rules

### Entry

If flat and **7-day ROC closes below −3%**, buy SPY at the **next session’s open**.

ROC(7) = `(close / close[7] − 1) × 100`.

### Exit

If long and **7-day ROC closes above +3%**, sell at the **next session’s open**.

### Risk

No stop-loss and no defined profit target.

### Live fill note

Match the research book with **next-open** fills after the signal close (not same-bar close, not MOC).

---

## Claimed performance (video, ~33y SPY)

| Metric | Claimed |
|--------|---------|
| Trades | 146 |
| Win rate | 71.2% |
| Profit factor | 2.40 |
| Compounded $1 | ~$9.15 |
| Max drawdown | ~29% |

### Eras (claimed)

| Era | n | WR | PF |
|-----|---|----|----|
| 1993–2004 | 54 | 72.2% | 2.71 |
| 2005–2015 | 51 | 64.7% | 2.07 |
| 2016–2026 | 41 | 78.0% | 2.45 |

Author caveats: in-sample only; excludes commissions, slippage, taxes, dividends; no walk-forward / live validation.

---

## Independent rebuild (this repo)

Yahoo SPY daily ~1993-01-29 → 2026-09-16, same rules, zero friction:

| Metric | Ours | Claimed |
|--------|------|---------|
| Trades | 147 | 146 |
| Win rate | 71.43% | 71.2% |
| Profit factor | 2.399 | 2.40 |
| Final equity ($1) | 9.11 | ~9.15 |
| Expectancy / trade | +1.65% | — |
| Max DD (trade equity) | 29.0% | ~29% |

### Eras (ours vs claimed)

| Era | Ours n / WR / PF | Claimed |
|-----|------------------|---------|
| 1993–2004 | 55 / 72.7% / 2.71 | 54 / 72.2% / 2.71 |
| 2005–2015 | 51 / 64.7% / 2.08 | 51 / 64.7% / 2.07 |
| 2016–2026 | 41 / 78.1% / 2.45 | 41 / 78.0% / 2.45 |

Claim **replicates** (off by one early-era trade). Round-turn friction 0.25%: n=147 WR 69.4% PF **2.12** E +1.40% eq 6.34 MDD 30.3%.

### Robustness (ours)

- ROC length 4–11: all profitable; PF ~1.79–2.46 (video said 1.75–2.40).
- Exit +1%…+5% (entry fixed −3%): PF ~2.03–2.62; n falls from 184 → 81.

TimescaleDB SPY (IB prefix + Alpaca, 2010→2026) is a shorter window: n=61 WR 78.7% PF 4.62 — not the claim-check book.

### Universe sample (seeded 50, with slippage)

`scripts/research/run_roc_rebound_universe.py` — 12 ETF + 19 large + 19 small, friction ETF 0.10% / large 0.25% / small 0.40% RT.

| Bucket | pooled n | PF | E | med PF / % PF>1 |
|--------|----------|----|---|-----------------|
| ETF | 1114 | 1.54 | +0.89% | 1.57 / 92% |
| Large | 1227 | 1.34 | +0.84% | 1.43 / 84% |
| Small | 1245 | 1.12 | +0.45% | 1.34 / 79% |

**No universe promote** — edge thins with slippage; SPY remains the clean claim-check. Status: `docs/status_log/edge_hunt/2026-09-22_roc_rebound_universe.md`.

### ETF overlay grid (soft-interesting)

18 ETFs, fric 0.10%. **A3 = SMA200 + vol-scaled entry** (exit still +3% ROC): n=781 PF **2.38** E +1.42 MDD -32 vs A0 PF 1.69 E +1.04 MDD -66. Eras all PF>2; drop-top-3 holds. Time-stops / vol-alone no promote. Status: `docs/status_log/edge_hunt/2026-09-22_roc_rebound_etf_grid.md`. Runner: `scripts/research/run_roc_rebound_etf_grid.py`.

### Walk-forward (2026-09-22)

Expanding annual OOS 2005–2026 + holdout IS≤2015/OOS≥2016. Frozen A3 pooled OOS **n=694 PF 2.34**; holdout OOS PF **2.49**; IS picks A3 every fold. Fragile years 2008/2011/2020 (PF ≪ 1). Verdict **SOFT_PASS_OOS_FRAGILE_YEAR** — not live-wired. Status: `docs/status_log/edge_hunt/2026-09-22_roc_rebound_walkforward.md`. Runner: `scripts/research/run_roc_rebound_walkforward.py`.

### Single-slot occupancy vs SPY (2026-09-22)

Pre-declared holdout 2018–2026 + max_open=1 equity. Holdout occ **n=56 PF 1.20**; Sharpe **0.12** vs SPY **0.72**; MDD −34% vs SPY −25%. **FAIL** Phase-5 gates. Pooled PF ~2.3 does not survive one-slot capital. Status: `docs/status_log/edge_hunt/2026-09-22_roc_rebound_occupancy.md`. Runner: `scripts/research/run_roc_rebound_occupancy.py`.

### Capital-constraint fixes (2026-09-22)

Ranked deepest-ROC, SPY regime, rel-weak, time-stop 7, deep−5%, 4/5 slots, Rf on cash. **Time-stop** is the main lift (Sharpe 0.12→0.68, MDD −34%→−11%). Best **B9** (regime+tstop7+4slot+Rf4%): Sharpe **0.93** MDD −9% (beats SPY) but still **FAIL** Sharpe>1. Verdict **SOFT_IMPROVE_NO_GATE**. Status: `docs/status_log/edge_hunt/2026-09-22_roc_rebound_capital_fix.md`.

---

## Assessment gate

Per `docs/assessing_strategies.md` / edge-hunt norms:

- PF ~2.4 and era PF > 2 look healthy **in-sample**.
- Trade count ~147 is thin for law-of-large-numbers confidence.
- MDD ~29% is large vs selective sleeves like Deep Pullback (~14%).
- High time-in-market vs a cash-heavy bounce sleeve; not a Sharpe>1 promote candidate without portfolio context.
- Treat as **claim-check / research sleeve**, not a hard promote.

---

## Code

| Piece | Path |
|-------|------|
| Core backtest | `utils/research/roc_rebound.py` |
| Runner + sweeps | `scripts/research/backtest_roc_rebound.py` |
| Unit tests | `tests/unit/test_roc_rebound.py` |
| Outputs | `reports/research/roc_rebound/` |

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\research\backtest_roc_rebound.py
python scripts\research\backtest_roc_rebound.py --friction 0.25 --refresh-yahoo
```
