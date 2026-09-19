# QS Five Swing — fill realism (2018–2026)

**Verdict:** On SPY, **oracle MOC stays the upper bound**. Honest **5m decision clocks** lose ~2–4pp CAGR vs oracle; **next-open** keeps most of the edge but worsens MDD. A **15:45 15m** proxy looks almost as good as oracle because that bar’s OHLC already spans into the close — treat 5m clocks as the live-relevant test.

**Wall-clock:** SPY 5m backfill ~19m (IB 8824, 2018→2026); suite 5m ~16s, 15m ~8s.

## Data

| TF | Range | Bars |
|----|-------|------|
| Yahoo 1d | aligned to intra | 2188 sessions |
| IB 5m | 2018-01-02 → 2026-09-17 | 179,166 |
| IB 15m | 2018-01-02 → 2026-09-17 | 56,737 |

## Books (combined S1→S5, exit = settled close > prior high)

| Book | n | E | PF | CAGR | MDD | Prec / Recall |
|------|---|---|----|------|-----|----------------|
| **oracle_moc** | 258 | 0.365 | 1.80 | **11.04** | -21.4 | — |
| oracle fric 0.03 | 258 | 0.335 | 1.72 | 10.05 | -21.8 | — |
| oracle fric 0.10 | 258 | 0.265 | 1.55 | 7.76 | -24.5 | — |
| oracle fric 0.25 | 258 | 0.115 | 1.22 | 3.02 | -30.6 | — |
| **next_open** | 251 | 0.352 | 1.68 | 10.23 | **-29.1** | — |
| proxy 15:45 5m | 271 | 0.273 | 1.58 | 8.51 | -16.0 | 0.80 / 0.85 |
| proxy 15:50 5m | 258 | 0.238 | 1.48 | 6.91 | -19.8 | 0.83 / 0.86 |
| proxy 15:55 5m | 259 | 0.261 | 1.53 | 7.67 | -17.8 | 0.88 / 0.93 |
| proxy 15:45 5m + IBS buf 0.05 | 262 | 0.296 | 1.63 | 8.94 | -15.7 | 0.82 / 0.83 |
| proxy 15:45 **15m** | 261 | 0.344 | 1.75 | 10.47 | -17.8 | 0.93 / 0.96 |

Prec/recall = agreement of proxy arm vs settled oracle signal day (not trade P&L).

## Interpretation

1. **Oracle MOC** on this window: CAGR ~11%, PF ~1.8 (selective combined sleeve).
2. **Friction-as-noise:** 0.03% barely dents; 0.25% RT collapses CAGR to ~3% — cost stress ≠ close-guess error.
3. **Next-open (strict executable):** CAGR ~10.2% still fine; MDD worsens to **-29%** (gap risk).
4. **5m proxies (honest early arm):** later clock → higher agreement with oracle, but P&L still below oracle because false arms fill MOC on days that should have been skipped. Best 5m buffered 15:45 ~8.9% CAGR.
5. **15m @ 15:45 is misleadingly close to oracle** — IB 15m bar starting 15:45 includes almost the whole final 15 minutes, so “developing” close ≈ settled close. Prefer **5m** for live decision tests.

## Code / reports

- `utils/research/qs_five_swing_realism.py`
- `scripts/research/run_qs_five_swing_realism.py`
- HTML: `reports/research/qs_five_swing_realism/qs_five_swing_realism_5m.html` (+ `_15m.html`)

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\research\run_qs_five_swing_realism.py --intraday-tf 5m --start 2018-01-01
```

S4 proxy uses **prior-session ADX(5)** (causal; today’s ADX needs the close).

No live wiring.

**Follow-up (1m 15:57 vs 15:58):** see `2026-09-18_qs_five_swing_1m_moc.md` — both clocks recover ~10.4–10.6% CAGR vs oracle ~10.9 on the candidate-day 1m span.
