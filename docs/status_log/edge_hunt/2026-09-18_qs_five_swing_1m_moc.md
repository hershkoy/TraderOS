# QS Five Swing — 1m MOC clocks 15:57 vs 15:58 (2018–2026)

**Verdict:** On SPY Arca MOC (lock **15:59 ET**), deciding at **15:57** vs **15:58** is nearly the same. Raw 1m **15:58** edges **15:57** by ~0.14pp CAGR; proximity∧1m funnel same story (~0.08pp). Both recover **~10.4–10.6% CAGR** vs oracle **~10.9%** — far closer than 5m@15:55 (~7.5%). Prefer **15:58** only for a slightly later print; either is fine for live MOC placement with ~1–2 min headroom.

**Wall-clock:** candidate-day IB 1m backfill ~25m (client 8825, 583 QS/proximity days); suite ~22s.

## Data

| Source | Range | Notes |
|--------|-------|-------|
| Yahoo 1d | 2018-01-19 → 2026-09-16 | 2176 sessions (aligned to 1m) |
| IB 1m | 2018-01-19 → 2026-09-16 | 293,610 bars; **752** distinct NY sessions (candidate-day + prior 2025 span) |
| IB 15m | same window | proximity gate @ 15:45 bar start |
| IB 5m | same window | baseline `proxy_1555_5m` |

Candidate backfill: `scripts/research/backfill_ib_1m_symbol.py --candidate-days-from-qs-realism --start 2018-01-01` (583 days; 50 already present; rest saved ~390 bars/day). Path is under `scripts/research/` (not `scripts/data/`) due to write ACL.

## Books (combined S1→S5; fill = settled close; exit = close > prior high)

| Book | n | E | PF | CAGR | MDD | Prec / Recall |
|------|---|---|----|------|-----|----------------|
| **oracle_moc** | 256 | 0.360 | 1.78 | **10.86** | -21.4 | — |
| **proxy_1558** (1m only) | 253 | 0.355 | 1.78 | **10.56** | -16.0 | 0.931 / 0.919 |
| **proxy_1557** (1m only) | 254 | 0.349 | 1.77 | **10.42** | -16.0 | 0.929 / 0.917 |
| **prox15m∧1m 15:58** | 252 | 0.352 | 1.77 | **10.44** | -16.0 | 0.936 / 0.917 |
| **prox15m∧1m 15:57** | 253 | 0.349 | 1.77 | **10.36** | -16.0 | 0.936 / 0.913 |
| next_open | 249 | 0.346 | 1.67 | 10.04 | **-29.1** | — |
| proxy_1555_5m | 257 | 0.255 | 1.51 | 7.47 | -17.8 | 0.884 / 0.928 |

Prec/recall = arm agreement vs settled oracle signal day.

## Interpretation

1. **1m late clocks ≈ oracle** on this window — false arms still exist (~7% precision miss) but P&L gap is small (~0.3–0.5pp CAGR).
2. **15:57 vs 15:58:** trivial difference; 15:58 slightly higher n-quality (fewer false arms / slightly better E). Not worth fishing further clocks.
3. **Proximity @15:45 then 1m decide** barely changes vs raw 1m (same MDD −16%, ~same CAGR) — useful as a live watchlist filter, not a P&L lifter here.
4. **5m@15:55** remains the honest early-clock baseline (~7.5% CAGR) — do not confuse with 1m@15:57/58.
5. **Next-open** still ~10% CAGR with worse MDD (−29%).

## Live model (research only — not wired)

```
15m proximity ~15:45  -->  candidate day
1m decide 15:57 or 15:58  -->  arm?
MOC fill = settled close (lock 15:59 Arca)
```

## Code / reports

- `utils/research/qs_five_swing_realism.py` — `PROXY_CLOCKS_1M_MOC`, `backtest_proximity_then_1m`, `run_moc_1m_suite`
- `scripts/research/run_qs_five_swing_realism.py --intraday-tf 1m --clocks 15:57,15:58 --with-5m-baseline`
- `scripts/research/backfill_ib_1m_symbol.py` (client **8825**, port 4001; not during RTH vs live 8826)
- HTML: `reports/research/qs_five_swing_realism/qs_five_swing_realism_1m_moc.html`

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\research\backfill_ib_1m_symbol.py --symbols SPY --candidate-days-from-qs-realism --start 2018-01-01 --ib-client-id 8825 --ib-port 4001
venv\Scripts\activate && set PYTHONPATH=. && python scripts\research\run_qs_five_swing_realism.py --intraday-tf 1m --start 2018-01-01 --clocks 15:57,15:58 --with-5m-baseline
```

Earlier same-day 5m/15m realism: `2026-09-18_qs_five_swing_realism.md`. No live `/hot` / CronRunner MOC placement.
