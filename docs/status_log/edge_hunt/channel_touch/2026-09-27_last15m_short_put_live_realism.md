# last-15m short-put live realism (listed + half-spread + occupancy)

**Date:** 2026-09-27  
**Verdict:** no promote (hard-stop path only; half-spread makes stops worse than constant-IV BS)

## What ran

Matched listed ATM puts for BS hard-stop rows (ADV≥$1M, buy≥2024-10), Friday-ladder expiry + rounded strike, Polygon day-bar mid as quote proxy, half-spread `max(0.05, 0.08*mid)` on entry credit / exit debit, FCFS occupancy at max_open 4/8/15.

Runner: `scripts/research/run_last_15m_short_put_live_realism.py`  
Helpers: `utils/research/short_put_live_realism.py`  
Artifacts: `reports/ascending_channels/2026-09-27/last_15m_short_put_live_realism_*.csv`, `*_summary.txt`, `live_realism_cache.jsonl`

Wall ~65 min (Polygon free ~4 rpm; many no_bars burned 4 aggs attempts).

## Results (n=100 attempted)

| slice | n | E (CS %) | PF | WR | notes |
|-------|---|----------|----|----|-------|
| mid fills | 48 | −2.00 | 0.06 | 10.4% | day close as mid |
| half-spread fills | 48 | −3.22 | 0.01 | 6.2% | sell mid−HS / buy mid+HS |
| BS same rows | 48 | −1.85 | 0.03 | 4.2% | constant-IV book |
| occ max_open=4/8/15 | 48 | −3.22 | 0.01 | 6.2% | skip=0 (short holds) |

- `exit_vs_bs` (live mid exit / BS exit): **med 0.860, mean 0.897** — stop buybacks ~14% cheaper than constant-IV BS on mid (not the IV×1.25 stress zone).
- Half-spread adds ~1.2pp of cash-secured loss vs mid and ~1.4pp vs BS on the same hard-stops.
- Hit rate 48/100: misses are mostly `no_bars` (thin names / wrong strike increment / free-tier history gaps), not a promote gate.

## Honest filter (2026-09-28)

Require `live_entry_date == buy_date` (option print on signal day). Slid entries (first print after signal, e.g. AVGO crash-day open/close) treated as no-fill.

| slice | n | note |
|-------|--:|------|
| status=ok | 48 | prior |
| slid entry | 14 | drop (incl. AVGO same-day open→close) |
| same-day among on-signal | 0 | all 3 same-day exits were slid |
| **honest ok** | **34** | entry on signal day |

| slice | n | E (CS %) | PF | exit_vs_bs med |
|-------|--:|---------:|---:|---------------:|
| prior half-spread | 48 | −3.22 | 0.01 | 0.860 |
| **honest half-spread** | **34** | **−3.28** | **0.01** | **0.862** |
| honest mid | 34 | −2.07 | 0.08 | |
| honest BS same rows | 34 | −1.87 | 0.03 | |

Dropping slid rows does **not** change the story: half-spread still ~1.4pp worse than BS; exit mids still ~0.86× BS. Hit rate honest **34/100**. Still **no promote**.

Artifacts: `*_honest_filter.csv`, `*_honest_summary.txt`. Runner summary now drops slid by default.

## Interpretation

1. **Stops are not secretly IV×1.25+ on mid** — live exit marks track closer to BS×0.85–0.90.
2. **Bid/ask haircut dominates** — realistic short-put hard-stops look worse than the BS overlay, not better.
3. **Occupancy is not the binding constraint** on this hard-stop-only sample (holds are short; max concurrent stayed under 4).
4. Full-book live realism (winners + TP 50% credit) still needed before any promote; this run only stress-tests the stop leg.

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
set POLYGON_REQUESTS_PER_MINUTE=4
python scripts\research\run_last_15m_short_put_live_realism.py --n 100
```

Resume-safe via `live_realism_cache.jsonl` (ok rows only).
