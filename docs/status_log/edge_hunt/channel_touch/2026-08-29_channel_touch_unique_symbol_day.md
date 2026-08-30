# Unique-symbol/day instead of RS top1 — 2026-08-29

RS top1 was a **cross-symbol capacity cap** (one name per calendar day), not a
quality filter. Live buying is allowed to take every distinct symbol that
triggers; the only same-day drop is a second fill in the **same** ticker
(keep the earliest).

`--max-entries-per-day 1` still restores RS vs SPY top1 if wanted.

## 15m full H2 span10 (friction 0.10)

| Book | n | E | PF | names/day |
|------|---|---|----|-----------|
| All fills | 67416 | +0.30 | 1.55 | med 30, p90 75, max 231 |
| Unique symbol/day | 65561 | +0.30 | 1.55 | same (1855 same-symbol extras dropped) |
| RS top1 | 1766 | +0.30 | 1.48 | 1 |

The 67k to 1.7k drop was the cap, not a 97% reject. Unique-symbol is the book
that matches the live rule.

## Daily H2 span365 (friction 0.25)

| Book | n | E | PF | names/day |
|------|---|---|----|-----------|
| Unique symbol/day (= all) | 2253 | +2.80 | 2.21 | med 2, p90 5, max 30 |
| RS top1 | 981 | +3.11 | 2.34 | 1 |

Daily HTML was already max/day=all (n=2253). Nightly Telegram now matches that.

## Code

- `keep_one_per_symbol_day` in `scripts/research/backtest_channel_touch_trades.py`
- `LIVE_DEFAULTS["max_entries_per_day"] = 0`
- Nightly / `scan_live_triggers` unique-symbol first, then optional RS cap
