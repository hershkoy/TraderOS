# 15m L3 wait-12 signal-close — full IB universe (2026-10-10)

## Why

`current_best` 15m L3 is the **300 ADV** panel. Expand to the full IB 15m list (~1480) with the same stack.

## Stack

```bat
venv\Scripts\activate && set PYTHONPATH=.
python scripts\research\backtest_channel_touch_trades.py --preset 15m --all-symbols --entry-mode l3_touch --min-l3-wait-bars 12 --realistic-fill --realistic-fill-mode signal-close --touch-error-pct 0 --workers 4 --load-workers 8
```

Preset: in-channel, span≤10d, friction 0.10, `max_entries_per_day=1` (RS top1). Wall-clock **~1093s** (sym list 43s, load 17s, scan 971s, RS 57s). Loaded **1480/1480**.

## Results

| Book | n | E% (fric0.10) | PF | notes |
|------|---|---------------|-----|-------|
| **300 ADV RS top1 (current_best)** | **1744** | **+0.13** | **1.22** | keeper |
| Full IB RS top1 | 1772 | **−0.04** | **0.95** | n barely rises (1/day cap); weaker names win RS |
| Full IB unique-symbol/day | 46885 | **−0.04** | **0.93** | quality filter → 47082 raw tags; all year buckets ≤1 PF net |

Full RS top1 year net: 2018-19 −0.10 / 0.83; 2020-21 −0.12 / 0.83; 2022-23 **+0.11 / 1.17**; 2024-26 −0.06 / 0.92.

Unique net year: all red/flat (2022-23 E −0.07 PF 0.89).

Beyond-width A/B on RS top1 does not salvage (0.25 → E −0.03 PF 0.96).

## Verdict

**No promote. Do not replace current_best.** Expanding past the liquid 300 ADV panel destroys the L3 wait-12 signal-close edge under both RS top1 and unique-symbol. Same pattern as full-IB 15m H2 vs 300 ADV: extra names dilute the book; RS top1 n stays ~1/day so you only swap in worse winners.

Keep `reports/ascending_channels/current_best/15m_channel_touch.html` on the 300-name CSV.

## Artifacts

- RS top1: `reports/ascending_channels/2026-10-10/channel_touch_15m_trades_20261010_145003.csv`
- Raw: `.../channel_touch_15m_trades_raw_20261010_145003.csv`
- Unique (post-filter): `.../channel_touch_15m_l3_wait12_full_unique.csv`
- Summary: `.../channel_touch_15m_trades_summary_20261010_145003.txt`
