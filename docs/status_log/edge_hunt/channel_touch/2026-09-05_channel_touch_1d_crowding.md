# 1d unique book — crowding and max-concurrent (2026-09-05)

Follow-up to the [loser-filter](2026-09-05_channel_touch_1d_loser_filter.md) miss. No fitted score. Pre-registered structural / capital rules on `channel_touch_full_h2_break_span365_unique_20260905_135126.csv` (**n=1239 E +2.27 PF 1.92**, **pre-`_as_session_date`** — see [session date](2026-09-05_channel_touch_session_date.md)).

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\filter_channel_touch_crowding.py
```

## Exposure

Unconstrained unique-symbol equal-dollar is not a live book: calendar concurrent **max 85 / p95 57 / median 18**. `n_open_at_entry` median **29**. Same-day unique fills: median **2**, p90 **6**, max **14**.

## H1 — n_open at entry (overlapping holds)

Not monotone. 83% of fills already have **13+** names open (E +2.06 PF 1.83, 2020-21 +1.11 / 1.44). Sparse 0–12 buckets are small-n. High concurrent is the default state, not a rare loser tag. **Do not** use `n_open_at_entry` as a quality gate.

## H2 — same-day fill count (cross-symbol crowding)

This one is structural:

| n_day | n | E | PF | 2020-21 | 2024-26 |
|-------|---|---|----|---------|---------|
| 1 | 337 | +2.61 | 2.09 | +4.09 / 2.81 | +3.01 / 2.35 |
| 2 | 302 | **+4.10** | **2.73** | +3.01 / 2.30 | +3.79 / 2.54 |
| 3 | 249 | +1.29 | 1.51 | **−0.16 / 0.94** | +0.77 / 1.29 |
| 4–5 | 187 | +2.02 | 1.84 | +0.95 / 1.36 | +2.91 / 2.24 |
| 6–8 | 91 | **−0.34** | **0.88** | **−2.02 / 0.32** | +0.71 / 1.25 |
| 9+ | 73 | +0.43 | 1.16 | +2.99 / 2.10 | **−1.21 / 0.55** |

Days with **3** fills fail 2020–21. Days with **6+** are a drag in-sample. Quiet 1–2 name days are the edge.

Full-day `n_day` is known at an **EOD / nightly** scan (all closes in). It is a **look-ahead for next-mid** (later 15m fills that day are unknown). Prefer a rank among today's triggers, not “skip the first fill because the day will end busy.”

## Rules (vs baseline +2.27 / 1.92; 2020-21 +1.65 / 1.64; 2024-26 +2.39 / 1.98)

| rule | n | E | PF | 2020-21 | 2022-23 | 2024-26 | drop3 PF |
|------|---|---|----|---------|---------|---------|----------|
| **cap 2/day wait** | **933** | **+3.01** | **2.25** | **+2.72 / 2.10** | **+3.50 / 2.34** | **+2.89 / 2.21** | 2.08 |
| cap 2/day RS | 933 | +2.84 | 2.17 | +2.33 / 1.93 | +3.07 / 2.20 | +2.95 / 2.22 | 2.00 |
| cap 3/day wait | 1080 | +2.54 | 2.04 | +2.14 / 1.85 | +3.20 / 2.25 | +2.44 / 2.00 | 1.90 |
| skip days n>3 (EOD) | 888 | +2.75 | 2.13 | +2.45 / 2.00 | +2.99 / 2.16 | +2.76 / 2.14 | 1.96 |
| max_open 8 wait | 338 | +3.13 | 2.29 | +4.99 / 2.98 | +1.60 / 1.58 | +3.25 / 2.50 | 2.06 |
| max_open 5 wait | 226 | +3.57 | 2.51 | +5.13 / 2.81 | +2.15 / 1.85 | +3.91 / 2.99 | 2.15 |
| max_open 10 wait | 399 | +3.34 | 2.42 | +5.07 / 3.12 | +2.03 / 1.75 | +3.34 / 2.56 | 2.22 |

**cap 2/day by `wait_bars` (not RS)** beats baseline on E and PF in **all four** year buckets, keeps n=933, drop-top-3 PF 2.08. Wait beats RS on 2020–21. Unique-symbol/day stays (one fill per ticker); this only caps how many *names* the same calendar day.

`max_open` 5/8/10 also lift overall E/PF and 2020–21, but they are a **capital constraint** (you cannot hold 85 equal-dollar names). 2022–23 E/PF **fall** vs baseline under max_open 5/8/10. Treat as the live sizing book, not a quality rewrite of unique-symbol.

`skip days n>3` is EOD-honest and lifts, but uses the full-day count (next-mid leak). Cap-2 among that day's triggers is the causal version for nightly.

## Gate

**Soft-promote `cap_same_day=2` ranked by wait, not RS.** Do not replace unique-symbol with RS top1. Do not wire nightly or replace `current_best/1d_channel_touch.html` until that cap is an explicit live choice (`select_same_day_rs(..., rs_col="wait_bars", max_per_day=2)`) **and** next-mid is rebuilt with `_as_session_date`.

**Live capital:** greedy max concurrent **8** (wait tie-break) is the equal-dollar book you can actually hold; report it as capacity, not as a new edge. HTML already has Max concurrent.

Do **not** promote skip-crowded-days as a next-mid HTML filter. Do **not** promote n_open-at-entry buckets.

15m unique-filter note still stands for that stack (med ~30 names/day). This 1d book is med 2 / p90 6 — crowding is a rare busy tape, not the whole sample.

## Code

- [`scripts/research/channel_touch_robustness.py`](../../../scripts/research/channel_touch_robustness.py): `n_open_at_entry`, `same_day_fill_count`, `skip_crowded_days`, `cap_same_day`, `apply_max_open(..., tie_break=)`
- [`scripts/research/filter_channel_touch_crowding.py`](../../../scripts/research/filter_channel_touch_crowding.py)
- Tests: `tests/unit/test_channel_touch_robustness.py`

## Artifacts

- `reports/ascending_channels/2026-09-05/channel_touch_1d_crowding_20260905_182644.csv`
- `..._nopen_20260905_182644.csv`
- `..._nday_20260905_182644.csv`
