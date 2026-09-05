# Channel-touch shakeout then second resist-break — 2026-09-05

## Why SXI stopped, and why the later rally was skipped

SXI 2020-08-11 @ 60.30 is a daily **H2 resist-break** (L1 2020-03-19 / H2 2020-07-16). 2020-08-25 @ 56.68 is the **6% ATR ceiling** (`gain_pct_net -6.25` = 6% stop + 0.25% friction). That first loss is not preventable with the current keeper stop.

The finder **stopped after the first close above resistance**, so the later re-break never emitted. Occupancy would have allowed a new trade after Aug 25.

This is **not** L3 `--shakeout-rebuy-bars` (close through support then reclaim; that rule **cancels** on a close above resistance and failed 2020-21).

## Rule

After a first H2 resist-break fill on the same frozen rails:

1. Require N closes back **inside** the channel (not still above resistance).
2. Support must not close through.
3. Fill the next close above resistance at the rail + slip.
4. Rebuy, not hold-through (`busy_until`).
5. At most one extra fill per H2 setup.
6. Window: remaining `max_l3_wait_bars=252` from H2.

No in-channel / RSI 50 (breakouts skip those). Span<=365, causal H2, unique-symbol/day, ATR k=2 clamp 1.5%–6%, 0.25% friction.

## Setup

Full ALPACA 1d universe (2216 + SPY), IB prefix, 2018-11-01 → 2026-08-27. One load, two scans (`min_inside` 1 and 5). Extras emitted with `parent_exit_reason`; hard-stop vs any-closed is a post-filter. Wall-clock **123s** (parquet cache hit).

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_shakeout_breakout.py --all-symbols --workers 4 --load-workers 8
```

## Results (0.25% friction)

Frozen causal unique-symbol H2 span365 (2026-09-03): **n=3364 E +2.40 PF 2.02**. Parent rows below are from the same scan as the extras (busy_until can skip a later parent if an extra is still open), so n is a bit lower.

### min_inside=1

| book | n | E | PF | 2020-21 | notes |
|------|---|---|----|---------|-------|
| Parent unique H2 span365 | 3119 | +2.37 | 1.996 | E +1.91 PF 1.79 | same-scan parent |
| **Sleeve any-closed** | **1439** | **+3.16** | **2.316** | **E +3.24 PF 2.39** | all year buckets + |
| Sleeve hard-stop | 511 | +2.31 | 1.964 | E +1.33 PF 1.51 | below parent; 2022-23 E +0.72 PF 1.24 |
| **Combined any-closed** | **4558** | **+2.62** | **2.098** | E +2.33 PF 1.97 | lifts parent and frozen 3364 / +2.40 / 2.02 |
| Combined hard-stop | 3630 | +2.36 | 1.992 | E +1.84 PF 1.75 | slight dilute vs parent |

Median still negative (fat tail). Sleeve any-closed med −1.54 vs parent −2.22.

### min_inside=5

| book | n | E | PF | 2020-21 |
|------|---|---|----|---------|
| Parent unique H2 | 3048 | +2.42 | 2.026 | E +1.91 PF 1.79 |
| Sleeve any-closed | 1786 | +2.70 | 2.093 | E +2.46 PF 2.02 |
| Sleeve hard-stop | 620 | +2.09 | 1.857 | E +1.77 PF 1.76; **2022-23 E +0.46 PF 1.15** |
| Combined any-closed | 4834 | +2.52 | 2.051 | E +2.11 PF 1.87 |
| Combined hard-stop | 3668 | +2.37 | 1.996 | E +1.89 PF 1.78 |

min_inside=5 any-closed still lifts, but the sleeve is weaker than min_inside=1. Hard-stop-only does not beat the parent.

## SXI sanity (loaded daily bars)

Parent 2020-08-11 @ 60.30, hard-stop 2020-08-25 @ 56.68, −6.25% net.

Extra (min_inside=1 and 5): **2020-10-05 @ 62.6564**, trail 2021-03-22 @ 97.353, **+55.13% net**, 36 inside bars, `parent_exit_reason=hard_stop`. Same L1/H2 rails as the Aug fill.

## Complete history (full daily universe)

Same recipe, ALPACA 1d union IB 1d (**2223 loaded + SPY**), IB prefix from **2006-01-01** through last stored bar (**2026-09-03**), windowed 504/252. SPY IB-prefix from **2010-01-04**. Wall-clock **144s** (merged cache hit).

`start=None` is **not** complete history: IB prefix only runs when Alpaca is empty or starts after `start`, so an unclipped load is Alpaca-native (2020–21 collapses, SXI 2020 disappears). `--complete-history` clips start at 2006-01-01 so prefix stitches.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_channel_touch_shakeout_breakout.py --complete-history --workers 4 --load-workers 8
```

### min_inside=1 (complete history)

| book | n | E | PF | 2020-21 | pre-2018 |
|------|---|---|----|---------|----------|
| Parent unique H2 span365 | 3137 | +2.41 | 2.013 | E +1.91 PF 1.79 | n=4 |
| **Sleeve any-closed** | **1445** | **+3.14** | **2.296** | **E +3.24 PF 2.39** | n=3 (noise) |
| Sleeve hard-stop | 514 | +2.27 | 1.929 | E +1.33 PF 1.51 | n=2 |
| **Combined any-closed** | **4582** | **+2.64** | **2.103** | E +2.33 PF 1.97 | n=7 |
| Combined hard-stop | 3651 | +2.39 | 2.001 | E +1.84 PF 1.75 | n=6 |

Same shape as the 2018-11-01 2216-name scan (sleeve n=1439 E +3.16 PF 2.32). Extra names and 2006–2017 add almost no trades (`max_low_pivots` still needs the 504/252 window). SXI 2020-10-05 extra still **+55.13%**.

min_inside=5 complete-history sleeve n=1793 E +2.67 PF 2.08 — still weaker than min 1. Gate unchanged.

## Gate

Sleeve any-closed **min_inside=1** beats the frozen unique H2 book on E and PF, combined unique-symbol **lifts** rather than dilutes, and **2020-21 is healthy** (the hole that killed L3 support-reclaim rebuy).

**Soft-promote and productionize any-closed + min_inside=1.** `--shakeout-breakout` is **on** for nightly (close fills) and `current_best` 1d (realistic next-mid). Prefer this over hard-stop-only and over min_inside=5.

**Nightly re-arm (2026-09-05):** `LIVE_DEFAULTS["shakeout_breakout"]=True`. After a taken H2 fill, if the first trade has exited (`busy_until` / ATR+10% trail occupancy), Telegram the next close above resist after ≥1 inside close. Occupancy skips extras while the first trade is still open. 1d `/hot` re-arms the same way. 15m live H5 path stays first-fill-and-done. Do not reuse `--shakeout-rebuy-bars`.

## Realistic-fill current_best 1d (same next-mid recipe as 2026-09-04)

**Caveat:** this scan converted daily midnight UTC to New York for the 15m join. Rebuild with `_as_session_date` before treating n=1239 as honest ([session date](2026-09-05_channel_touch_session_date.md)).

2018-11-01 → 2026-08-27, 2216 ALPACA 1d, IB prefix, window 504/252, `--realistic-fill --realistic-fill-mode next-mid`, unique-symbol/day, 0.25% friction. Wall-clock **4068s** (1d cache 16.6s + IB 15m 255s + scan 3796s).

| Book | n | E | PF | vs frozen realistic unique H2 (n=713 E +1.78 PF 1.72) |
|------|---|---|-----|------|
| Parent unique span365 | 655 | +1.93 | 1.79 | occupancy from extras skips some later first-breaks (713→655) |
| **Sleeve extras** | **584** | **+2.66** | **2.07** | 2020-21 E +2.61 PF 2.04; all year buckets + |
| **Combined unique** | **1239** | **+2.27** | **1.92** | **lifts** n/E/PF; RS top1 n=635 E +2.42 PF 1.97 |

SXI: parent 2020-08-11 @ 60.22 hard-stop 2020-08-25 −6.25% still there. Realistic extra is **2020-11-05 @ 66.55** trail 2021-03-22 **+46.04%** (close-fill extra was 2020-10-05 @ 62.66 / +55% — 15m next-mid reprices/delays). A later extra 2024-02-28 is in the book.

HTML: deleted old `current_best/1d_channel_touch.html` and `1d_h2_resist_break.html` first, then copied `...h2_break_span365_shakeout_realistic_20260905_135228.html`. Do not `copy /Y` onto a hard link. 15m books and `1d_l3_touch.html` unchanged. Nightly remains **unrealistic close / rail-clip**; HTML is **next-mid** (**pre-`_as_session_date`** — [session date](2026-09-05_channel_touch_session_date.md)).

## Code

- [`scripts/research/backtest_channel_touch_trades.py`](../../../scripts/research/backtest_channel_touch_trades.py): `_shakeout_breakout_fill`, `--shakeout-breakout`
- [`scripts/research/backtest_channel_touch_shakeout_breakout.py`](../../../scripts/research/backtest_channel_touch_shakeout_breakout.py)
- [`scripts/research/backtest_channel_touch_h2_break.py`](../../../scripts/research/backtest_channel_touch_h2_break.py): `--shakeout-breakout` on the daily H2 recipe
- [`utils/scanning/channel_touch.py`](../../../utils/scanning/channel_touch.py): live occupancy + `LIVE_DEFAULTS`
- [`scripts/scanners/channel_touch_nightly.py`](../../../scripts/scanners/channel_touch_nightly.py)
- [`utils/scanning/channel_touch_15m.py`](../../../utils/scanning/channel_touch_15m.py) `walk_h2_resist_asof` (1d `/hot` only; 15m live stays off)
- Tests: `tests/unit/test_l3_touch_entry.py`, `tests/unit/test_channel_touch.py`, `tests/unit/test_channel_touch_nightly.py`, `tests/unit/test_channel_touch_15m.py`

## Artifacts

- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_any_min1_20260905_114330.csv` (n=1439)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_hs_min1_20260905_114330.csv` (n=511)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_any_min5_20260905_114330.csv` (n=1786)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_hs_min5_20260905_114330.csv` (n=620)
- `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_summary_20260905_114330.csv`
- Complete history (IB prefix from 2006, 2223 names, through 2026-09-03):
  - `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_fullhist_any_min1_20260905_122540.csv` (n=1445)
  - `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_fullhist_hs_min1_20260905_122540.csv` (n=514)
  - `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_fullhist_any_min5_20260905_122540.csv` (n=1793)
  - `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_fullhist_hs_min5_20260905_122540.csv` (n=623)
  - `reports/ascending_channels/2026-09-05/channel_touch_shakeout_breakout_fullhist_summary_20260905_122540.csv`
- Realistic-fill current_best 1d (next-mid, unique span365):
  - `reports/ascending_channels/2026-09-05/channel_touch_full_h2_break_span365_unique_20260905_135126.csv` (n=1239)
  - HTML stamp `...h2_break_span365_shakeout_realistic_20260905_135228.html`
  - Stable trades `reports/ascending_channels/channel_touch_h2_break_span365.csv`
