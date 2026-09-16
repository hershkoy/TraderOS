# Last-15m consecutive-loss sit-out (2026-09-16)

Causal account-level circuit breaker on the frozen last-15m H2 book **n=2297**. Outcomes update state only for **taken** fills; skipped signals do not invent paper P&L. Occupancy **not** re-walked. Locked params only (no full-sample grid). **No promote.**

Runner: `scripts/research/overlay_last_15m_loss_streak.py`. Logic: `utils/research/last_15m_loss_streak.py`. Summary CSV: `reports/ascending_channels/2026-09-16/last_15m_loss_streak_summary.csv`.

## Baseline

| Book | n | E | PF | geo | 2022 E / PF | 2022-23 E / PF |
|------|---|---|----|-----|-------------|----------------|
| Last-15m H2 | 2297 | +0.27 | 1.09 | 1.10 | −2.37 / 0.41 | −0.90 / 0.73 |

## 1. Diagnostic: are 2022-23 losses streak-clustered?

Consecutive-loss runs on the **unfiltered** chronological book:

| bucket | n | max_consec | mean_run | n_runs_ge5 | pct_trades_in_run_ge5 |
|--------|---|------------|----------|------------|------------------------|
| 2018-2019 | 141 | 10 | 3.22 | 6 | 31.2% |
| 2020-2021 | 693 | **26** | 3.29 | **33** | 38.4% |
| 2022-2023 | 417 | 21 | **4.24** | 20 | **48.7%** |
| 2024-2026 | 1046 | 27 | 3.53 | 53 | 39.8% |

`streak_longer_in_hostile` vs 2020-21: **False** (hostile max 21 < 26; n_runs_ge5 20 < 33). Mean run and pct-in-ge5 are higher in 2022-23, but bull years already have longer absolute streaks. The slaughter is not a unique streak-tail sensor the way a 5-loss breaker would need.

## 2. Locked rules

State machine: Active → after L consecutive taken losers (or dollar sum ≤ −15) → Sit S signals → Probe P taken; any probe win → Active; all P lose → Sit again.

| Rule | n | E | PF | geo | 2022 taken / skip | leftover 2022 E | 2022-23 E / PF | blunt? |
|------|---|---|----|-----|-------------------|-----------------|----------------|--------|
| count L5 S3 P2 | 1745 | +0.65 | 1.22 | **1.20** | 73 / 44 | **−2.62** | −0.46 / 0.86 | No |
| count L5 S3 P3 | 1853 | +0.53 | 1.18 | 1.17 | 92 / 25 | −1.89 | −0.42 / 0.87 | No |
| dollar ≤−15 S3 P2 | 1620 | +0.80 | 1.27 | 1.22 | 69 / 48 | −2.44 | −0.26 / 0.92 | No |

Pooled geo rises (sits through clusters in every regime), but **2022 leftover stays red** and is **worse** on the primary L5/S3/P2 rule (−2.62 vs −2.37). Same failure mode as rolling WR20: early / mid-cluster chases still print; sit-outs also fire in 2020-21 / 2024-26 (165 and 231 skips on L5/S3/P2). Holdout 2022-26 on L5/S3/P2 is E +0.51 PF 1.17 — not a 2022 fix.

## 3. Exploratory L∈{4,5,6} (report-only)

| L | n | geo | 2022 E | 2022-23 E |
|---|---|-----|--------|-----------|
| 4 | 1643 | 1.24 | −1.77 | −0.37 |
| 5 | 1745 | 1.20 | −2.62 | −0.46 |
| 6 | 1844 | 1.25 | −2.22 | −0.39 |

Geo fishing does not clear 2022. Do not promote L4/L6.

## Verdict

**No promote.** Diagnostic fails the longer-tail gate vs 2020-21; locked count and dollar rules leave 2022 red (primary leftover worse). Do not fish more L/S/P ratios; do not wire `/hot`. Closest prior: [regime switch WR20](2026-09-13_last_15m_regime_switch.md) (same leftover-2022 pattern).
