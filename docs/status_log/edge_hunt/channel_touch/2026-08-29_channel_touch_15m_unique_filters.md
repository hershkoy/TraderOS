# 15m unique-symbol filter hypotheses — 2026-08-29

Five new gates on the full IB 15m H2 span10 book after **unique-symbol/day**
(n=65561 E +0.30 PF 1.55, friction 0.10). Not RS top1. Prior-bar features
only. Expanding-year OOS: train years &lt; Y, apply to Y (first year has no
OOS). Detector v1 unchanged. **Research only.**

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\mine_15m_unique_filters.py
```

## The five hypotheses (written before looking at quintiles)

| ID | Idea | Rule |
|----|------|------|
| H1 | RS as a **floor**, not a rank | `rs_spy_126d >= 0` |
| H2 | Tight rail break (just cleared) | `1.00 <= channel_pos <= 1.10` |
| H3 | ATR **ceiling** (reject chop) | `atr_pct <= train p60` |
| H4 | Not already crowded **so far today** | `day_n_asof <= train p75` |
| H5 | Prior-bar volume confirmation | `volume_rel_20 >= 1` |

Derived features added without a rescan: `overshoot`, `wait_sessions`,
`buy_hour`, `day_n_asof` (running unique names with `buy_time <= now`; the
full calendar-day count is a 15m look-ahead and was dropped).

## Expanding-year OOS

| Book | n | E | PF | vs baseline |
|------|---|---|----|-------------|
| Baseline unique-symbol | 65561 | +0.30 | 1.55 | — |
| H1 RS floor | 28925 | +0.28 | 1.51 | worse |
| H2 tight break | 53486 | +0.23 | 1.40 | **hurts** |
| H3 ATR ceiling | 38494 | +0.26 | 1.53 | worse |
| H4 quiet so far | 45174 | +0.29 | 1.53 | flat/worse |
| **H5 vol confirm** | **39442** | **+0.34** | **1.63** | **only pre-registered lift** |
| Logistic p >= train median | 34400 | +0.37 | 1.67 | best leak-safe keep |
| Ridge pred gain > 0 | 60495 | +0.31 | 1.57 | barely filters |
| Post-hoc wide overshoot (train p80) | 12779 | +0.61 | 2.43 | **discovered after H2 failed** |

## Regression

L2 logistic on 16 prior-bar / geometry features, expanding year, keep if
score >= train median. Ridge on net gain, keep if pred &gt; 0.

Spearman vs net gain (n=65561): **overshoot +0.20**, atr_pct -0.17, width -0.14,
RS ~0, buy_hour ~0. Quintiles: tight overshoot (0–0.019) E +0.09 PF 1.14;
widest (0.088+) E +0.60 PF 2.42. ATR is **non-monotonic** (Q4 best, Q1 and Q5
worse), so a p60 ceiling cuts the good band.

H2 was the wrong direction. The flipped rule (overshoot >= train p80) is
labeled **post-hoc** — OOS still uses only past years for the cutoff, but we
looked at quintiles first. Stress with drop-top-N before any promote.

## Gate

- **H5** is the only pre-registered simple filter that lifts E and PF.
- **Logistic keep** beats H5 on E/PF but is not a nightly hard gate (15m stays
  research; lagged logistic was previously a mild lift, not a promote).
- Do **not** promote tight-break, RS floor, ATR ceiling, or quiet-day.
- Do **not** use full-day name counts on 15m.

## Artifacts

- `scripts/research/mine_15m_unique_filters.py`
- `reports/ascending_channels/channel_touch_15m_unique_filter_hypotheses_20260829_235607.csv`
- `reports/ascending_channels/channel_touch_15m_unique_filter_spearman_20260829_235607.csv`
- `reports/ascending_channels/channel_touch_15m_unique_filter_quintiles_20260829_235607.csv`
- `reports/ascending_channels/channel_touch_15m_unique_filter_logistic_coef_20260829_235607.csv`
