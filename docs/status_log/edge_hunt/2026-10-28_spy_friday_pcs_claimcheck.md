# SPY Friday 7 DTE put credit spread — claim-check (2026-10-28)

Video: Nathan / optionsforall.com [Best Put Credit Spread](https://youtu.be/HbV4sTEyoBM) (`HbV4sTEyoBM`).

**Verdict:** mid **TP50 / SL300%** roughly replicates the claim and **passes Phase-5 vs SPY** on peak-margin×1.5 capital — treat as a **soft-promote research sleeve**, not a live wire. Mid **TP25** clears Sharpe>1 but **fails MDD vs SPY** by a hair (34.5% vs 34.1%). **Natural ask/bid collapses** (Sharpe 0.35, MDD 68.6%) — do not promote fillable P&L from mid alone.

## Rules tested

| Rule | Implementation |
|------|----------------|
| Underlying | SPY |
| Entry | Fridays (next session if Friday closed) |
| DTE | 7 ± 2 (Friday→Friday weekly) |
| Structure | Short ~40Δ put, long $5 lower |
| Fill | Mid (claim) and natural ask/bid stress |
| TP | 25% or 50% of credit (debit ≤ 0.75× / 0.50× credit) |
| SL | 300% of credit lost (debit ≥ 4× credit); also nostop |
| Commission | $0 (per video) |
| Capital | Peak defined risk `(width−credit)×100` × 1.5 buffer (video auto-size) |
| Spot | IB SPY 1d (Alpaca+IB prefix has a **2019 hole** for SPY — do not use `load_spy_daily` here) |
| Options | OptionsDX EOD through 2023-12-29; Theta EOD from 2024-01-02 |

Harness: `utils/research/spy_friday_pcs.py`, `scripts/research/claimcheck_spy_friday_pcs.py`  
Artifacts: `reports/research/spy_friday_pcs/2026-10-28/`  
Wall clock: **54.8 s** (Theta chain/path cache warm; first full run ~496 s)

EOD marks only. An intraday TP/SL touch that reverts before the close is invisible.

## Claim vs local (2016-01-01 → 2025-12-31)

| Book | n | WR% | CAGR | MDD | Sharpe | RoC% | Phase-5 |
|------|--:|----:|-----:|----:|-------:|-----:|---------|
| Claim TP25 / SL300 | 519 | 85 | 23.8% | 52.5% | 0.74 | ~748 | — |
| **Local mid TP25** | **486** | **86.2** | **28.0%** | **34.5%** | **1.15** | **1078** | **FAIL MDD≥SPY** |
| Claim TP50 / SL300 | ~519 | 80.3 | 23% | 30.7% | 0.73 | ~691 | — |
| **Local mid TP50** | **486** | **82.3** | **29.1%** | **30.4%** | **1.22** | **1185** | **PASS** |
| Local mid TP25 nostop | 486 | 86.2 | 27.5% | 40.4% | 1.07 | 1036 | FAIL MDD |
| Local natural TP25 | 486 | 85.0 | 13.7% | 68.6% | 0.35 | 259 | FAIL |
| Local natural TP50 | 486 | 81.5 | 17.2% | 56.3% | 0.51 | 388 | FAIL |
| Mid TP50 slip $0.05 | 486 | 81.7 | 27.1% | 31.4% | 1.06 | 997 | **PASS** (2022 MDD fail) |
| Mid TP50 slip $0.10 | 486 | 81.7 | 26.4% | 32.0% | 1.01 | 941 | **PASS** (2022 MDD fail) |
| SPY buy-hold (same $) | — | — | 13.0% | 34.1% | 0.72 | — | BENCHMARK |

### TP50 friction ladder (2026-10-28 follow-up)

Package slip is mid ± slip **clamped to NBBO** (never worse than natural). Suite: `--variants tp50_slip` → `reports/research/spy_friday_pcs/2026-10-28_tp50_slip/` (wall 46.6 s).

| Mark | n | WR% | CAGR | MDD | Sharpe | SL exits | Phase-5 FULL | 2022 |
|------|--:|----:|-----:|----:|-------:|---------:|:------------:|:----:|
| Mid $0 | 486 | 82.3 | 29.1% | 30.4% | 1.22 | 26 | PASS | PASS |
| Mid ±$0.05 | 486 | 81.7 | 27.1% | 31.4% | 1.06 | 30 | PASS | FAIL MDD |
| Mid ±$0.10 | 486 | 81.7 | 26.4% | 32.0% | 1.01 | 33 | PASS | FAIL MDD |
| Natural | 486 | 81.5 | 17.2% | 56.3% | 0.51 | 52 | FAIL | FAIL |

A nickel/dime of package slip (limit-fill style) keeps the FULL Phase-5 PASS. Natural still kills it. 2022 is the fragile year once any slip is on.

Capital on the mid books ≈ **$697** (peak ~$5-wide defined risk × 1.5). That matches the video’s tiny margin base and inflates RoC/CAGR vs a fixed $10k account.

Skips: credit_below_min 13, no_wing 11, no_chain 9, max_concurrent 3. n=486 vs claim 519 is mostly those skips plus a few weeks without a 5–9 DTE listed put wing.

### Year split — mid TP50 (Phase-5 PASS book)

| Year | n | WR% | mean RoR% | PF | PnL |
|------|--:|----:|----------:|---:|----:|
| 2016 | 49 | 87.8 | +10.04 | 3.62 | +$1,994 |
| 2017 | 44 | 93.2 | +8.55 | 8.54 | +$1,639 |
| 2018 | 44 | 75.0 | −3.00 | 0.79 | −$602 |
| 2019 | 48 | 89.6 | +10.98 | 3.64 | +$2,092 |
| 2020 | 49 | 81.6 | +2.59 | 1.14 | +$418 |
| 2021 | 51 | 80.4 | +1.58 | 1.07 | +$230 |
| 2022 | 47 | 74.5 | −3.51 | 0.86 | −$623 |
| 2023 | 51 | 78.4 | +1.93 | 1.08 | +$295 |
| 2024 | 51 | 82.4 | +10.87 | 1.85 | +$1,999 |
| 2025 | 52 | 80.8 | +4.38 | 1.24 | +$818 |

2022 calendar window: TP50 **PASS** under the Sharpe>1 waiver (Sharpe ≥ SPY, MDD better than SPY).

## What this says

1. **Claim shape is real on mid.** Friday 7 DTE, 40Δ / $5, TP50 / SL300% lands near the video’s WR/MDD and beats the stated CAGR on our OptionsDX+Theta book. TP25 is close on WR and richer on CAGR but loses the MDD race to SPY.
2. **Friction.** Mid ±$0.05 / ±$0.10 (NBBO-clamped) still **PASS** FULL Phase-5 on TP50 (Sharpe 1.06 / 1.01). Natural ask/bid still **FAIL**. 2022 fails MDD once slip is on. Live design that matches the slip books: combo credit limit near mid−nickel, cancel if not filled — not a marketable cross.
3. **Sizing.** Peak-margin×1.5 is the video’s method. Absolute dollar P&L on 1-lot is modest (~$8.3k over 10y for TP50). Percentage metrics assume that ~$700 stays the risk budget.
4. **Data footgun.** `load_spy_daily` (Alpaca + IB prefix) omitted all of 2019 for SPY here (Alpaca primary after first Alpaca bar is sparse). Claim-check uses IB-only SPY via `load_spy_daily_ib`.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_spy_friday_pcs.py --variants claim --outdir reports\research\spy_friday_pcs\2026-10-28
```
