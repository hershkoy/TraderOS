# SPXW 0DTE short iron fly — expand 2022-05-11 → now (2026-10-27)

**Verdict: no soft-promote on the full window.** Natural settle **fails Phase-5** because MDD ≥ SPY. The prior ~2y sample (2024-10→2026-10) still PASSes and remains the soft-promote reference — see [path_phase5](2026-10-27_spxw_0dte_iron_fly_path_phase5.md). Not wired.

## What changed

1. Re-ran claim-check with `--start 2022-05-11 --end 2026-10-27` (first true daily SPXW).
2. Fixed `load_index_eod`: Theta FREE/STANDARD index EOD often starts **2024-01-02**; a partial Theta frame previously short-circuited Yahoo and silently clipped the book to 2024+. Now falls back to Yahoo `^GSPC` / `^VIX` when Theta's first date is >7d after the requested start.
3. Path MTM + Phase-5 overlay on the expanded trade list.

Artifacts: `reports/research/spxw_0dte_iron_fly/2026-10-27_from_20220511/`  
HTML: `.../spxw_0dte_iron_fly_report.html`  
Wall: claim-check **188 s**, path **47 s** (chain/path cache warm for 2024+).

## Coverage

| | |
|--|--:|
| Eligible (Mon/Tue/Thu/Fri, VIX open <20, SPX settle) | **619** |
| Filled (natural / mid) | **611** |
| First fill | **2022-08-11** (almost no 2022: VIX≥20 most days after May) |
| Year counts (natural) | 2022: 1 · 2023: 159 · 2024: 177 · 2025: 142 · 2026: 132 |

## Natural settle book

| Metric | Full window | Prior 2y (soft-promote) |
|--------|------------:|------------------------:|
| n | **611** | 311 |
| WR | 57.8% | 57.2% |
| E / trade | **+$88.8** | +$93 |
| PF | 1.43 | 1.45 |
| CAGR (1 lot, $25k) | 31.5% | 45.4% |
| Equity MDD | **26.7%** | 14.2% |

Shape still matches the claim (WR ~57%, avgW/L ~$511 / $489). Longer sample adds 2023 and one 2022 day; expectancy stays positive but drawdown deepens.

## Path risk (natural entry, mid MTM)

| Metric | Full n=611 |
|--------|----------:|
| MAE median | **−$168** |
| Underwater >½ credit | 21.6% |
| Winners underwater >½ | **9.9%** (<25% soft-fail) |
| Hit 50% TP before settle | 29.8% |
| Settle E / PF | +$89 / 1.43 |
| 50% TP overlay E | +$61 (weaker) |

Path MAE soft-fail does **not** trip. Gate fail is MDD vs SPY, not path.

## Phase-5 vs SPY ($25k, 1 lot, natural settle)

| | Iron fly natural | SPY buy-hold |
|--|-----------------:|-------------:|
| Window | 2022-08-11 → 2026-10-26 | same |
| Sharpe | **1.78** | 0.96 |
| Max DD | **26.7%** | **19.0%** |
| Gate | **FAIL (MDD≥SPY)** | benchmark |

Mid settle also FAILs MDD (20.0% vs 19.0%) despite Sharpe 2.24.

## Promote / wire

| Gate | Full window | Prior 2y |
|------|-------------|----------|
| Claim replicate | Yes (shape) | Yes |
| Phase-5 natural settle | **FAIL MDD** | **PASS** |
| Path MAE soft-fail | No | No |
| Live BAG fill log | Missing | Missing |
| Soft-promote | **No** | **Yes** |
| Wire | **No** | **No** |

Keep soft-promote scoped to the 2y sample. Full history is research-interesting but not Phase-5 clean.

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_spxw_0dte_iron_fly.py --start 2022-05-11 --end 2026-10-27 --outdir reports\research\spxw_0dte_iron_fly\2026-10-27_from_20220511 --workers 4
python scripts\research\overlay_spxw_0dte_iron_fly_path.py --trades reports\research\spxw_0dte_iron_fly\2026-10-27_from_20220511\trades.csv --outdir reports\research\spxw_0dte_iron_fly\2026-10-27_from_20220511\path_phase5 --workers 4
python scripts\research\generate_spxw_0dte_iron_fly_report.py --indir reports\research\spxw_0dte_iron_fly\2026-10-27_from_20220511 --out reports\research\spxw_0dte_iron_fly\2026-10-27_from_20220511\spxw_0dte_iron_fly_report.html
pytest tests\unit\test_spxw_0dte_iron_fly.py -v
```
