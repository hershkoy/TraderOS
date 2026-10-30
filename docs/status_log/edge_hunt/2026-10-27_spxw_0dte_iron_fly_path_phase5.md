# SPXW 0DTE short iron fly — path MTM + Phase-5 vs SPY (2026-10-27)

**Verdict: soft-promote on natural settle (Phase-5 PASS) for this ~2y sample. Not wired** — need live/paper BAG fill log during RTH. Path MAE is acceptable (not soft-fail).

**Full-history expand (2022-05-11→now) fails Phase-5 on MDD** — see [2026-10-27_spxw_0dte_iron_fly_from_20220511.md](2026-10-27_spxw_0dte_iron_fly_from_20220511.md). Soft-promote stays scoped to the 2y window below.

Parent claim-check: [2026-10-27_spxw_0dte_iron_fly_claimcheck.md](2026-10-27_spxw_0dte_iron_fly_claimcheck.md).

## What was added

1. **1m package MTM** (13:45→16:00) for all four legs — MAE/MFE, % underwater > half credit, optional 50% TP overlay.
2. **Single-slot $25k equity vs SPY** via `passes_phase5_gates` (one trade/day → flat sessions return 0).
3. **Entry package slip** (mid − natural credit) and exit mid−natural haircut at TP/last bar.
4. **IB paper BAG profiler** (client **8831**) — dry-run qualified today’s SPXW legs; `--place` not run (pre-RTH).

Harness: `utils/research/spxw_0dte_iron_fly_path.py`, `scripts/research/overlay_spxw_0dte_iron_fly_path.py`, `scripts/research/paper_spxw_iron_fly_combo.py`  
Artifacts: `reports/research/spxw_0dte_iron_fly/2026-10-27/path_phase5/`  
HTML (how-to + results): `reports/research/spxw_0dte_iron_fly/2026-10-27/spxw_0dte_iron_fly_report.html` (`scripts/research/generate_spxw_0dte_iron_fly_report.py`)  
Wall: **64 s** (path cache warm after first fill).

## Path risk (natural entry, mid MTM mark)

| Metric | Natural n=311 |
|--------|--------------:|
| MAE median | **−$180** |
| MFE median | +$255 |
| Underwater >½ credit | 21.2% of trades |
| Winners that were underwater >½ | **12.4%** (<25% soft-fail) |
| Hit 50% TP before settle | 28.0% |
| Settle E / PF | +$93 / 1.45 |
| 50% TP overlay E / PF | +$56 / 1.28 (weaker — do not prefer) |

Settle-only remains the better book. Early 50% TP clips winners more than it helps.

## Package slip

| | Median | p90 |
|--|-------:|----:|
| Entry mid−natural credit | **$20** | $28 |
| Exit mid−natural (at TP/last) | **$20** | — |

Natural entry already embeds ~$20/lot vs mid. That book still clears Phase-5.

## Phase-5 vs SPY ($25k, 1 lot, natural settle)

| | Iron fly natural | SPY buy-hold |
|--|-----------------:|-------------:|
| CAGR | ~45% (compound path) / equity stats **~** | **15.1%** |
| Sharpe | **2.20** | 0.93 |
| Max DD | **14.2%** | 19.0% |
| Gate | **PASS** (Sharpe>1 + Sharpe≥SPY + MDD) | benchmark |

Mid settle also PASSes (Sharpe 2.79, MDD 12.9%). Ranking book stays **natural**.

## Live fill path

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\paper_spxw_iron_fly_combo.py
python scripts\research\paper_spxw_iron_fly_combo.py --place --wait-sec 120
```

2026-10-27 ~07:22 ET dry-run: Gateway :4001 connected, Yahoo ATM selection, **BAG legs qualified**, IB package mid credit **$12.70**. Status `dry_run_ok`. **No `--place`** (outside RTH / no fill evidence yet). Latest: `reports/research/spxw_0dte_iron_fly/paper_fills/latest.json`.

## Promote / wire

| Gate | Result |
|------|--------|
| Claim replicate | Yes (prior) |
| Phase-5 natural settle | **PASS** |
| Path MAE soft-fail (≥25% winners deep red) | No (12.4%) |
| Live BAG fill log | **Missing** |
| **Wire** | **No** |
| **Soft-promote** | **Yes** (research sleeve candidate) |

## Reproduce

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\overlay_spxw_0dte_iron_fly_path.py --fill both --workers 4
pytest tests\unit\test_spxw_0dte_iron_fly_path.py -v
```
