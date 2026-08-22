# Phase 1 — TradingView verification of Weekly BigVol + TTM examples

Date: 2026-08-21

Source catalog: `reports/examples/weekly_bigvol_full_setups_20260821_185113.csv`  
Screenshots: `C:\Users\Hezi\tradingview-mcp\screenshots\phase1_*.png`  
TV setup: LazyBear `Squeeze Momentum Indicator`, HV Volume, weekly (1W)

## Verdict

**Examples are real.** Several catalog setups show the intended weekly pattern on TradingView (volume context + TTM flip + subsequent expansion). Proceed to edge validation (Phase 2) is justified, with caveats below.

## Checklist scores (unique confirm weeks)

| Symbol | DB ignition | DB confirm | TV pattern | Timing vs LazyBear | Notes |
|--------|-------------|------------|------------|--------------------|-------|
| UBER | 2023-11-03 | 2023-11-10 | MATCH | Tight | Clean red→green late Oct/early Nov 2023, then strong rally |
| MSFT | 2023-05-26 | 2023-11-24 | MATCH | Tight | Flip ~Nov 2023 aligns with DB confirm; breakout from ~340–350 |
| NVDA | 2023-06-16 | 2023-12-15 | MATCH | ~1–2w skew | Late-2023 squeeze dots → green; big 2024 move. DB close ~489 is pre-split |
| WMT | 2023-10-06 | 2024-02-09 | MATCH | Tight | Flip late Feb 2024 then rally. DB fwd_return **-64% is WRONG** (3:1 split artifact) |
| CRM | 2023-12-01 | 2023-12-08 | PARTIAL | TV ~4w earlier | TV flip ~early Nov; DB confirm Dec 8. Pattern OK, date skew |
| QCOM | 2023-11-03 | 2023-12-01 | PARTIAL | TV ~4w earlier | TV flip ~late Oct; big vol on breakout. Pattern OK |
| AMZN | 2024-05-03 | 2024-10-04 | PARTIAL | TV ~8w earlier | TV flip ~Jul/Aug; DB Oct cross is late/weak (mom≈0.13) |
| META | 2024-02-02 | 2024-07-05 | UNCLEAR | Later signal dominates | Chart window shows strong later squeeze release; July confirm harder to isolate |
| INTC | 2025-02-14 | 2025-03-21 | PARTIAL | Mixed | Feb volume spike visible; post-confirm path was poor (as catalog showed) |

## Findings

1. **Concept validated visually** — UBER, MSFT, NVDA, WMT are clear “webinar-style” weekly setups on LazyBear SQZMOM.
2. **Date skew is common** — DB W-FRI resample + our squeeze-on heuristic often place the zero-cross **1–8 weeks after** the first visual LazyBear flip on TV. Still same regime, but entries would differ.
3. **Corporate actions corrupt naive fwd returns** — WMT Feb 2024 split made catalog `fwd_4w/13w` look catastrophic while TV shows a rally. NVDA also has split-adjusted vs raw close mismatch. Any Phase 2 scoring must use split-adjusted series or drop returns around known splits.
4. **Losers exist** — INTC is a reminder that a valid-looking confirm can still fail; do not cherry-pick only winners.
5. **Not verified exhaustively** — Did not re-check every duplicated confirm row (e.g. multiple ignitions → same META/AMZN/WMT confirm).

## Exit criteria (from plan)

- Target ~5–10 TV-confirmed full setups → **met** (at least 4 strong MATCH + several PARTIAL).
- Continue to backtest: **yes**, after fixing split-adjusted return handling for metrics.

## Recommended next step

Phase 2 universe backtest / metrics, with:
- split awareness in any forward-return or equity stats
- awareness that DB confirm dates may lag TV LazyBear by a few weeks
