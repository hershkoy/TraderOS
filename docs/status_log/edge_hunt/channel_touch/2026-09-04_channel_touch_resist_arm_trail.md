# Peak trail from entry (fixed %%) — corrected exit A/B

Date: 2026-09-04

## Clarification

Earlier "resist-arm" waits for the upper rail were **wrong** vs the intended rule. Correct exit:

1. Entry stop = `entry * (1 - stop_pct)` (e.g. 15.33 → 14.87)
2. Each bar: `peak = max(peak, high)`; stop ratchets to `peak * (1 - trail_pct)`
3. Exit when **low** reaches that stop (fill at stop)

No resist gate. MTSI 2019-07-03 @ 15.33 with 3%/3% exits **2019-07-05** @ 14.938 (peak 15.4) — not a multi-year hold. A later 31.03 high never happens under this rule.

## Implementation

`--resist-arm-trail` now means this peak trail (CLI name kept; help text updated). ATR/squeeze overlays off.

## 300-name panel (L3 wait-12, signal-close, touch-0)

`--stop-pct 0.03 --trail-pct 0.03 --resist-arm-trail`

| Book | n | E | PF | eod | avg hold bars |
|------|---|---|----|-----|---------------|
| Keeper (scaled ATR + ~2% trail) | 1744 | +0.13 | 1.22 | — | — |
| Peak trail 3%/3% | **1728** | **+0.11** | **1.10** | 3 | 74 |

Mostly `resist_arm_trail` exits (1682); almost no eod pathology.

## Verdict

Logic fixed and matches the chart intent. **Do not promote vs keeper** on this panel (E/PF slightly worse). Optional next: retest `--trail-pct 0.0125` with the same from-entry ratchet.

Trades: `reports/ascending_channels/2026-09-04/channel_touch_15m_trades_20260904_190235.csv`
