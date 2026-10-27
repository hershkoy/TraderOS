# Channel-touch L2 bounce claim-check (2026-10-10)

## Why

L3 (`l3_touch` after H2) and L4 (`--entry-touch 4`) were already scanned; L2 was never a buy path because `entry_touch` is clamped to ≥3 post-H2, and L2 is the formation low that *defines* the rail. Hypothesis: buy the causal L2 bounce (L1–H1–L2) at L2+pivot_len close, without waiting for H2.

## Rules

- Detector: new `find_l2_bounce_setups` / `find_l2_bounce_setups_windowed` (v1 `find_channels` / `find_h2_l3_setups` unchanged).
- Support on two higher lows (L1, L2) with intervening rally; provisional width from peak intervening high (H1).
- Arm when L2 is a confirmed swing low (`confirm = L2 + pivot_len`). Abort if any close between L2 and confirm breaks support.
- Entry: confirm-bar close (`--entry-mode l2_touch`). No H2 required (requiring H2 before filling L2 would look ahead).
- CLI: `backtest_channel_touch_trades.py --entry-mode l2_touch`.

## Run

```bat
venv\Scripts\activate && set PYTHONPATH=. && python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --entry-mode l2_touch --workers 4 --load-workers 8
```

Wall-clock ~500s (cache-hit load 2.6s, scan 235s, RS 117s). Raw 217094 → keeper **n=1778** (RS top1 / in-channel / span≤10 / friction 0.10).

## Result vs L3 / L4 (same 15m 300 panel)

| stack | n | E% (fric0.10) | PF | notes |
|-------|---|---------------|-----|-------|
| **L2 bounce (this)** | **1778** | **-0.05** | **0.93** | confirm close; no H2 |
| L3 wait-12 beyond-off (2026-08-29) | 1751 | +0.32 | 1.58 | post-H2 support tag |
| L4 entry-touch 4 (2026-08-29) | 1312 | +0.20 | 1.36 | fewer than L3 |
| 15m pivot prior | 1764 | +0.15 | 1.24 | — |

Year buckets (net / fric0.10): 2018-19 E −0.04 PF 0.94; 2020-21 E −0.08 PF 0.90; 2022-23 E +0.06 PF 1.08; 2024-26 E −0.15 PF 0.81. Gross (pre-friction) is only E +0.05 PF 1.07 — still far below L3.

Beyond-width A/B: off / 0.25 both ~E −0.04..−0.05 PF ~0.93–0.95 — no salvage.

## Verdict

**No promote.** Causal L2 bounce loses to L3 and to L4 on the same panel. Keep L3 wait-12 as the 15m support-touch book; do not fish L2 confirm timing or provisional-width variants as a rescue.

## Code / tests

- `scripts/research/find_ascending_channels.py`: `find_l2_bounce_setups`, `_dedupe_l2_bounce_setups`, windowed wrapper
- `scripts/research/backtest_channel_touch_trades.py`: `--entry-mode l2_touch`
- `tests/unit/test_l2_touch_entry.py` (3 passed)

## Artifacts

- `reports/ascending_channels/2026-10-10/channel_touch_15m_trades_20261010_125334.csv`
- `reports/ascending_channels/2026-10-10/channel_touch_15m_trades_raw_20261010_125334.csv`
- `reports/ascending_channels/2026-10-10/channel_touch_15m_trades_summary_20261010_125334.txt`
- `reports/ascending_channels/2026-10-10/channel_touch_15m_beyond_width_ab_20261010_125334.csv`
