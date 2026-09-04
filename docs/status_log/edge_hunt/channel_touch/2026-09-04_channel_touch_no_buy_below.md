# No buy below the channel (re-entry or resist-break)

GOOGL 15m CTF (`fill` 2025-11-28 15:00 UTC, `enp` 318.325) printed a BUY **below** support (~321.83). Cause: L3 from-above tag was valid, then `--realistic-fill` took the **next 15m mid**, which can sit well under the rail. `_limit_fill_at_support` also clipped to `bar_high` even when that high never reached support. `require_in_channel` only capped `channel_pos <= 1` (above resist), not `channel_pos >= 0`.

## Fill rule (research backtester)

In `scripts/research/backtest_channel_touch_trades.py`:

- `_limit_fill_at_support` returns `None` if the bar never trades at/above the rail.
- If the would-be fill is still below support, `_ensure_fill_not_below_support` walks forward (within `h2 + max_l3_wait_bars`) for:
  1. **Re-entry** — prior close below support, this bar trades back through support (fill at support + slip).
  2. **Breakout** — close above resistance (fill at resist + slip). Mark `resist_break=True`.
- `require_in_channel` now requires `0 <= channel_pos <= 1`, **except** `resist_break` trades (those sit above the rail by construction).

Do **not** copy onto `current_best/`. Nightly / 15m live remains H5 close-above-resist (already a breakout). Wait-12 15m L3 keeper was **not** rescanned with this fill rule.

## Wait-1 A/B (same 300 IB 15m panel, RS top1, friction 0.10)

Same stack as [wait-1](2026-09-04_channel_touch_15m_l3_wait1.md): `--preset 15m --n-symbols 300 --entry-mode l3_touch --min-l3-wait-bars 1 --realistic-fill`. Cache hit; scan 470s (`_raw_20260904_133623`).

| Book | n | WR% | median% | E% | PF | 2018-19 | pos&lt;0 |
|------|---|-----|---------|-----|-----|---------|----------|
| Wait-1 realistic (below-channel fills) | 1770 | 30.3 | -0.77 | **+0.02** | **1.03** | E −0.13 PF 0.80 | **436** |
| Wait-1, no below, breakouts **stripped** by old in-channel cap | 1771 | 29.9 | -0.75 | +0.02 | 1.03 | E −0.10 PF 0.85 | 0 |
| **Wait-1, no below, resist-break kept** | **1771** | 30.4 | -0.74 | **+0.04** | **1.06** | E −0.09 PF 0.85 | **0** (45 breakouts) |

Year net E (kept-breakout book): 2018-19 −0.09, 2020-21 +0.07, 2022-23 +0.11, 2024-26 +0.02.

Raw: 114211 → 105445; `pos<0` 26200 → 0; `resist_break=2684`.

**GOOGL 2025-11-28 15:00 buy 318.325 (`channel_pos=-0.149`) is gone.** Nov 2025 RS-top1 GOOGL leftover is 2025-11-18 15:15 at 281.625 (`channel_pos=0.177`).

## Verdict

Mild lift vs the below-channel wait-1 book (E +0.02→+0.04, PF 1.03→1.06). 2018-19 still fails. **Keep wait-12** (`n=1746 E +0.09 PF 1.16` on realistic fill **before** this rule). Rescan wait-12 with no-buy-below before treating that keeper as updated.

Trades: `reports/ascending_channels/2026-09-04/channel_touch_15m_trades_20260904_133623.csv` / `_raw_` / `_summary_`.
HTML: `reports/ascending_channels/2026-09-04/channel_touch_tv_report_interactive_rs_top1_default_fric0.10_15m_l3_wait1_reentry_realistic_20260904_134155.html` (quality-filtered embed 56352; UI default RS top1).
