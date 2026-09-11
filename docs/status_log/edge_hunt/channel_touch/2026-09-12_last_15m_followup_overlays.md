# Last-15m follow-up overlays (2026-09-12)

Isolated same-list tests on the LAUR-fixed last-15m-open-mid + 15m N+1 mid book **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01; geo-mean year PF 1.095). Parent: [volume geometry](2026-09-11_last_15m_volume_geometry.md). Occupancy not re-walked. **No promote.** Does not beat hot-cross n=4079 E −0.19 PF 0.94 as a live 1d clock. 2022-23 stays negative on every overlay.

Ranked by geometric mean of YEAR_BUCKETS PFs (skip n<30). Winner-$ gate: do not occupancy-walk an overlay that lifts geo PF by clipping fat winners.

## Smoke (AMPL / VST / TARS)

| Name | Fill | F1 doji-only | F2 failed-breakout | F3 session-OHLC seller |
|------|------|--------------|--------------------|------------------------|
| **AMPL** 2024-02-09 | 14.17 → 11.15 (−21.3%) | No doji. Unchanged. | **Used** 2024-02-13; gain **−0.53**. Close back at/below resist before the Feb 21 gap. | **No seller streak.** Daily-bar Volume Delta at 55%×2 still misses AMPL. ATR already sells 09:45 ET. |
| **VST** 2021-02-18 | 23.20 (−19.6%) | No. | **Used** 2021-02-19; gain **−0.78**. | Seller day 2021-02-26 is not earlier than ATR. Unchanged. |
| **TARS** 2023-07-20 | 23.24 (−19.4%) | No. | Fail day 2023-07-25 but trail already out. Unchanged. | Seller day 2023-07-25 not earlier than trail. Unchanged. |

F2 is the only overlay that actually catches the AMPL gap and the VST dump. TARS stays a trail loser.

## F1 — Evening Doji Star only

Drop the 2× seller-sessions leg. Fill next 15m mid if strictly earlier than ATR/trail.

| | n | E | PF | geo year PF | WR | 2022-23 E/PF | used |
|--|---|---|----|-------------|----|--------------|------|
| Baseline | 2297 | +0.27 | 1.086 | 1.095 | 31.4 | −0.90 / 0.73 | — |
| **F1** | 2297 | +0.26 | 1.083 | 1.096 | 31.8 | −0.81 / 0.76 | 70 |

Winner-cut: 2 winners flip; 11 losers flip; winner $ **−129** vs loser $ **+107**. Geo PF is flat. Pass needed geo lift **and** winner $ near 0/positive — **fail**. Original overlay A’s 41 doji *fills* were the subset where doji beat seller; isolated doji uses 70. AMPL still has no textbook star. **No promote.** Occupancy skipped.

## F2 — failed-breakout exit

First post-fill daily close at/below the projected resist rail; sell the next 15m mid if strictly earlier than ATR/trail. Not overlay C (C delayed the *buy*).

| | n | E | PF | geo year PF | WR | 2022-23 E/PF | used |
|--|---|---|----|-------------|----|--------------|------|
| Baseline | 2297 | +0.27 | 1.086 | 1.095 | 31.4 | −0.90 / 0.73 | — |
| **F2** | 2297 | +0.18 | 1.094 | 1.086 | 29.0 | −0.73 / 0.64 | **1451** |

Winner-cut: 232 winners flip; 177 losers flip; winner $ **−3457** vs loser $ **+3256**. Fires on 63% of the book — a second trail, not a rare reversal. Geo PF falls. E falls. AMPL/VST examples work; the sleeve does not. **No promote.** Occupancy skipped.

## F3 — session-OHLC volume-delta (seller only)

Same 2× `sell_pct>=55%` as overlay A, but split once on the day’s OHLC with summed 15m volume (`VOLUME_MODE_SESSION_OHLC`). No doji. Do not loosen 55%.

| | n | E | PF | geo year PF | WR | 2022-23 E/PF | used |
|--|---|---|----|-------------|----|--------------|------|
| Baseline | 2297 | +0.27 | 1.086 | 1.095 | 31.4 | −0.90 / 0.73 | — |
| A (15m-sum seller+doji) | 2297 | +0.46 | 1.192 | 1.298 | 40.6 | −0.27 / 0.90 | 1119 |
| **F3** | 2297 | +0.32 | 1.159 | **1.178** | 41.8 | **−0.14 / 0.94** | **1543** |

Winner-cut: 129 winners flip; 368 losers flip; winner $ **−3484** vs loser $ **+3606**. Geo PF lifts vs baseline (1.178 vs 1.095) but wrecks winner dollars the same way A did; 2022-23 still red. Session OHLC fires *more* often than 15m-sum (1543 vs 1078 seller fills), not less — a close near the session low is a seller day even when 15m bars mixed. AMPL still has no 2× 55% streak before the gap. Do not loosen 55% to fish it. **No promote.** Occupancy skipped (would free slots earlier, but the winner-$ gate fails).

## Occupancy / next

Skipped. None of the three clears geo PF **without** clipping winner dollars. F2 is the only smoke hit on AMPL/VST and it is too blunt to rescan. F1 is noise. F3 is A’s seller-leg with a different volume split — same failure mode.

Wall-clock: IB 15m cache hit 940 symbols **7.1s**, three overlays **~473s**.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_volume_geometry.py --followups --workers 8
```

Artifacts: `reports/ascending_channels/2026-09-12/last_15m_overlay_F1_doji_only.csv`, `last_15m_overlay_F2_failed_breakout.csv`, `last_15m_overlay_F3_seller_session_ohlc.csv`, `last_15m_overlay_followups_summary.txt`.
