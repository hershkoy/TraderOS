# Causal Deepvue weekend scan (2026-09-12)

Research translation of [weekly_deepvue.md](../../strategies/weekly_deepvue.md) (Moglen / Deepvue weekend routine). **Not Deepvue.** Not a promote.

Runner: `scripts/research/backtest_weekly_deepvue.py`. Logic: `utils/research/weekly_deepvue.py`.

## Clock (realistic purchasing)

Weekend signal uses **completed daily** bars through the week's last session (W-FRI; Thursday if Friday is missing). That is known after the cash close.

- **Buy:** next RTH session **09:30 ET 15m mid** (`purchase_next_session_open_mid`). Live after Friday. Not a same-session lookback.
- **Sell:** 3% hard stop on **15m bar N**, fill **N+1 mid** (overnight OK). Or **two daily closes below the 21 EMA**, fill the next 15m mid. Entry 15m is skipped for the stop. Not the daily occupancy clip.

QQQ gate is Friday close **above a rising 21 EMA** (aggressive stance only). Tightness is 5d/15d range ratio (RMV proxy). RS is 21d and 63d cross-section percentile >= 85 among liquid names that day. Sector cap 2, weekly focus 15, take 4, occupancy max 4 unique symbols.

## Sample

Top 300 IB 15m names by ALPACA 1d ADV. Daily ALPACA + IB prefix. 2018-11-01 → 2026-08-21. Wall **318s** (list 120s, daily 39s, 15m 69s, fills 9s).

Weekend focus rows 204. All filled. Occupancy kept **179** (93 symbols). QQQ risk-on 1312 / 1959 panel days.

## Results

| Book | n | E | PF | geo | WR | median |
|------|---|---|----|-----|----|--------|
| Gross | 179 | **+2.25** | **1.85** | **0.985** | 15.1% | **−2.94** |
| Fric 0.25 | 179 | +2.00 | 1.70 | 0.886 | 14.5% | — |

Year buckets (gross):

| Bucket | n | E | PF | WR |
|--------|---|---|----|----|
| 2018-2019 | 12 | −2.01 | 0.17 | 25% |
| 2020-2021 | 57 | +4.57 | 2.76 | 18% |
| **2022-2023** | 32 | **−2.16** | **0.17** | 13% |
| 2024-2026 | 78 | +3.02 | 2.09 | 13% |

Exits: **hard_stop 149 / 179** (83%), ma_2close 30. Avg win +32.6% vs avg loss −3.13% (the 3% stop). Fat tails: mean +2.25 while median is the stop. A few MA-trail names (AFRM 2021-08-23 +118%, APP 2024-10-21 +108%, MSTR 2024-02-26 +105%) carry pooled PF. HOOD 2024-10-28 hard-stop **−11.6%** is a 15m next-mid after a gap through 3% (honest, not a same-bar clip).

Geo uses YEAR_BUCKETS with n>=30 (skips 2018-19). 2022-23 PF 0.17 pulls geo under 1.0.

## Verdict

**No promote.** Pooled E/PF is not the gate: geo-PF < 1, 2022-23 is a hard-stop slaughter, n=179 is a 300-name ADV slice not the full IB 15m set, and the median trade is negative. Do not grade this against clip books or 1d signal-close lookbacks. Number to beat for a *different* 1d live book remains last-15m **n=2297 E +0.27 PF 1.09**. This is not an overlay on that book.

Do not occupancy-walk a 2%/day variant or full-universe rescan until a causal change actually lifts 2022-23 without clipping 2020-21 / 2024-26 winner dollars.

AND with last-15m H2 (2026-09-13): trade-week overlap **n=1** (MRNA). Ever-on-list last-15m n=161 wrecks 2020-21. **No promote.** See [Deepvue AND](channel_touch/2026-09-13_last_15m_deepvue_and.md).

HTML (TV-style, no rails / Channel column empty; default friction 0.25): `reports/weekly_deepvue/2026-09-12/weekly_deepvue.html`.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\backtest_weekly_deepvue.py --n-symbols 300
python scripts\research\generate_weekly_deepvue_report.py
```
