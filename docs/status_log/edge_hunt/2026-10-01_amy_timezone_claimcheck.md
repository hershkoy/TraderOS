# Amy Meissner TimeZone — ThetaData claim-check (2026-10-01)

**Source:** Aeromir roundtable 2025-10-02 / YouTube [pvRJ7lnubM8](https://www.youtube.com/watch?v=pvRJ7lnubM8); local transcript `temp/amy_messenier_timezone_transcript.txt`.

**Verdict: no promote** — and **cannot claim-check her exact system**. Public materials only disclose structure/cadence/targets; strike selection, term-structure filters, and the adjustment playbook are workshop-gated. A transparent **no-adjust** RUTW proxy on Theta listed marks is **mid-claim-adjacent** on 2025 weekly (PF/return near her slide; WR short) and **killed by marketable ask/bid**.

## Claimed (from transcript)

| Window | Cadence | n | WR | PF | Return on capital | Notes |
|--------|---------|--:|---:|---:|------------------:|-------|
| 2016–2019 | 1 wk/mo | — | 75% | — | — | out of market most of time |
| 2016–2019 | more active | — | 83% | ~10 | — | — |
| 2020 Jan–Aug | — | — | 80% | — | — | crash+recovery |
| Last ~3y | 1 wk/mo 2-lot RUT | 38 | **87%** | **>5** | **~130%** | 154 days in market |
| 2025 Jan–Aug | every week 2-lot | 30 | **80%** | **~4** | **>~100%** | avg profit ~$400/2-lot |

Also: Thursday entry; calendar + credit spread; +5% / −5% capital targets; ~3–5 day holds; ~$7–10k capital for 2-lot RUT; adjustments via alerts (up to ~3).

## Proxy (honest, not her recipe)

Harness: `utils/research/amy_timezone_proxy.py`, `scripts/research/claimcheck_amy_timezone.py`  
Data: ThetaData **RUTW** Friday weeklies EOD bid/ask; spot Yahoo `^RUT`.  
Artifacts: `reports/research/amy_timezone/2026-10-01/`  
Wall: **~327s** for 2025 weekly + 3y monthly × mid + ask_bid.

Rules used:

- Enter each **Thursday** (or first Thursday of month)
- Front = first Friday with DTE in **[5, 12]**; back = next Friday
- **ATM call calendar** (long back / short front)
- **Bull put credit** on front: short ≈3% OTM, long $10 lower (upside-edge credit she described)
- 2 lots; capital **$8,000**; TP/SL **±5%** of capital; SL fill clamped at −$400 when EOD gaps through
- Max hold **5** sessions or day before front expiry
- **No adjustments**

### Results vs claim

| Book | n | WR% | PF | Ret% cap | Avg PnL | vs claim |
|------|--:|----:|---:|---------:|--------:|----------|
| 2025 weekly **mid** | 31 | **67.7** | **3.78** | **+125.1** | $323 | n≈30; PF/ret near; WR <80%; avg≈$400 claim |
| 2025 weekly **ask/bid** | 31 | **6.5** | **0.08** | **−130.1** | −$336 | **fails hard** |
| 3y monthly **mid** | 29 | **69.0** | **3.62** | **+118.1** | $326 | WR <<87%; PF/ret soft-near 5 / 130 |
| 3y monthly **ask/bid** | 29 | **13.8** | **0.27** | **−91.9** | −$253 | **fails hard** |

## Interpretation

1. **Recipe incomplete.** Without her entry geometry and adjustment paths, this is a method check on a public sketch, not a reprint of OptionNet Explorer trades.
2. **Mid/mid no-adjust looks surprisingly claim-shaped on 2025 weekly** (PF ~3.8 vs ~4; +125% vs ~100%; WR 68% vs 80%). That does **not** validate TimeZone — mids on a 4-leg index package are optimistic, and her examples lean on adjustments.
3. **Marketable ask/bid destroys the book.** Paying ask on longs / hitting bid on shorts flips 2025 to WR 6.5% / PF 0.08. Same pattern as other options claim-checks: mid slides ≠ live fills.
4. **3y one-week/month mid** gets ~69% WR / PF 3.6 / +118% — below her 87% / PF>5 headline; still mid-only.
5. **No promote** under edge-hunt gates (listed realism first; occupancy/Sharpe vs SPY not even reached). Do not wire live.

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\claimcheck_amy_timezone.py --window both --marks mid,ask_bid
```
