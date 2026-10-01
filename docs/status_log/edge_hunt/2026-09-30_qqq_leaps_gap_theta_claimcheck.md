# QQQ LEAPS gap — ThetaData listed claim-check (2026-09-30)

**Source:** Options With Ravish — YouTube [dY2ralsWdQ8](https://www.youtube.com/watch?v=dY2ralsWdQ8) (2025-07-05).

**Verdict: no promote** — but the **headline 96% WR / ~$180k does reprint** on Theta listed marks when uncapped (mid/mid and nearly ask/bid). Cap-3 (his live risk rule) and risk-adjusted gates still fail promote.

Prior BS method check: [2026-09-30_qqq_leaps_gap_claimcheck.md](2026-09-30_qqq_leaps_gap_claimcheck.md) (BS undershot WR; ATR-IV was rich vs mid on gaps).

## Claimed

| Book | n | WR | PnL | PF | Avg d | MDD |
|------|--:|---:|----:|---:|------:|----:|
| Strat1 gap≥1%+SMA100 | 53 | 96% | $88,400 | 10.48 | 102 | $9,330 |
| Strat2 gap≥2% | 27 | 96% | $91,000 | 15 | — | $6,000 |
| Combined | — | 96% | ~$180k | — | — | — |

## Ours (Theta listed)

- Window: **2020-01-03 → 2025-06-05**
- Signals: open gap; strat1 also prior close > SMA100
- Contract: Jan/Mar/Jun/Sep ~365 DTE; strike ≈ BS 60δ among listed with RTH NBBO
- Entry: first nonzero 1m NBBO after 09:30 ET
- Path: Theta `history/eod` bid/ask (≤365d)
- Wall: ~17m mid books + ~8m ask/bid cap + ~13m ask/bid uncapped

Harness: `utils/research/qqq_leaps_gap_theta.py`, `scripts/research/claimcheck_qqq_leaps_gap_theta.py`  
Artifacts: `reports/research/qqq_leaps_gap/2026-09-30_theta*`  
HTML (current_best style): [`reports/research/qqq_leaps_gap/current_best/qqq_leaps_gap_theta.html`](../../reports/research/qqq_leaps_gap/current_best/qqq_leaps_gap_theta.html) — generator `scripts/research/generate_qqq_leaps_gap_report.py`  
Defaults: **$100k** account, **max-open 3**, cash-debit check, **PF** = gross win$/gross loss$, equity vs **SPY buy&hold**. On mid/mid uncapped pool + those defaults: kept **n=40** net **+$50k** (PF 4.78) vs SPY B&H **+$85k** on the same window — underperforms buy&hold; still **no promote**.

### A) mid/mid uncapped (closest to OptionsKit mid)

| Book | n | WR% | PnL | PF | Avg d | MDD | TP% | vs claim |
|------|--:|----:|----:|---:|------:|----:|----:|----------|
| Strat1 | **66** | **97.0** | **$99.7k** | 14.8 | 106 | $7.3k | 97 | WR/PnL match+; n high |
| Strat2 | **31** | **96.8** | **$78.7k** | 13.9 | 62 | $6.1k | 97 | WR match; PnL a bit light |
| Combined | **97** | **96.9** | **$178k** | 14.4 | 92 | $13.3k | 97 | **~$180k / 96% reprints** |

Losses: strat1 **2**, strat2 **1** — same shape as the video. Avg win $1.7k / avg loss $3.6k (fat left tail).

Year buckets (combined): 2020-21 WR 96 PF 10.8; 2022-23 WR 95.5 PF 7.8; **2022** WR 91 PF 4.4; 2024-26 WR 100.

### B) mid/mid max_open=3 (video risk note)

| Book | n | WR% | PnL | PF | MDD |
|------|--:|----:|----:|---:|----:|
| Strat1 | 39 | 94.9 | $54.1k | 8.5 | $7.3k |
| Strat2 | 20 | 95.0 | $53.0k | 9.7 | $6.1k |
| Combined | 59 | 94.9 | **$107k** | 9.0 | $13.3k |

Occupancy skips ~40% of strat1 signals — combined PnL far below $180k.

### C) ask/bid uncapped (marketable long)

| Book | n | WR% | PnL | PF | Avg d | MDD | TP% |
|------|--:|----:|----:|---:|------:|----:|----:|
| Strat1 | 66 | **95.5** | **$99.1k** | 12.2 | 120 | $7.4k | 95.5 |
| Strat2 | 31 | **96.8** | **$84.4k** | 14.6 | 73 | $6.2k | 96.8 |
| Combined | 97 | **95.9** | **$184k** | 13.1 | 105 | $13.7k | 95.9 |

Still claim-like on this bull window even paying ask / exiting bid (TP requires bid ≥ 1.5× entry ask).

### D) ask/bid max_open=3

| Book | n | WR% | PnL | PF | MDD |
|------|--:|----:|----:|---:|----:|
| Strat1 | 36 | 91.7 | $45.2k | 6.1 | $7.4k |
| Strat2 | 19 | 94.7 | $52.6k | 9.5 | $6.2k |
| Combined | 55 | 92.7 | **$97.8k** | 7.5 | $13.7k |

## Interpretation

1. **The OptionsKit slide is real on listed mids** for uncapped QQQ LEAPS gap-buys in 2020–mid-2025. Theta mid/mid combined **n=97 WR 96.9 PnL $178k** is a clean method match (more signals than their 53+27, same loss count shape).
2. **Marketable ask/bid uncapped still ~96% / ~$184k** on this sample — spreads did not kill the path while QQQ kept recovering into TP50.
3. **His own max-3 rule cuts the story in half** (~$98–107k, WR ~93–95%). The $180k headline is an uncapped concurrent book.
4. **Fat left tail remains:** avg loss ≈ 2× avg win; three full-debit-ish losers still matter. No stop by design.
5. **Not a Sharpe>1 promote.** Unlimited slots on one name’s dips is not risk-adjusted edge vs QQQ; occupancy equity / MDD vs buy-hold not walked. Bull-regime sleeve (video admits 2022 pain; SMA100 only softens it).
6. **BS claim-check was the wrong pessimism direction on WR** (ATR-IV rich vs mid → fewer TPs). Listed marks are the honest reprint; they confirm the marketing path, not a portfolio edge.

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_qqq_leaps_gap_theta.py --mark-modes mid_mid
python scripts\research\claimcheck_qqq_leaps_gap_theta.py --mark-modes ask_bid --no-max-open --outdir reports\research\qqq_leaps_gap\2026-09-30_theta_askbid_uncapped
pytest tests\unit\test_qqq_leaps_gap_theta.py -v
python scripts\research\generate_qqq_leaps_gap_report.py
```
