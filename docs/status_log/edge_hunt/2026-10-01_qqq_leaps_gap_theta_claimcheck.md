# QQQ LEAPS gap — ThetaData listed claim-check (2026-09-30)

**Verdict: no promote (Theta mid can look claim-like on WR; still fat-left-tail / bull-regime; ask/bid and Sharpe>1 gate not cleared)**

**Source:** Options With Ravish — YouTube [dY2ralsWdQ8](https://www.youtube.com/watch?v=dY2ralsWdQ8).
Prior BS method check: [2026-09-30_qqq_leaps_gap_claimcheck.md](2026-09-30_qqq_leaps_gap_claimcheck.md).

## Claimed

| Book | n | WR | PnL | PF |
|------|--:|---:|----:|---:|
| Strat1 | 53 | 96% | $88,400 | 10.48 |
| Strat2 | 27 | 96% | $91,000 | 15 |
| Combined | — | 96% | ~$180k | — |

## Ours (Theta listed)

- Window: `2020-01-03` → `2025-06-05`
- Signals: same mechanical gap rules; SMA=`prior`
- Contract: Jan/Mar/Jun/Sep ~365 DTE; strike ≈ BS 60δ among listed with RTH NBBO
- Entry: first nonzero 1m NBBO after 09:30 ET
- Path: Theta `history/eod` bid/ask (≤365d pulls)
- Max open: `None`
- Wall: 754.8s

Harness: `utils/research/qqq_leaps_gap_theta.py`, `scripts/research/claimcheck_qqq_leaps_gap_theta.py`

### ask_bid_capNone

| Book | n | WR% | PnL | PF | Avg d | MDD | TP% | avg debit |
|------|--:|----:|----:|---:|------:|----:|----:|----------:|
| strat1 | 66 | 95.5 | $99138.0 | 12.15 | 119.8 | $7447.0 | 95.5 | $34.55 |
| strat2 | 31 | 96.8 | $84386.0 | 14.55 | 73.1 | $6226.0 | 96.8 | $30.23 |
| combined | 97 | 95.9 | $183524.0 | 13.14 | 104.9 | $13673.0 | 95.9 | $33.17 |

Meta: s1 signals=66 filled=66 skip_occ=0 skip_other=0; s2 signals=31 filled=31 skip_occ=0 skip_other=0

| Bucket | n | WR% | PnL | PF |
|--------|--:|----:|----:|---:|
| 2020-21 | 53 | 96.2 | $77224.0 | 11.37 |
| 2022-23 | 22 | 95.5 | $41772.0 | 7.71 |
| 2024-26 | 22 | 95.5 | $64528.0 | 45.78 |
| 2022 | 11 | 90.9 | $20926.0 | 4.36 |

## Interpretation

1. **mid/mid** is the closest to OptionsKit-style mid marks (video enters at mid on Robinhood).
2. **ask/bid** is the marketable long path (pay ask, exit bid) — haircuts WR/PnL vs mid.
3. Skips (`no_contract` / `no_eod`) mean Theta had no LEAPS-cycle NBBO for that gap day — not silent wins.
4. Even if mid/mid looks claim-like on $PnL, WR≠96 or occupancy/IV path failure still **no promote**; Sharpe>1 vs QQQ not walked until a clean fill book exists.

## Reproduce

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\thetadata_terminal.py
python scripts\research\claimcheck_qqq_leaps_gap_theta.py
```

Artifacts: `reports/research/qqq_leaps_gap/2026-09-30_theta_askbid_uncapped/`
