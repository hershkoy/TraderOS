# Animesh / Dhan HA + 55 EMA channel + MACD swing

Research translation of the Dhan masterclass with Animesh K: [The Ultimate Swing Trading STRATEGY to Compound Your Capital](https://www.youtube.com/watch?v=2YBmiyVmNNw). YouTube transcript was unavailable; the encoding follows the published summary (Heikin-Ashi, 55 EMA high/low band, MACD 12/26/9, 1:3 R, Friday close).

**Verdict (2026-09-19):** **No promote.** See [status](../../status_log/edge_hunt/2026-09-19_ha_ema_macd_swing.md).

---

## Rules (this repo)

| Piece | Implementation |
|-------|----------------|
| Chart | Heikin-Ashi for **signals only** |
| Channel | EMA(55) of regular High and Low. Inside the band = skip. Long when HA close **breaks above** EMA High |
| Momentum | MACD(12,26,9) on HA close > 0 |
| Visual | HA color flip (prior red, signal green) |
| Fill | Next session **open** (not HA prices) |
| Stop | Signal-bar EMA Low (below the band). Gap through = loss at open |
| Target | 3R (4R is a named variant) |
| Other exits | Friday close; MACD line cross below signal (next open) |

Long-only. One position per symbol. Occupancy ranks by MACD histogram (stronger first).

---

## Independent rebuild

ALPACA 1d, PIT top-500 ADV, 2018-01-01 → 2026-09-18 (panel starts ~2020). Wall 320s.

| Book | n | E % | PF | geo | Notes |
|------|---|-----|----|-----|-------|
| Daily + Friday (primary) | 4047 | +0.23 | 1.20 | 1.26 | 75% Friday exits; median hold **1 day** |
| Daily occ-15 | 3483 | +0.21 | 1.18 | 1.23 | Sharpe **0.51** vs SPY 0.67 |
| Daily fric 0.25 | 4047 | −0.02 | 0.99 | 1.03 | Edge gone |
| Daily no-Friday | 3741 | +0.40 | 1.18 | 1.22 | Median **−1.40**; max concurrent 103 |
| Weekly resample | 634 | +2.64 | 1.55 | 1.97 | Median **−2.79**; 2022 n=2 |

Not a Sharpe>1 sleeve and not a 3:1 payoff book (realized daily payoff 1.17).

HTML/CSV: `reports/research/ha_ema_macd_swing/2026-09-19/`.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\run_ha_ema_macd_swing.py --weekly
```

Logic: `utils/research/ha_ema_macd_swing.py`
