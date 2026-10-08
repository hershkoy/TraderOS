# Gozlan PCE weekly claim-check

- Generated: 2026-10-07T18:29:35.857335+00:00
- Wall clock: 23.5s
- Source: [Gozlan PCE model](https://www.youtube.com/watch?v=8ru-XJOs7bU)
- Pair: QQQ vs SPY (ETF proxy for NQ vs ES from the video)
- Scope: **weekly signal book only** — long the green leg for one week after a Precision Candle
  (one closes green, the other red). No 4H FVG/OB entry layer (no QQQ intraday / no NQ-ES futures in DB).

## Coverage

- Aligned Friday weeks: **1440**
- Weekly PCE events: **201** (14.0% of weeks)

## Books

| Book | n | WR | PF | Avg% | CAGR | Sharpe | MDD | Exposure | Gate |
|------|---|----|----|------|------|--------|-----|----------|------|
| long_green_claim_close | 172 | 52.91% | 0.948 | -0.054 | -0.48% | -0.069 | 44.53% | 9.86% | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| long_green_next_open | 172 | 52.91% | 0.905 | -0.104 | -0.75% | -0.11 | 48.53% | 9.86% | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| long_green_next_open_sweep | 130 | 50.0% | 0.852 | -0.159 | -0.77% | -0.13 | 36.06% | 7.49% | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| long_green_next_open_fric0.10 | 172 | 50.0% | 0.745 | -0.304 | -1.76% | -0.258 | 54.38% | 9.86% | FAIL (Sharpe<=1.0, Sharpe<SPY) |
| long_green_next_open_sweep_fric0.10 | 130 | 46.92% | 0.695 | -0.359 | -1.53% | -0.259 | 44.86% | 7.49% | FAIL (Sharpe<=1.0, Sharpe<SPY) |

SPY buy&hold on same window: CAGR 8.9% Sharpe 0.48 MDD 56.47%.

## Verdict

**No promote.** Weekly PCE as an HTF long-the-green signal on QQQ/SPY does not clear Sharpe>1 / Sharpe>=SPY / MDD<SPY. This does **not** validate the full video model (True Opens bias + 4H FVG/OB entry + London/NY session filter).
Honest next-open book: n=172 E -0.104% PF 0.905 Sharpe -0.11 (FAIL (Sharpe<=1.0, Sharpe<SPY)).

### Data gaps vs video

- No NQ/ES futures OHLCV in TimescaleDB (used QQQ/SPY Yahoo daily).
- QQQ has no IB 15m/5m — cannot resample a causal 4H entry book on the Nasdaq leg.
- DIA / IWM not in TimescaleDB; Dow proxy skipped.
- True London/Daily Open discount/premium filter not applied (needs intraday opens).

Artifacts: `D:/WORK/Projs/IB/backTraderTest/reports/research/pce_weekly/`; harness `utils/research/pce_weekly.py` / `scripts/research/claimcheck_pce_weekly.py`.

