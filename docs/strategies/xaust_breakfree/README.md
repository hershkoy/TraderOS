# XAUSTBreakFree (Jesse port)

Port of Jesse `XAUSTBreakFree`: 1h Donchian breakout + 4h SuperTrend regime, long/short, ATR stops/trails, 3% risk.

## Code

- Strategy: `strategies/xaust_breakfree.py`
- Config: `defaults.yaml` → `strategies.xaust_breakfree`
- Tests: `tests/unit/test_xaust_breakfree.py`

## Data reality

| Instrument | Status |
|---|---|
| XAUUSD / GC IB futures | Not in DB yet |
| **GC1!** 1h (TV `COMEX_DL:GC1!`) | **Ingested** TRADINGVIEW **n=22062** 2023-01-02 → 2026-09-24 |
| GLD 1h | Alpaca proxy (RTH ETF) |
| GLD 1d | ALPACA + IB |

TV free/delayed continuous 1h does not page before ~2023. Dump: `reports/tv_dumps/gc1_1h.json`. Re-ingest: `python scripts\\research\\_ingest_tv_gc1_1h.py`.

## Run (GC1! futures proxy)

```bat
venv\Scripts\activate
set PYTHONPATH=.
python backtrader_runner_yaml.py --config defaults.yaml --symbol GC1! --provider TRADINGVIEW --timeframe 1h --strategy xaust_breakfree --fromdate 2023-01-01 --todate 2026-09-25
```

Or: `python scripts\\research\\_run_xaust_gc1_summary.py` (CME session filter on).

## Smoke results

### GC1! TV (CME session filter, Jesse default HPs, 2023-01 → 2026-09)

| Metric | Value |
|---|---|
| Equity | 100k → **91.8k (−8.2%)** |
| Max DD | 18.6% |
| Sharpe (approx) | **≈ −0.22** full / **−0.26** 2024+ |
| Trades | 89 closed — **all shorts** |

Better than GLD ETF proxy but still fails Sharpe≥1.5. Cash-broker sizing ≠ IB GC/MGC multipliers — directional gate only.

### GLD Alpaca (earlier)

| Metric | Value |
|---|---|
| Equity | 100k → 73.9k (−26.1%) |
| Sharpe | ≈ −1.4 |
| Trades | 28 shorts |

Not a promote / not a claim-check of the original XAU futures study.


