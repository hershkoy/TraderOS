# Cup & Handle swing claim-check (2026-09-25)

Research translation of [This 5 Minute Swing Trade Beats 10,000 Hours Of Day Trading](https://www.youtube.com/watch?v=mExyCQrddTU) (The Cashflow Academy / Noah Davidson). Video transcript unavailable; rules from the published checklist + classical U-cup geometry.

Logic: [`utils/research/cup_handle.py`](../../../utils/research/cup_handle.py). Runner: [`scripts/research/backtest_cup_handle.py`](../../../scripts/research/backtest_cup_handle.py). Reports: `reports/research/cup_handle/2026-09-25/`.

## Rules (causal daily)

1. Trend: close > SMA20/50/200 with rising SMA50.
2. Cup: rounded U (not V), depth 12–35%, 20–65 bars; recovery volume vs 50-bar vol MA.
3. Handle: 5–25 bars, upper half of cup, depth ≤ ½ cup depth, volume dry-up.
4. Room to run: rim within 5% of prior 252d high.
5. Entry: buy-stop when high ≥ handle rim; fill `max(open, rim)`; cancel if close < handle low.
6. **Claim exit:** 1.5×ATR trail (prior-bar ratchet) + 50% at ½ cup depth + 50% at ⅔ cup depth.
7. **Keeper exit (robustness):** ATR k=2 clamp 1.5–6% + 10% peak trail (channel-touch).

Universe: top 300 IB 15m by ALPACA 1d ADV; liquid screen price ≥ $20 and 30d ADV shares ≥ 2M → **140** names. ALPACA 1d + IB prefix. 2018-11-01 → 2026-09-24. No P/E or optionable (not in TimescaleDB). Sector ETF trifecta skipped in practice (XLF/XLK/… not in `market_data`).

## JPM sanity

JPM last close **337.75** (near video $340 buy-stop). Detector armed **14** setups; **7** completed claim trades (mostly 2022–24, rims ~200 not 340 — the video’s live plan is a then-current setup, not a historical fill in our book). Example: 2024-06-28 fill 199.93 → trail exit +2.78%.

## Results

Wall-clock full claim+keeper first pass **~19m**; claim+occ+trifecta re-run (cache, skip-keeper) **10m 25s**.

| Book | n | E | PF | geo | WR | median | max_concurrent |
|------|---|---|----|-----|----|--------|----------------|
| CLAIM fric0 | 559 | **−0.31** | **0.81** | 0.85 | 36% | −1.08 | 15 |
| CLAIM fric0.25 | 559 | −0.56 | 0.69 | 0.72 | 33% | −1.33 | 15 |
| KEEPER fric0 | 458 | +0.44 | 1.15 | 1.19 | 28% | −3.52 | 26 |

Year buckets (claim fric0): 2018–19 n=23 E −0.17; 2020–21 n=208 E −0.30; 2022–23 n=129 E +0.24 PF 1.17; 2024–26 n=199 E −0.69. Drop-top-3 claim E −0.38.

### Occupancy vs SPY (claim)

| Cap | kept | Sharpe | MDD | vs SPY | Gate |
|-----|------|--------|-----|--------|------|
| occ1 | 180 | −0.34 | −61% | SPY 0.77 / −25% | **FAIL** |
| occ2 | 299 | −0.19 | −36% | SPY 0.77 / −25% | **FAIL** |

Trifecta overlay: **n=0** (sector ETFs missing from TimescaleDB).

## Verdict

**No promote.** Claim book is net negative expectancy / PF < 1. Keeper ATR trail is only soft (PF ~1.15, median is the stop, 2020–21 red, concurrent 26). Fixed-slot occupancy fails Sharpe>1 and loses to SPY on Sharpe and MDD. Do not wire live / Finviz scanner. Treat as a failed claim-check of the video framework under classical causal rules — not a reject of discretionary cup-handle discretionary trading.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\backtest_cup_handle.py --jpm-sanity --symbols JPM --n-symbols 1
python scripts\research\backtest_cup_handle.py --n-symbols 300 --skip-keeper
pytest tests\unit\test_cup_handle.py -v
```
