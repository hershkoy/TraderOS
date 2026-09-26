# Last-15m short put via Black–Scholes (research gate)

**Verdict: interesting under constant IV, fragile under stop-IV stress — worth a cheap real-quote spot-check on stop fills, not a full-universe options API backfill / not a promote.**

Hypothesis: same last-15m H2 signal as `current_best` 1d, sell cash-secured puts instead of buying stock; take profit at 50% of credit; stop when underlying tags ATR k=2 (same stock hard-stop).

## Rules (locked)

- Signal unchanged: last-RTH 15m open at/above daily resist (`channel_touch_1d_last_15m.csv`, n=2297).
- Sell put at fill: **ATM** (rounded strike) or **delta ≈ −0.30**; default **DTE 45** (calendar, snapped to last weekday).
- IV at entry = annualized ATR% (`atr_pct/100 * sqrt(252)`), clamped 15–90%; held **constant** unless `--stop-iv-mult`.
- TP: buy back at `0.5 * credit` (limit-style).
- SL: underlying daily low ≤ stock `hard_stop_price` → buy put at BS marked at the stop (optional `stop_iv_mult` on that mark only).
- Else settle intrinsic on expiry session.
- Primary P&L: **cash-secured %** = `(credit − exit) / K` (comparable notional to stock). Also report ROP on credit.
- Occupancy **not** re-walked. No option bid/ask.

Constant IV is **optimistic on stops** (live crash IV expands). Stress `--stop-iv-mult 1.25/1.5` on hard-stop marks only.

## Results (wall ~244s cold 1d load / ~4–16s walk)

| Book | n | E % (CS) | PF | geo | 2022 E / PF | 2022-23 E / PF |
|------|--:|---------:|---:|----:|-------------:|---------------:|
| stock baseline | 2297 | +0.27 | 1.09 | 1.10 | −2.37 / 0.41 | −0.90 / 0.73 |
| ATM 45d IV×1 | 2297 | **+0.51** | **1.51** | **1.70** | **+0.11 / 1.08** | +0.22 / 1.21 |
| Δ−0.30 45d IV×1 | 2297 | +0.38 | 1.62 | 1.85 | +0.16 / 1.19 | +0.18 / 1.27 |
| ATM 45d stop-IV×1.25 | 2297 | −0.27 | 0.85 | 0.96 | −1.07 / 0.58 | −0.58 / 0.69 |
| ATM 45d stop-IV×1.5 | 2297 | −1.05 | 0.59 | 0.67 | −2.25 / 0.39 | −1.39 / 0.48 |
| Δ−0.30 stop-IV×1.5 | 2297 | −1.17 | 0.46 | 0.53 | −2.23 / 0.31 | −1.46 / 0.37 |

ATM IV×1 exits: **tp50 1097** (E +3.13 CS) / **hard_stop 1199** (E −1.89) / expiry 1.  
Avg win ~3.1% CS vs stock ~10.8%; avg loss ~−1.9% vs stock ~−4.5%. Higher WR (~49%) is the put payoff shape, not a better edge.

ROP (credit basis) ATM IV×1: E +8.0 PF 1.50 — looks fat because capital is not the premium.

## Read

- Constant-IV BS **clears** stock E/PF/geo and turns **2022 green** — opposite of the structural worry that puts only clip winners.
- That edge is **almost entirely stop-mark sensitive**: +25% IV on stop exits alone flips ATM to E −0.27 PF 0.85; +50% destroys it.
- So the open question for a real API is not “do winners hit 50%?” (they do under BS) but **what bid do you pay on ATR-stop days** (IV crush vs spike, spread).

## Real-API recommendation

1. **Do** a cheap spot-check: sample ~50–100 `hard_stop` rows from the ATM IV×1 trade CSV; pull EOD (or near-close) put quotes for strike/DTE via Polygon or IB; compare realized stop loss vs BS IV×1 / ×1.25 / ×1.5.
2. Optional: OptionsDX SPY-only claim-check on overlapping dates (already in TimescaleDB) — not the multi-name book.
3. **Do not** full-universe options backfill until stop fills land closer to IV×1 than IV×1.25.
4. **Do not** promote / wire `/hot` from this BS run.

## Artifacts

- Logic: `utils/research/last_15m_short_put_bs.py`
- Runner: `scripts/research/overlay_last_15m_short_put_bs.py`
- Tests: `tests/unit/test_last_15m_short_put_bs.py`
- CSVs: `reports/ascending_channels/2026-09-26/last_15m_short_put_bs_dte45_*`

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\overlay_last_15m_short_put_bs.py --workers 8 --strike-mode atm --dte 45
python scripts\research\overlay_last_15m_short_put_bs.py --workers 8 --strike-mode atm --dte 45 --stop-iv-mult 1.25
python scripts\research\overlay_last_15m_short_put_bs.py --workers 8 --strike-mode delta --dte 45
```

## Polygon free spot-check (2026-09-26)

Massive/Polygon Options Basic ($0, 5 req/min). Sample 12 recent liquid hard-stops from the ATM IV×1 book (`min_buy=2024-10-01`). Day-bar aggs on ATM put; snap to nearest listed expiry when synthetic DTE ticker is empty.

| | |
|--|--|
| usable | **9 / 12** (2 no_bars, 1 no_entry_bar) |
| median `live_exit / bs_exit` | **0.85** (mean 0.86) |
| bands | &lt;0.9: **6** · 0.9–1.1: **3** · ≥1.1: **0** |
| median live CS | −2.24 vs BS −1.83 |

Live stop closes were **cheaper** than constant-IV BS (opposite of the ×1.25 stress that killed the book). Caveats: EOD trade print ≠ buyback **ask**; thin names can share entry/exit session; n=9 only. Still **not a promote** — enough to keep free Polygon useful before paying for a larger sample.

Runner: `scripts/research/spotcheck_last_15m_short_put_polygon.py`  
CSV: `reports/ascending_channels/2026-09-26/last_15m_short_put_polygon_spot_20260926_223455.csv`

## HTML report (current_best style)

Same dark TV report shell as `reports/ascending_channels/current_best/` (equity, robustness, trade list, Run info, filter-books comparison).

- HTML: `reports/ascending_channels/2026-09-26/1d_last_15m_short_put_bs.html`
- Generator: `scripts/research/generate_last_15m_short_put_report.py`
- Not copied into `current_best/` (research / no promote)

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\generate_last_15m_short_put_report.py
```
