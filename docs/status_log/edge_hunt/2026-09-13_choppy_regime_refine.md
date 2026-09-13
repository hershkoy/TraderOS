# Choppy-regime refine (2026-09-13)

Follow-up to [first book](2026-09-13_choppy_regime.md). **No promote.** Do not mix these numbers with SMA200 + VIX>30 skip + 0.25% friction.

Changes vs first run:

1. Trend filter **SMA50** (not SMA200). Comparison book: **no MA + RSI(2)<2**.
2. **No VIX>30 sit-out.** Prior-session VIX scales size (`clip(20/VIX, 0.25, 1.0)`). Occupancy is still 15 names; size_mult only scales reported P&L.
3. Research friction **0.10%** (also log 0.05%). Not 0.25%.
4. Pullback **time stop 6** sessions (next open).
5. Live RSI-2 scan: **IB 15m last RTH close (prefer 15:45 ET)** overlaid on Alpaca daily — not Alpaca 17:00 EOD. Stored TimescaleDB, same path as the 15m hot list; do not stream the full daily universe at 15:45. Pullback limits stay an evening completed-daily job.

CPI-eve skip stays. PIT top-500 ADV lagged 1 session. ALPACA 1d, no IB prefix. Panel 1527×2217. Wall **93.4s** (panel cache hit).

## Results

SPY unchanged: **+183.7%**, Sharpe 0.67, MDD −34.1%.

| Book | n | E | PF | geo | WR | 2022 n / E / PF |
|------|---|---|----|-----|----|-----------------|
| RSI-2 SMA50 unconstrained | 3674 | +0.61 | **1.55** | 1.54 | 68% | **194 / +1.08 / 2.57** |
| RSI-2 SMA50 occ-15 | 2082 | +0.47 | 1.38 | 1.33 | 66% | 105 / +0.89 / 2.07 |
| RSI-2 SMA50 occ-15 fric 0.10 | 2082 | +0.37 | **1.30** | 1.24 | — | — |
| RSI-2 SMA50 occ-15 fric 0.05 | 2082 | +0.42 | 1.34 | 1.28 | — | — |
| RSI-2 extreme<2 noMA unconstrained | 6501 | +0.63 | 1.47 | 2.13 | 67% | 792 / +0.70 / 1.56 |
| RSI-2 extreme<2 noMA occ-15 | 2471 | +0.37 | 1.25 | 1.93 | 64% | **198 / −0.09 / 0.95** |
| RSI-2 extreme<2 noMA occ-15 fric 0.10 | 2471 | +0.27 | 1.18 | 1.76 | — | — |
| Pullback time-6 unconstrained | 4564 | +0.27 | 1.24 | 1.06 | 58% | 111 / +0.82 / 2.92 |
| Pullback time-6 occ-15 | 2785 | +0.33 | 1.29 | 1.14 | 60% | 48 / +0.86 / 2.33 |
| Pullback time-6 occ-15 fric 0.10 | 2785 | +0.23 | 1.19 | 1.05 | — | — |

Year buckets (gross; geo skips n<30 so **2018-19 and SMA50 2020-21 n=21 do not enter**):

| Book | 2020-21 | 2022-23 | 2024-26 |
|------|---------|---------|---------|
| RSI-2 SMA50 | n=21 E −0.02 PF 0.98 | n=999 E +0.52 PF 1.52 | n=2654 E +0.65 PF 1.56 |
| RSI-2 SMA50 occ-15 | n=21 | n=566 E +0.25 PF 1.22 | n=1495 E +0.56 PF 1.45 |
| RSI-2 extreme noMA | n=47 E +1.05 PF 4.85 | n=2264 E +0.36 PF 1.29 | n=4190 E +0.76 PF 1.56 |
| Pullback time-6 | n=15 | n=1304 E **−0.29 PF 0.76** | n=3245 E +0.50 PF 1.46 |
| Pullback occ-15 | n=15 | n=720 E **−0.10 PF 0.92** | n=2050 E +0.49 PF 1.42 |

Calendar **2023** (the 2022-23 red): SMA50 RSI-2 n=805 E +0.38 PF 1.36. Pullback n=1193 E **−0.40 PF 0.70** (occ-15 n=672 E −0.16 PF 0.86). Time-stop exits were 85 unconstrained / 57 occ-15 — not enough to fix 2023 knives.

VIX size-scale barely moves E (SMA50 occ-15 +0.47 → +0.46). Drop-top-3 barely moves E (not a fat-tail book).

First-run vs this refine (do not blend):

| | First (SMA200, VIX skip, fric 0.25) | Refine (SMA50, no skip, fric 0.10) |
|--|--|--|
| RSI-2 2022 n | 10 | **194** |
| RSI-2 occ-15 + friction PF | **0.96** | **1.30** |

## Verdict

**No promote.** SMA50 + dropping the VIX sit-out **does** put RSI-2 into 2022 (n=194 E +1.08), and 0.10% friction **does** pull occ-15 back above PF 1. That is the parameterization bug, not a new edge vs SPY.

Still not a 2022 last-15m H2 rescue and still not a Sharpe>1 sleeve: Alpaca empty 2018-19, SMA50 2020-21 n=21, unconstrained max concurrent 185, occupancy is 15 names not 15 units of VIX-scaled capital. Extreme RSI<2 with **no MA** wrecks occ-15 calendar 2022 (PF 0.95) — do not drop SMA50 to “trade the bear.” Pullback time-stop 6 does not clear **2023 PF 0.70**. Do not occupancy-walk onto last-15m.

Live 15:45: `scripts/scanners/choppy_regime_scan.py` overlays stored IB 15m (not Alpaca EOD). A live shortlist can reuse the IB 15m already pulled for `/hot` (client 8826). `--vix-skip` restores the old sit-out.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python -u scripts\research\run_choppy_regime.py --outdir reports\choppy_regime\refine
python scripts\scanners\choppy_regime_scan.py
```

Artifacts: `reports/choppy_regime/refine/2026-09-13/`. First-run CSVs stay in `reports/choppy_regime/2026-09-13/`.
