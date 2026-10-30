# SPXW 0DTE iron fly — live `/hot` + combo

Detector **IF** (`timeframe=if`) on `/hot`. Soft-promote research sleeve (2y Phase-5 PASS; full-window near-miss with L5/S3/P2).

## Rules (live)

| Filter | Rule |
|--------|------|
| Weekday | Skip Wednesday |
| VIX | Session **open** &lt; 20 (Yahoo `^VIX`) |
| L5/S3/P2 | After 5 consecutive **taken** losses → Sit next 3 signals (paper) → Probe 2; any probe win → Active |
| Entry | 13:45 ET SPXW 0DTE; sell ~50Δ put / ~40Δ call + 15-pt wings; hold to cash settle |

## Notifications

Telegram + browser via `/hot` notify matrix (defaults **on** for IF hot / armed / approaching):

- **13:30–13:45 ET** — APPROACHING (or PAPER/SIT if L5 sitting)
- **13:45–14:05 ET** — ENTER NOW when filters + L5 take

CronRunner job `spxw_iron_fly_live` every minute **13:20–14:05 ET** Mon–Fri (`crons/spxw_iron_fly_live.bat`).

## Friday auto-park

On **Friday**, when ENTER NOW fires (filters + L5 take + 13:45–14:05 ET), the scanner runs:

```cmd
python scripts\trading\spxw_iron_fly_combo.py --place --no-transmit
```

Once per Friday (marker `logs/scanners/spxw_iron_fly_friday_park.json`). Order lands in TWS Orders **not transmitted** — review and click **Transmit**. Disable with `--no-auto-park`. Failures retry on the next minute tick until success or the window ends.

## Combo place (paper / live)

Same safety pattern as credit spreads (`--live-en` required for live `U*` accounts):

```cmd
venv\Scripts\activate
set PYTHONPATH=.
python scripts\trading\spxw_iron_fly_combo.py
python scripts\trading\spxw_iron_fly_combo.py --place --wait-sec 120
python scripts\trading\spxw_iron_fly_combo.py --place --no-transmit
python scripts\trading\spxw_iron_fly_combo.py --place --live-en --account Uxxxxxxx --wait-sec 120
```

`--no-transmit` parks the BAG in TWS Orders (`transmit=False`) so you can review and click **Transmit** yourself.

Or `crons\run_spxw_iron_fly_combo.bat` / `...\place` / `...\park` / `...\live-en Uxxxxxxx`.

Client id **8831**. API parent is **BUY BAG + negative LMT credit** with ComboLegs SELL body / BUY wings (same as credit spreads). TWS Mosaic "Short Iron Condor" often shows **SELL + positive credit** — that is a different bag definition; copying SELL/+credit onto this API bag causes Error 201 "Riskless combination..." even with a flat portfolio. Cancel Debit / V-shaped tickets. Fills under `reports/research/spxw_0dte_iron_fly/fills/` (and `paper_fills/latest.json` mirror).

## L5 state

Seeded from expand natural book; live decisions append to `logs/scanners/spxw_iron_fly_l5_state.json`. Settle PnL can be filled later via `record_signal_day(..., pnl=...)`.

## Code

- `utils/scanning/spxw_iron_fly_live.py`, `spxw_iron_fly_l5.py`
- `scripts/scanners/spxw_iron_fly_live.py`
- `scripts/trading/spxw_iron_fly_combo.py`
- Research HTML: `.../loss_streak/spxw_iron_fly_L5_S3_P2_report.html`
