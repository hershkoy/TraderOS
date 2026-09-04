# IB 5m backfill (same universe as 15m)

**2026-09-04.** Start collecting IB **5m** for the IB 15m symbol set so later realistic fills can use a tighter clock. Not live. Not a 15m drop-in.

## Ops

- Universe = TimescaleDB IB `15m` symbols.
- Year-first: **2025-01-01 through now** for all symbols, then calendar 2024 … 2020. Resume is `MIN/MAX(ts)` inside that year window (global `MAX(ts)` would skip older years). Stopped 2026-09-04 after A/AAL full-history to switch.
- **Mon–Fri 09:15 ET:** stop 5m (`stop_ib_5m_backfill`) so live 15m (client **8826**) owns Gateway.
- **16:30 ET Mon–Fri:** `after_rth_ib_backfill` — IB 15m catch-up (8822) then 5m (8823) until next 09:15 ET. Friday continues through the weekend.
- **1d** stays Alpaca `channel_touch_nightly` (local 23:00 ≈ 16:00 ET). Does not use IB.
- Overnight 02:30 `backfill_ib_15m_universe` **disabled** (same 15m catch-up now runs at 16:30 ET).
- Weekend 00:00 ET safety restart if Friday’s job died.

Playbook: `docs/features/ib_5m_backfill.md`.

## Files

- `scripts/data/backfill_ib_5m_universe.py` (client 8823)
- `scripts/data/backfill_ib_5m_symbol.py` (client 8824)
- `crons/after_rth_ib_backfill.bat`, `crons/backfill_ib_5m_universe.bat`, `crons/stop_ib_5m_backfill.bat`
- `tests/unit/test_backfill_ib_5m_universe.py`, `tests/unit/test_backfill_ib_5m_symbol.py`
