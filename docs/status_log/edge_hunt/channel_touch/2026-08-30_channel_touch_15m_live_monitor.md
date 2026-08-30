# 15m live monitor + IB 15m backfill — 2026-08-30

Built the research 15m H5 live loop and an incremental IB 15m universe backfill.
**Does not replace daily nightly.**

Inventory 2026-08-30: **1,479** IB 15m symbols; last-bar **median 2025-11-28**,
max 2026-08-26 (1 name). **1,478 names have no 2026 bars.** Run the backfill
with Gateway up before trusting live fills.

Playbook: `docs/features/channel_touch_15m_live.md`.

## Why this shape

Full-universe IB every 15m does not fit (hist pacing ~25 min; ~100 streaming
lines; full detector on 2018–now was 51 min). Alpaca IEX snapshot of 1,478
names was **1.3s** (2026-08-30) for last price + daily volume. Alpaca IEX 15m
bars only came back for ~487/1478 and cannot feed the IB `volume_rel` gate.

Live design: armed H2 watchlist on stored IB 15m → Alpaca last at/above resist
→ completed-bar H5 stack (wait-12, span<=10, unique-symbol/day, prior-bar
vol>=2, fill overshoot>=0.08 frozen floor).

Proximity is **at or above** resist. Tight-break-from-below was the rejected
unique H2 filter. Quality names are often already through the rail after wait-12.

## Commands

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\data\backfill_ib_15m_universe.py --inventory
python scripts\data\backfill_ib_15m_universe.py --sleep 1 --ib-client-id 8822
python scripts\scanners\channel_touch_15m.py --mode run --dry-run --max-symbols 50
```

Crons: `crons\backfill_ib_15m_universe.bat`, `crons\channel_touch_15m.bat`.

## Files

- `utils/scanning/channel_touch_15m.py`
- `scripts/scanners/channel_touch_15m.py`
- `scripts/data/backfill_ib_15m_universe.py`
- `tests/unit/test_channel_touch_15m.py`
- `tests/unit/test_channel_touch_15m_scanner.py`
- `tests/unit/test_backfill_ib_15m_universe.py`

Coverage CSV after `--inventory`: `reports/ascending_channels/ib_15m_coverage.csv`.
