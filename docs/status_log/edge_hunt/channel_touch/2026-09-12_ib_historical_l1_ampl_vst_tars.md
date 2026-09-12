# IB historical top-of-book on AMPL / VST / TARS (2026-09-12)

Follow-up to [last-15m overlays](2026-09-12_last_15m_followup_overlays.md). Tried to pull **historical Level-2** around the three smoke losers. IB does **not** store DOM. What the API can return is historical **top-of-book** (BID/ASK/TRADES bars; BidAsk/Last ticks ~6 months).

Puller: `utils/research/ib_historical_l1.py` + `scripts/research/pull_ib_historical_l1.py` (client **8828**, port **4001**, light handshake so it does not wait on open orders next to 5m backfill). Tests: `tests/unit/test_ib_historical_l1.py` (13 passed). **No promote.**

Names in the overlay note are **AMPL / VST / TARS** (not APML / TATS; aliases accepted).

## Pull result (blocked)

Gateway API accepted the Python client (`Logged on to server version 176`, `usfarm` OK, `secdefil` OK). Then:

- **2110** Connectivity between Trader Workstation and **server is broken**
- **Error 200** qualifyContracts empty for AMPL/VST/TARS on SMART + NASDAQ/NYSE/AMEX/ARCA/BATS/ISLAND
- **10159** `reqMatchingSymbols` failed: Error sending message to a CCP
- **2107** HMDS `ushmds` inactive (available on demand — never reached hist)

Python-to-Gateway handshake is fine. Gateway-to-IB is not. Reconnect-to-IB in Gateway (not another `ib.connect()`). Do not pile 8826/8822/8823. TimescaleDB was also down (Docker engine pipe missing), so no 15m OHLCV fallback from `market_data`.

`get_ib_connection()` first failed on **open-orders / executions timeout** after API-ready (weekend 5m load). The script now uses the same **light connect** as VIX ingest (`ib.client.connectAsync`, skip account snapshots).

Even after Reconnect-to-IB, **ticks for these dates will likely be empty** (AMPL 2024-02, VST 2021-02, TARS 2023-07 are all older than the usual ~6 month tick window). The useful request is **1m + last-15m 5s BID/ASK/TRADES**.

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\research\pull_ib_historical_l1.py --smoke --ib-client-id 8828 --ib-port 4001 --no-ticks
```

## What L1 could still do on these three (not measured)

Tape was not pulled. The ideas below are the *questions* for a healthy Gateway re-run, mapped onto the overlay facts. They are not skip rules and not a sleeve.

| Name | Book | Overlay fact | L1 question (re-run) | Already known without L1 |
|------|------|----------------|----------------------|---------------------------|
| **AMPL** 2024-02-09 fill 14.17, sell 11.15 (−21.3%) | pos 1.03 form 0.04 | No doji. No 55% x2 seller. F2 uses **2024-02-13** (gain −0.53). Crash is **2024-02-21 gap** 14.07 to 9.22; ATR sells 09:45 ET. | Fill 15:45-16:00 ET: spread / last size / fill vs ask. **02-13:** does **bid** lose the rail while last still prints above? **02-21 09:30:** gap is overnight — bid cannot save the open. | Failed-break **daily close** already catches 02-13. F2 on the 2297 book fires 63% and clips winner $ (−3457). A bid-fail rule has to stay **rare** or it is F2 again. |
| **VST** 2021-02-18 fill 23.20 (−19.6%) | pos 1.007 **form 1.82** | F2 2021-02-19 gain −0.78. | Fill-day NBBO can look clean and still be a smashed channel. | Form 0.25 already drops it and **costs more winner $ than it saves**. Do not use tape to confirm a 1.82 overshoot. |
| **TARS** 2023-07-20 fill 23.24 (−19.4%) | **pos 1.492** form 0.37 | F2 day 2023-07-25 is after trail. | Last-15m **ask vs rail**: is the offer already ~80+ bps through resist (buying extension)? | Same family as `channel_pos` 1.25 (drops TARS, mild pooled bump, 2022-23 still red). |

## Ideas worth logging after a real pull (display-only)

1. **Thin/wide last-15m** — skip if median last-15m spread >= ~20 bps or last size <= 200. Only interesting if it flags AMPL-like ghost breaks without a TARS-style extension that geometry already has.
2. **Ask-extension vs rail** — TARS. Same as pos 1.25 using ask instead of 15m mid. Only interesting if it is *tighter live* than waiting for EOD pos.
3. **Bid loses rail on a later session, last still above** — AMPL 02-13. Early warning, **not** an occupancy-walked F2. F2 already failed the winner-$ gate.
4. **Do not** time AMPL's Feb 21 gap with L1. Overnight gap is not in the book.
5. **Do not** replace ATR k=2 with "bid wall gone."

**No promote.** Occupancy not re-walked. Grade vs hot-cross n=4079 E −0.19 PF 0.94 only after a real quote log exists.

Artifacts: `reports/ascending_channels/2026-09-12/ib_l1_smoke_summary.csv` (qualify errors only).
