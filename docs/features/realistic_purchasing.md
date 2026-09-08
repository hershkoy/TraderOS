# Realistic purchasing (1d and 15m)

How channel-touch backtests turn a **signal** into a **buy price** that a live trader could have gotten.

Code: `utils/research/realistic_purchaser.py`. Flags: `--realistic-fill` and `--realistic-fill-mode` on `scripts/research/backtest_channel_touch_trades.py` and `scripts/research/backtest_channel_touch_h2_break.py`.

This is a **research fill**. The nightly scanner (`scripts/scanners/channel_touch_nightly.py`) still uses the **unrealistic** daily rail clip (not this purchaser). Occupancy and ATR/trail **exits** stay on the strategy timeframe (daily bars for 1d, 15m bars for 15m) even when the entry is priced from 15m.

---

## What belongs in `current_best/`

`reports/ascending_channels/current_best/` is the **live-executable, causal** gate — even if the book loses. That is the number the next scan has to beat. Clip / post-EOD overlays stay in `1d_unrealistic/`.

| Slot | File | Why it is here |
|------|------|----------------|
| **1d number to beat** | `current_best/1d_hot_cross.html` (same bytes as `1d_channel_touch.html`) | Buy-now `--intraday-trigger hot-cross` `--hot-cross-fill lerp85`. Unique span365 + shakeout **n=4079 E −0.19 PF 0.94**. Causal. Live analog is `/hot` armed + Alpaca last. **Do not promote.** |
| **15m keeper** | `current_best/15m_channel_touch.html` | L3 wait-12 **signal-close** + `--touch-error-pct 0`. **n=1744 E +0.13 PF 1.22**. Completed 15m close is workable. |

The detector on the **clip** book fires on the **daily close** above resistance. Default fill (`_limit_fill_at_support`) is rail + slip **clipped to that same daily bar's [low, high]**. After 16:00 ET you know the close printed above resist; you cannot go back and buy the rail or the day's low. Nightly still uses this path. Call it **unrealistic**, not optimistic. Those reports live in `reports/ascending_channels/1d_unrealistic/`:

| File | What it is |
|------|------------|
| `1d_channel_touch.html` / `1d_h2_resist_break.html` | Nightly H2 resist-break + shakeout visualization. Entry is the clip (or a next-mid overlay still keyed off that clip). |
| `1d_l3_touch.html` | Retired L3 support-tag keeper |
| `1d_keeper_plus_h2_resist_break.html` | Research L3 + H2 sleeve |

`--realistic-fill` on **1d** (next-mid / open-cross / next-open) is a research overlay, not the `current_best` 1d slot:

- **next-mid** still starts from the unrealistic rail **X**, then blends with a 15m mid. The 2026-09-05 HTML (n=1239) joined the **wrong 15m session** until `_as_session_date`. Honest rebuild **n=1986 E +2.24 PF 1.89**. CTF stamps midnight UTC, so the BUY still sits on the daily wick (RDWR **24.64** was the wrong-session blend; honest Friday wild-cancels). Not `current_best`.
- **open-cross** / **next-open** lost the E/PF gate vs the rail-clip keeper — not promoted. Nightly stays the clip.

Hot-cross is in `current_best/` because it is live-executable, not because it clears the clip book's E/PF. Grade the next 1d idea against **n=4079 E −0.19 PF 0.94**, not against clip n=4558 E +2.62 PF 2.10.

---

## Open vs close

IB/TimescaleDB **15m** bars are labeled at **period start** (naive UTC). Session rules use America/New_York.

| Clock (ET) | Bar `ts` | What is known |
|------------|----------|----------------|
| 09:30 | 09:30 start | **Open** of that 15m |
| 09:45 | same bar completes | **Close** of the 09:30 bar (also the open of the 09:45 bar) |
| 16:00 | 15:45 bar completes | Last RTH **close**. There is no following RTH 15m |

A 15m **open** is known at the print. A 15m **close** is known only when that bar finishes (`signal_time` = period start + 15m).

**Daily (1d)** bars are keyed by the US cash **session calendar date at 00:00** (naive or UTC). Do not convert that midnight stamp to New York: Monday `00:00` UTC would become Sunday evening, and a 15m lookup would miss every Monday. Helper: `_as_session_date` in the purchaser.

The **detector** always uses **close** to fire:

- 1d H2 resist-break: daily **close** above resistance after H2.
- 15m H5 / L3: completed 15m **close** (above resist, or a from-above support tag on that bar).

`--realistic-fill` does **not** change that signal. It only changes **where you buy** after the signal exists.

### Without `--realistic-fill` (unrealistic)

Call this **unrealistic**, not optimistic. Fill is `_limit_fill_at_support`: rail + slip, **clipped to that same bar's [low, high]**. On a 1d H2 resist-break that is often the rail sitting on a wick, or the **daily low** on a gap-through day. The detector only knows close-above-resist **after** the session; you cannot go back and buy the rail. Nightly still uses this path.

**RDWR 2025-06-13** in `reports/ascending_channels/1d_unrealistic/1d_channel_touch.html` (CTF):

```json
{"v":1,"sym":"RDWR","src":"rails","fill":"2025-06-13T00:00:00Z","fill_ms":1749772800000,"en":"2025-06-13T00:00:00Z","en_ms":1749772800000,"l1t":"2024-07-19T00:00:00Z","l1_ms":1721347200000,"l1p":17.42,"l2t":"2025-04-07T00:00:00Z","l2_ms":1743984000000,"l2p":18.885,"w":5.200391,"enp":24.64,"ex":"2025-07-11T00:00:00Z","ex_ms":1752192000000,"exp":28.251,"h2t":"2025-05-13T00:00:00Z","h2_ms":1747094400000}
```

CTF stamps `fill` at midnight UTC, so the BUY sits on the **daily** candle at **24.64** (`enp`) — the resistance rail through that bar. Alpaca 1d that session: O=L=**24.65**, H=26.93, C=**26.58**. The day gapped through resist; the clip buys the daily low / rail. On the chart that is a BUY on the wick/rail, not a price you could work after 26.58 printed. IB 15m opened **24.37** at 09:30 ET (still under the rail). Same exit 2025-07-11 @ 28.251.

### With `--realistic-fill`

Four modes (`--realistic-fill-mode`):

| Mode | 15m strategy | 1d strategy |
|------|----------------|------------------------------------------|
| **`signal-close`** (CLI default) | Buy the **close** of the signal 15m. Last RTH bar is allowed. | First 15m whose range **contains daily X**; buy that 15m **close**. Last RTH print is allowed. |
| **`next-mid`** | Buy the **next** same-session 15m **mid** `(high+low)/2`. Last RTH (15:45) is cancelled. Wild-bar cancel if `(mid-low)/mid > 0.5%`. | First 15m that printed X; fill `(X + next_15m_mid) / 2`. Same EOD / wild cancel. |
| **`open-cross`** | Same as `signal-close` (native 15m already has a completed bar). | First 15m whose **open** is already above resist; buy that bar's **close**. Last RTH allowed. Skip if no 15m opens above resist. |
| **`next-open`** | Next same-session 15m **open** (last RTH cancelled). | Next **daily** session **open** after the EOD close. No IB 15m join. Last bar of the sample is skipped. |

**Open** is used as a fill gate in **1d `open-cross`** and as the fill price in **`next-open`**. Close/mid modes still buy a close or a next-bar mid.

Live-plausible reading:

- **Close fill:** the bar finished; you can work the close (or MOC-style) once you see it.
- **Next-mid:** you confirmed at T's close (which is T+1's open) and assume you get the next 15m's midpoint. Cancels when there is no next RTH bar.
- **Open-cross (1d):** you see a 15m **open already through** the rail, wait for that 15m to **close**, buy the close. You do not buy a bar that opened below.
- **Next-open (1d):** EOD close confirms; buy the **next session open** (MOO). Occupancy and ATR/trail start on that next daily bar; same-day stop after the open is allowed.

---

## How a 1d strategy prices from 15m bars

The daily book still **scans daily OHLCV** (Alpaca 1d, IB prefix for history). When `--realistic-fill` is on and `--timeframe 1d`, modes other than **`next-open`** also load **IB 15m** (`load_ohlcv_many(..., timeframe="15m", provider="IB")`). Names with no IB 15m are skipped (no prior-day fallback). **`next-open`** uses the next daily bar's **open** and keeps the full daily universe.

Per daily signal:

1. Take the detector fill price **X** (the unrealistic rail clip above) and the session date of that daily bar.
2. Keep only RTH 15m starts that session (`09:30` through `15:45` ET, weekdays).
3. Reprice:

**`signal-close`** — walk 15m until `[low, high]` contains X. Fill = that bar's **close**.

**`next-mid`** — same first print of X. If the next same-session 15m exists and is not wild, fill = `(X + next_mid) / 2`. A 15:45 print of X has no window and is dropped.

**`open-cross`** — ignore X. Walk until **open > resist** (resist from the daily rails at that session). Fill = that 15m **close**. Gap-through days that never open above resist are skipped.

**`next-open`** — skip 15m. Fill = next daily session **open**. `buy_date` is that next session. Skip if there is no next bar.

`buy_time` is that 15m bar (UTC). `buy_date` stays the calendar session (occupancy / unique-symbol-per-day key). Exits still walk **daily** highs/lows/closes.

Same RDWR day: 09:30 ET 15m O **24.37** is skipped (`open-cross` requires open already above resist). First open above is 09:45 ET O 25.19; fill that bar's close **25.71**. That is the workable gap-through purchase. The HTML CTF BUY 24.64 is the unrealistic daily clip.

Do not convert the daily index to New York for the 15m join. Pass the calendar date (`YYYY-MM-DD` from the daily stamp at midnight).

---

## 15m strategies

The strategy **is** 15m. No extra join.

- Detector fires on completed bar **T**.
- **`signal-close`:** stay on T, fill = T **close**. BUY sits on the tag/break candle.
- **`next-mid`:** fill on T+1 at mid. The 15:45–16:00 bar cannot fill (confirm at 16:00 has no RTH window; do not roll overnight).
- **`next-open`:** fill on T+1 at that 15m **open**. Same last-RTH cancel.

`next-mid` can print a BUY **mid-channel** that never tagged the rail (GOOGL 2025-11-18: tag at 10:00 ET close 280.79, next mid 281.63). That is why 15m L3 `current_best` uses **signal-close**, not next-mid. 1d next-mid never earned that slot — the overlay still keys off the daily rail clip.

Optional `--max-chase-pct` (off by default) caps how far the exec price may run vs the signal. `--max-low-to-mid-pct` (default 0.5%) applies to **next-mid** only.

---

## What is live vs research

| Book | Fill |
|------|------|
| Nightly 1d H2 (`channel_touch_nightly`) | Unrealistic **daily rail clip** (not this purchaser) |
| `current_best/1d_hot_cross.html` / `1d_channel_touch.html` | Buy-now hot-cross lerp85 unique **n=4079 E −0.19 PF 0.94**. Live-executable. **The 1d number to beat.** Not promoted. |
| `1d_unrealistic/1d_channel_touch.html` | Same clip, or next-mid overlay still keyed off that clip (HTML n=1239 **pre-`_as_session_date`**; honest rebuild unique **n=1986 E +2.24 PF 1.89**). |
| `current_best/15m_channel_touch.html` (L3 wait-12) | `--realistic-fill` **signal-close** (live-executable completed 15m) |
| 1d `open-cross` | Research switch; **not** promoted; not `current_best` |
| 1d `next-open` | Research switch (EOD then MOO); **not** promoted; not `current_best` |

`--intraday-trigger hot-cross` is not a post-EOD reprice. Daily H2 arms from prior completed bars; the first RTH 15m whose high reaches resist is the buy-now bar (Alpaca last on `/hot` is the live analog). Fill on that 15m:

- **`lerp85`:** `rail + 0.85 * (close - rail)` if close >= rail, else rail; clamp to `[low, high]`.
- **`rail`:** rail, or **open** if that 15m already opened >= rail.
- **`close`:** that 15m close.

A **15m gap** is `open >= rail` on the trigger bar. That is not a daily gap: RDWR 2025-06-13 Alpaca 1d opened through the rail, IB 15m **09:30 O 24.35** still under **24.64** (`gap_15m=False`). `open-cross` skips that 09:30 bar and buys the 09:45 close. IB 5m is not the universe fill clock yet.

```bat
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --intraday-trigger hot-cross --hot-cross-fill lerp85 --shakeout-breakout --workers 4 --load-workers 8
```

IB **5m** is being ingested for a tighter clock later (`docs/features/ib_5m_backfill.md`). It is not wired into this purchaser yet.

**2026-09-06 full-universe A0 (lerp85 + shakeout, unique span365):** **n=4079 E −0.19 PF 0.94**. Gap splits stay ~PF 0.92–0.95. RDWR 2025-06-13 is absent (already in a trade from the 2025-06-05 rail poke). **Do not promote.** This is the `current_best/` 1d slot (`1d_hot_cross.html`) — the number to beat. Nightly stays the clip. Write-up: [hot-cross](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_1d_15m_trigger.md).

`--intraday-trigger close-cross` waits for a 15m **close** above the daily rail.
Fill follows `--realistic-fill-mode`: **`signal-close`** buys that confirm bar's
close (last RTH allowed); **`next-mid`** / **`next-open`** buy the next
same-session mid/open (15:45 confirm cancels). Not buy-now; not `current_best`.
Formation hygiene: `--max-formation-beyond-width 0.25` drops L1→H2 overshoots
(ADM 2021-11-23). Full unique confirm-close + form + span365 **n=1947 E +0.24 PF 1.07**
(2022-23 fails) — **no promote.** See [close-confirm formation](../status_log/edge_hunt/channel_touch/2026-09-07_channel_touch_h2_close_confirm_formation.md).
RDWR microscope + `--trail-mae` next-mid: [close-cross MAE](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_close_cross_mae.md).

```bat
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --intraday-trigger close-cross --realistic-fill --realistic-fill-mode signal-close --max-formation-beyond-width 0.25 --workers 4 --load-workers 8
```

---

## Commands

```bat
venv\Scripts\activate
set PYTHONPATH=.

python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --intraday-trigger hot-cross --hot-cross-fill lerp85 --shakeout-breakout --workers 4 --load-workers 8

python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --realistic-fill --realistic-fill-mode next-mid --shakeout-breakout --workers 4 --load-workers 8

python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --realistic-fill --realistic-fill-mode open-cross --shakeout-breakout --workers 4 --load-workers 8

python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --realistic-fill --realistic-fill-mode next-open --shakeout-breakout --workers 4 --load-workers 8
python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --shakeout-breakout --touch-error-pct 0 --workers 4 --load-workers 8

python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --entry-mode l3_touch --min-l3-wait-bars 12 --realistic-fill --touch-error-pct 0 --workers 4 --load-workers 8
```

---

## Related

- Status notes: [hot-cross in current_best](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_current_best_hot_cross.md), [1d clip out of current_best](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_1d_out_of_current_best.md), [realistic fill rescan](../status_log/edge_hunt/channel_touch/2026-09-04_channel_touch_realistic_fill.md), [signal-close](../status_log/edge_hunt/channel_touch/2026-09-04_channel_touch_signal_close.md), [open-cross](../status_log/edge_hunt/channel_touch/2026-09-05_channel_touch_open_cross.md), [next-open A/B](../status_log/edge_hunt/channel_touch/2026-09-05_channel_touch_next_open_ab.md), [hot-cross](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_1d_15m_trigger.md)
- Live 15m monitor (H5 close-confirm, not this purchaser): [channel_touch_15m_live.md](channel_touch_15m_live.md)
