# Realistic purchasing (1d and 15m)

How channel-touch backtests turn a **signal** into a **buy price** that a live trader could have gotten.

Code: `utils/research/realistic_purchaser.py`. Flags: `--realistic-fill` and `--realistic-fill-mode` on `scripts/research/backtest_channel_touch_trades.py` and `scripts/research/backtest_channel_touch_h2_break.py`.

This is a **research fill**. The nightly scanner (`scripts/scanners/channel_touch_nightly.py`) still uses the **unrealistic** daily rail clip (not this purchaser). Occupancy and ATR/trail **exits** stay on the strategy timeframe (daily bars for 1d, 15m bars for 15m) even when the entry is priced from 15m — that daily stop fill is the same-bar clip (see [Realistic sells](#realistic-sells-1d-last-15m-book)).

**`--realistic-fill` on 1d does not make the book live.** After a daily close-above-resist, buying that same session's 15m close / mid / open-cross is a lookback. **Avoid those as strategies** ([Do not use](#do-not-use-1d-post-eod-lookbacks)). The 15m L3 keeper and 1d `--intraday-trigger hot-cross` are the causal clocks.

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
| `1d_h2_signal_close.html` | 1d `--realistic-fill-mode signal-close` lookback (2026-09-09 unique **n=2794 E +2.48 PF 1.99**). Same class as the clip. **Do not use.** |
| `1d_l3_touch.html` | Retired L3 support-tag keeper |
| `1d_keeper_plus_h2_resist_break.html` | Research L3 + H2 sleeve |

Hot-cross is in `current_best/` because it is live-executable, not because it clears the clip book's E/PF. Grade the next 1d idea against **n=4079 E −0.19 PF 0.94**, not against clip n=4558 E +2.62 PF 2.10, and not against 1d signal-close n=2794 E +2.48 PF 1.99.

---

## Do not use: 1d post-EOD lookbacks

If the detector waits for that session's **daily close** above resist, any fill on **that same session** is unrealistic. At 16:00 ET you know the close printed; you cannot go back and buy a 15m close, a rail, or a mid from earlier that day. Occupancy still walks **daily** bars and skips the fill-day stop (`skip_entry_bar_stop`), so the book also misses same-day adverse excursion.

**Avoid these as strategy candidates.** Do not promote, wire to `/hot` or nightly, rescan for “better” filters, or grade live ideas against their E/PF. The numbers look good because the fill is earlier than the information.

| Book | Why it is a lookback | Do not treat as |
|------|----------------------|-----------------|
| Daily rail clip (`_limit_fill_at_support`) | Buy rail/wick on the signal daily bar. | Live fill. Nightly still clips for the watchlist only. |
| 1d `--realistic-fill-mode signal-close` (no `--intraday-trigger`) | First RTH 15m whose `[low, high]` contains rail **X**; buy that 15m **close**. Confirm is still the daily close. HTML List of trades is **calendar dates only** (no `buy_time` / `sell_time`; CTF BUY at midnight UTC). Dated report: `reports/ascending_channels/2026-09-09/channel_touch_tv_report_interactive_fric0.25_1d_h2_signal_close_span365_shakeout_20260909_013839.html`. | Realistic. Unique **n=2794 E +2.48 PF 1.99**. |
| 1d `next-mid` | Same **X**, then `(X + next_15m_mid) / 2`. The 2026-09-05 HTML (n=1239) joined the **wrong 15m session** until `_as_session_date`. Honest rebuild **n=1986 E +2.24 PF 1.89**. CTF still sits on the daily wick. | Live fill. |
| 1d `open-cross` without `--intraday-trigger` | First 15m **open** above resist on the **signal** day, fill that bar's close. Still needs the daily close first. | Live fill. Lost E/PF vs clip besides the lookback. |

`--intraday-trigger hot-cross` / `close-cross` do **not** wait for that session's daily close. Those are a different clock (see below). **15m** `signal-close` is also different: the detector *is* the completed 15m, so buying that close is causal (15m keeper).

1d `next-open` (EOD then next-session MOO) is live-executable and still **not** promoted (lost E/PF vs clip). It is not a same-session lookback.

**`next-open-mid`** is the same clock with a 15m fill: buy the **mid** of the next session's 09:30 ET 15m. Live after the daily close. Compare vs the next-mid lookback: `reports/ascending_channels/1d_unrealistic/next_open_mid_compare.csv`. Do not promote unless it beats hot-cross **n=4079 E −0.19 PF 0.94**.

**Last-15m-open-mid** (same-session last RTH 15m mid if that bar **opened** above the rail) is EOD-contemporaneous on the **buy**. The first HTML (`…last_15m_open_mid_span365_20260911_132918.html`) **kept daily occupancy sells** — that is the same-bar stop clip. WVE 2023-12-06 buy 6.85 / sell 2023-12-07 @ 6.0254 is that clip (original ATR stop from the cheaper signal-close fill, filled on the next daily bar's low). See [Realistic sells](#realistic-sells-1d-last-15m-book).

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

`--realistic-fill` does **not** change that signal. It only changes **where you buy** after the signal exists. On **1d** without `--intraday-trigger`, that reprice is still a same-session lookback — **do not use it as a strategy** (see above).

### Without `--realistic-fill` (unrealistic)

Call this **unrealistic**, not optimistic. Fill is `_limit_fill_at_support`: rail + slip, **clipped to that same bar's [low, high]**. On a 1d H2 resist-break that is often the rail sitting on a wick, or the **daily low** on a gap-through day. The detector only knows close-above-resist **after** the session; you cannot go back and buy the rail. Nightly still uses this path.

**RDWR 2025-06-13** in `reports/ascending_channels/1d_unrealistic/1d_channel_touch.html` (CTF):

```json
{"v":1,"sym":"RDWR","src":"rails","fill":"2025-06-13T00:00:00Z","fill_ms":1749772800000,"en":"2025-06-13T00:00:00Z","en_ms":1749772800000,"l1t":"2024-07-19T00:00:00Z","l1_ms":1721347200000,"l1p":17.42,"l2t":"2025-04-07T00:00:00Z","l2_ms":1743984000000,"l2p":18.885,"w":5.200391,"enp":24.64,"ex":"2025-07-11T00:00:00Z","ex_ms":1752192000000,"exp":28.251,"h2t":"2025-05-13T00:00:00Z","h2_ms":1747094400000}
```

CTF stamps `fill` at midnight UTC, so the BUY sits on the **daily** candle at **24.64** (`enp`) — the resistance rail through that bar. Alpaca 1d that session: O=L=**24.65**, H=26.93, C=**26.58**. The day gapped through resist; the clip buys the daily low / rail. On the chart that is a BUY on the wick/rail, not a price you could work after 26.58 printed. IB 15m opened **24.37** at 09:30 ET (still under the rail). Same exit 2025-07-11 @ 28.251.

### With `--realistic-fill`

Four modes (`--realistic-fill-mode`):

| Mode | 15m strategy (causal) | 1d strategy **without** `--intraday-trigger` |
|------|----------------|----------------|
| **`signal-close`** (CLI default) | Buy the **close** of the signal 15m. Last RTH bar is allowed. **Keeper.** | **Unrealistic lookback. Do not use.** First 15m whose range **contains daily X**; buy that 15m **close**. Confirm is still the daily close at 16:00 ET. |
| **`next-mid`** | Buy the **next** same-session 15m **mid** `(high+low)/2`. Last RTH (15:45) is cancelled. Wild-bar cancel if `(mid-low)/mid > 0.5%`. | **Unrealistic lookback. Do not use.** First 15m that printed X; fill `(X + next_15m_mid) / 2`. Same EOD / wild cancel. |
| **`open-cross`** | Same as `signal-close` (native 15m already has a completed bar). | **Unrealistic lookback. Do not use.** First 15m whose **open** is already above resist on the signal day; buy that bar's **close**. |
| **`next-open`** | Next same-session 15m **open** (last RTH cancelled). | Next **daily** session **open** after the EOD close. No IB 15m join. Live-executable MOO; **not** promoted. |
| **`next-open-mid`** | Same as 15m `next-open` (next 15m mid is a different mode). | Next session **09:30 ET 15m mid**. Needs IB 15m. Live-executable. **Not** promoted until it beats hot-cross. |

**Open** is used as a fill gate in **1d `open-cross`** and as the fill price in **`next-open`**. Close/mid modes still buy a close or a next-bar mid.

Live-plausible reading applies to **15m** native fills and to `--intraday-trigger` / 1d `next-open` only — **not** to 1d same-session lookbacks:

- **15m close fill:** that 15m finished; you can work the close once you see it. This is the 15m L3 keeper, not the 1d overlay.
- **15m next-mid:** you confirmed at T's close (which is T+1's open) and assume you get the next 15m's midpoint. Cancels when there is no next RTH bar.
- **1d next-open:** EOD close confirms; buy the **next session open** (MOO). Occupancy and ATR/trail start on that next daily bar. Research only; lost the gate.
- **1d `--intraday-trigger close-cross`:** first 15m **close** above the daily rail (no EOD wait). Fill follows `--realistic-fill-mode` on that 15m. Research only; **no promote.**

---

## How a 1d strategy prices from 15m bars

This join exists to **measure** the lookback. It is **not** a live path. Do not use 1d `signal-close` / `next-mid` / `open-cross` (no `--intraday-trigger`) as strategy candidates.

The daily book still **scans daily OHLCV** (Alpaca 1d, IB prefix for history). When `--realistic-fill` is on and `--timeframe 1d`, modes other than **`next-open`** also load **IB 15m** (`load_ohlcv_many(..., timeframe="15m", provider="IB")`). Names with no IB 15m are skipped (no prior-day fallback). **`next-open`** uses the next daily bar's **open** and keeps the full daily universe.

Per daily signal:

1. Take the detector fill price **X** (the unrealistic rail clip above) and the session date of that daily bar.
2. Keep only RTH 15m starts that session (`09:30` through `15:45` ET, weekdays).
3. Reprice:

**`signal-close`** — walk 15m until `[low, high]` contains X. Fill = that bar's **close**. **Avoid.**

**`next-mid`** — same first print of X. If the next same-session 15m exists and is not wild, fill = `(X + next_mid) / 2`. A 15:45 print of X has no window and is dropped. **Avoid.**

**`open-cross`** — ignore X. Walk until **open > resist** (resist from the daily rails at that session). Fill = that 15m **close**. Gap-through days that never open above resist are skipped. **Avoid** unless `--intraday-trigger` removed the EOD gate (still not promoted).

**`next-open`** — skip 15m. Fill = next daily session **open**. `buy_date` is that next session. Skip if there is no next bar.

The 1d signal-close HTML does **not** record `buy_time` / `sell_time` (`exec_fill_daily_with_15m` returns a price only). List of trades is `buy_date` / `sell_date`. CTF stamps midnight UTC on the daily candle. Exits still walk **daily** highs/lows/closes and skip the fill-day stop.

Same RDWR day: 09:30 ET 15m O **24.37** is skipped (`open-cross` requires open already above resist). First open above is 09:45 ET O 25.19; fill that bar's close **25.71**. That price is what you could have worked **if** you were buying on the 15m clock without waiting for the daily close (`--intraday-trigger`). After an EOD daily confirm it is still a lookback. The clip HTML CTF BUY 24.64 is the rail on the daily wick.

Do not convert the daily index to New York for the 15m join. Pass the calendar date (`YYYY-MM-DD` from the daily stamp at midnight).

---

## 15m strategies

The strategy **is** 15m. No extra join.

- Detector fires on completed bar **T**.
- **`signal-close`:** stay on T, fill = T **close**. BUY sits on the tag/break candle.
- **`next-mid`:** fill on T+1 at mid. The 15:45–16:00 bar cannot fill (confirm at 16:00 has no RTH window; do not roll overnight).
- **`next-open`:** fill on T+1 at that 15m **open**. Same last-RTH cancel.

`next-mid` can print a BUY **mid-channel** that never tagged the rail (GOOGL 2025-11-18: tag at 10:00 ET close 280.79, next mid 281.63). That is why 15m L3 `current_best` uses **signal-close**, not next-mid. Do not copy that 15m keeper onto **1d** `signal-close`: the 1d overlay still keys off the daily rail clip after EOD and is a lookback (**n=2794 E +2.48 PF 1.99** — avoid).

Optional `--max-chase-pct` (off by default) caps how far the exec price may run vs the signal. `--max-low-to-mid-pct` (default 0.5%) applies to **next-mid** only.

---

## Realistic sells (1d last-15m book)

Daily occupancy fills the ATR/trail **stop on the same bar whose low tagged it**. After a 16:00 ET last-15m buy you cannot sell that stop on the next daily candle. Code: `utils/research/realistic_exits.py`. Overlay: `scripts/research/compare_1d_last_15m_realistic_sells.py` (occupancy **not** re-walked; hard-stop is daily ATR k=2 clamp 1.5%–6% + 10% trail).

| Mode | Decision | Fill | Last-15m unique (gross / fric 0.25) |
|------|----------|------|-------------------------------------|
| Kept daily occupancy (first HTML) | Next daily bar low vs stop | Stop price on that daily bar (clip) | n=2779 E +0.96 PF 1.29 / E +0.71 PF 1.21 |
| **`15m-next-mid`** | RTH 15m N low vs stop (bar close) | Next RTH 15m **mid** (overnight OK) | n=2779 E +0.35 PF 1.11 / E +0.10 PF 1.03 |
| **`daily-close-next-open-mid`** | Session high/low vs stop at 16:00 ET | Next session **09:30 ET 15m mid** | n=2778 E +0.47 PF 1.14 / E +0.22 PF 1.06 |

WVE 2023-12-06 last-15m mid **6.85**: clip sell next day **6.0254** (−12%); 15m N+1 mid **4.80** @ 09:45 ET (−30%); EOD then next 09:30 mid **4.65** (−32%). 2022-23 fails on both realistic books. **Do not promote** vs hot-cross n=4079 E −0.19 PF 0.94. HTML: `reports/ascending_channels/2026-09-11/channel_touch_tv_report_interactive_fric0.25_1d_h2_last_15m_open_mid_sell_15m_next_mid_span365_20260911_142523.html` and `…sell_eod_next_open_mid…_20260911_142526.html`. Write-up: [realistic sells](../status_log/edge_hunt/channel_touch/2026-09-11_channel_touch_last_15m_realistic_sells.md).

The entry bar/session is skipped (`skip_entry_bar_stop`). A 15:45 last-RTH buy starts the stop walk on the next session.

---

## What is live vs research

| Book | Fill | Use? |
|------|------|------|
| Nightly 1d H2 (`channel_touch_nightly`) | Unrealistic **daily rail clip** (not this purchaser) | Watchlist only. Not a live fill. |
| `current_best/1d_hot_cross.html` / `1d_channel_touch.html` | Buy-now hot-cross lerp85 unique **n=4079 E −0.19 PF 0.94**. Live-executable. **The 1d number to beat.** | Causal baseline. **Do not promote.** |
| `1d_unrealistic/1d_h2_signal_close.html` | 1d `signal-close` lookback unique **n=2794 E +2.48 PF 1.99** | **Avoid.** |
| `1d_unrealistic/1d_channel_touch.html` | Clip, or next-mid overlay still keyed off that clip (HTML n=1239 **pre-`_as_session_date`**; honest rebuild unique **n=1986 E +2.24 PF 1.89**) | **Avoid** as a strategy. |
| `current_best/15m_channel_touch.html` (L3 wait-12) | `--realistic-fill` **signal-close** (live-executable completed 15m) | 15m keeper. |
| 1d `open-cross` without `--intraday-trigger` | Same-session lookback | **Avoid.** |
| 1d `next-open` | EOD then MOO | Live-executable; **not** promoted. |
| 1d `next-open-mid` | EOD then next session 09:30 ET 15m mid | Live-executable; **not** promoted until it beats hot-cross. |
| 1d last-15m-open-mid + kept daily sells | Last RTH 15m mid if open > rail; **sell is the daily stop clip** | **Avoid** the kept-sell HTML. |
| 1d last-15m + `15m-next-mid` / `daily-close-next-open-mid` sells | Same buy; realistic delayed sells n=2779/2778 E +0.35/+0.47 | Research only. **No promote.** |

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

python scripts\research\backtest_channel_touch_h2_break.py --all-symbols --realistic-fill --realistic-fill-mode next-open --shakeout-breakout --workers 4 --load-workers 8

python scripts\research\compare_1d_next_open_mid.py

python scripts\research\compare_1d_last_15m_realistic_sells.py

python scripts\research\backtest_channel_touch_trades.py --preset 15m --n-symbols 300 --entry-mode l3_touch --min-l3-wait-bars 12 --realistic-fill --touch-error-pct 0 --workers 4 --load-workers 8
```

Do **not** run 1d `--realistic-fill-mode signal-close` / `next-mid` / `open-cross` without `--intraday-trigger` as a strategy hunt. Those commands still exist on the backtester for diagnostics; their E/PF is not a candidate. The clip-only command (`--shakeout-breakout --touch-error-pct 0` with no `--realistic-fill`) is the nightly visualization, not a live book.

---

## Related

- Status notes: [hot-cross in current_best](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_current_best_hot_cross.md), [1d clip out of current_best](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_1d_out_of_current_best.md), [realistic fill rescan](../status_log/edge_hunt/channel_touch/2026-09-04_channel_touch_realistic_fill.md), [signal-close](../status_log/edge_hunt/channel_touch/2026-09-04_channel_touch_signal_close.md), [open-cross](../status_log/edge_hunt/channel_touch/2026-09-05_channel_touch_open_cross.md), [next-open A/B](../status_log/edge_hunt/channel_touch/2026-09-05_channel_touch_next_open_ab.md), [hot-cross](../status_log/edge_hunt/channel_touch/2026-09-06_channel_touch_1d_15m_trigger.md)
- Live 15m monitor (H5 close-confirm, not this purchaser): [channel_touch_15m_live.md](channel_touch_15m_live.md)
