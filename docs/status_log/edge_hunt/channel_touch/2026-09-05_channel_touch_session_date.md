# 1d next-mid 15m join — `_as_session_date` (2026-09-05)

RDWR 2025-06-13 in `current_best/1d_channel_touch.html` is a **1d H2** trade. CTF `fill`/`en` are midnight UTC, so a 15m chart snaps the BUY to **09:30 ET**. `enp` **24.64** is not that open (IB 09:30 O **24.37**).

## Bug

`--realistic-fill next-mid` joins that session's IB 15m. The daily bar is keyed at **00:00** (naive or UTC). Converting that stamp to America/New_York turns Monday midnight into **Sunday evening**, so:

- Every **Monday** fill missed the session's 15m (no RTH bars on Sunday).
- **Tue–Fri** used the **prior session's** 15m.

Nightly close-fills never used this path.

Helper: `_as_session_date` in `utils/research/realistic_purchaser.py`. Pass the calendar `YYYY-MM-DD`; do not convert midnight UTC to New York. Playbook: [realistic purchasing](../../../features/realistic_purchasing.md).

## RDWR 2025-06-13

Alpaca 1d: O=L=**24.65**, H=26.93, C=**26.58** (gapped through resist ~24.47). Unrealistic same-bar clip is the daily low / rail (**24.64** on the wick).

Pre-fix next-mid converted 2025-06-13T00:00Z to **June 12** evening, so it priced off Thursday 15m: 24.65 first prints in the 10:15 bar; next bar 10:30 mid **24.63**; blend `(24.65 + 24.63) / 2 = 24.64`. That is the HTML `enp`.

Honest June 13 next-mid would see 09:30 (24.35–25.29) then 09:45 (25.06–25.83) and **cancel as wild** (`low-to-mid` ~1.5% vs 0.5% cap). This name should not stay in a rebuilt next-mid book.

`open-cross` on the real session: skip 09:30 O 24.37; first open above resist is 09:45; fill that close **25.71**.

## Live clocks (not this purchaser)

- **/hot hot:** Alpaca last vs resist, including 09:30. Crossing the rail is **not** a BUY.
- **15m fills:** completed 15m only (09:30 bar eligible at 09:45).
- **1d nightly:** post-close. June 12 close 24.42 was still below resist; fill only after June 13 close 26.58.

June 12: armed all day; hot most of the session; **not** hot at the 15:45 close (~0.17% below).

## Numbers that are pre-fix

Superseded 2026-09-06: honest unique next-mid + shakeout is **n=1986 E +2.24 PF 1.89**. See [rebuild](2026-09-06_channel_touch_next_mid_session_date.md).

Until that rescan, these were the stale HTML / crowding figures:

| Book | n | E | PF |
|------|---|----|----|
| Unique next-mid + shakeout (`1d_unrealistic` HTML) | 1239 | +2.27 | 1.92 |
| Cap 2/day by wait (crowding) | 933 | +3.01 | 2.25 |

## Decision

Code join is fixed. **2026-09-06 rescan:** unique **n=1986 E +2.24 PF 1.89** (all year buckets +). Nightly stays **unrealistic close / rail-clip** fills (never this join). HTML in `1d_unrealistic/` was not regenerated.
