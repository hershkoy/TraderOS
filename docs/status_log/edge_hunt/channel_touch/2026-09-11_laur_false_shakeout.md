# LAUR 2021-10-28 fill was a false shakeout

Pasted CTF from `…last_15m_open_mid_sell_15m_next_mid_span365_20260911_142523.html`:
L1 2020-09-08 / L2 2021-03-04 / H2 2021-06-11 / fill 2021-10-28 16.83.

On those rails the stock had already closed above resistance (strict first close 2021-08-13; first close clearing the 1.2% detector band 2021-09-10). Oct 27 close 16.60 was still **above** the rail (~+1.0%) but inside `error_pct` 1.2%, so `_shakeout_breakout_fill` counted it as "back inside" and bought Oct 28.

A sibling pair (same H2, L2 2021-02-01) filled 2021-09-09 and occupancy skipped the Mar-4 first break.

Fixes (detector v1 `find_channels` unchanged):
- Shakeout "inside" is close at/below the rail, not "failed the 1.2% breakout band".
- `find_h2_l3_setups` keeps one L1-L2 pair per H2 bar (latest L2, then earliest L1).

Rebuilt 2026-09-11 (wall-clock ~40 min: unique scan 1831s cache-hot daily + 15m purchase 238s). Unique signal-close span365 **n=2310 E +2.50 PF 2.03** (was 2794 / +2.48 / 1.99). Last-15m + 15m N+1 mid **n=2297 E +0.27 PF 1.09** (fric0.25 E +0.02 PF 1.01).

LAUR 2021 on the pasted rails is now **2021-09-10** last-15m mid **16.82**, sell 2021-10-12 14:15 ET **17.155**. Oct 28 is gone.

HTML: `reports/ascending_channels/2026-09-11/channel_touch_tv_report_interactive_fric0.25_1d_h2_last_15m_open_mid_sell_15m_next_mid_span365_20260911_165800.html`
