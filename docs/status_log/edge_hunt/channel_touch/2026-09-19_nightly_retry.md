# Nightly Alpaca 1d fail retries + Saturday catch-up

Date: **2026-09-19**

`/hot` Alpaca 1d was **Behind** Saturday morning: as_of **2026-09-17**, expected last close **2026-09-18**. Friday session should have been ingested at **17:00 ET Friday**.

## Cause

CronRunner `channel_touch_nightly` did fire Friday 17:00 ET (`last_fired=2026-09-18T17:00`, ~6 min, **exit 0**). Scanner log `as_of=2026-09-17`. Same settle lag as [2026-09-13](2026-09-13_nightly_1700_et.md): Alpaca daily still the prior session, but the job succeeded so `skip_if_ran_today` skipped any later tick. Weekend schedule is Mon–Fri only, so `/hot` sat Behind until Monday 17:00 ET.

Thursday’s 17:00 ET run had the same lag (`as_of=2026-09-16`).

## Fix

- Nightly **exits 1** when stored ALPACA 1d as_of is before `last_completed_session_date` (Friday close is required at/after Friday 16:00 ET, including Saturday).
- Crontab `retry_hours: [1, 2, 3]`: after a failed 17:00 ET run, CronRunner re-fires +1h/+2h/+3h. Success clears retries (`skip_if_ran_today`).
- `/hot` next-run uses those retry slots while the job is still failed.
- Saturday catch-up: `python scripts\pipeline\cron_manager.py run channel_touch_nightly` (bypasses the weekday schedule).
