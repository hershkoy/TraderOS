# Cron manager (hidden Windows minute tick)

One Windows Task Scheduler job runs **every minute in the background** (`pythonw`, no CMD window). It reads `crons/crontab.yaml` and starts any enabled job whose cron string matches the current minute.

## Setup (once)

```bat
venv\Scripts\activate
set PYTHONPATH=.
python scripts\pipeline\cron_manager.py install-task
```

Or: `crons\install_cron_runner_task.bat`

Task name: `backTraderTest\CronRunner`. Hidden, Interactive (your logged-on user), repeats every minute. Jobs themselves also spawn with `CREATE_NO_WINDOW` so bats do not flash a console.

```bat
python scripts\pipeline\cron_manager.py task-status
python scripts\pipeline\cron_manager.py uninstall-task
```

If you already have `backTraderTest\ChannelTouchNightly`, delete or disable it before enabling `channel_touch_nightly` in the crontab so it does not double-run.

## Crontab

Jobs live in `crons/crontab.yaml`:

```yaml
timezone: local
jobs:
  - name: channel_touch_nightly
    enabled: true
    schedule: "0 23 * * 1-5"
    command: crons\channel_touch_nightly.bat
    description: Daily H2 resist-break
```

Cron is five fields: `minute hour day-of-month month day-of-week` (0 or 7 = Sunday). A job may use a list of cron strings (fires if any match) and an optional IANA `timezone` (for example `America/New_York`).

## Manager CLI

```bat
venv\Scripts\activate
set PYTHONPATH=.

python scripts\pipeline\cron_manager.py list
python scripts\pipeline\cron_manager.py add my_job --schedule "0 23 * * 1-5" --command "crons\foo.bat"
python scripts\pipeline\cron_manager.py add rs_scan --schedule "*/15 10-16 * * 1-5" --command "python scripts\scanners\foo.py" --timezone America/New_York
python scripts\pipeline\cron_manager.py enable my_job
python scripts\pipeline\cron_manager.py disable my_job
python scripts\pipeline\cron_manager.py remove my_job
python scripts\pipeline\cron_manager.py next
python scripts\pipeline\cron_manager.py run my_job
python scripts\pipeline\cron_manager.py tick --dry-run
python scripts\pipeline\cron_manager.py tick --at 2026-08-31T23:00
```

`tick` is what Task Scheduler calls. Use `--dry-run` / `--at` to see what would fire. `run` ignores the schedule and starts one job now.

Seeded jobs in `crontab.yaml` start **disabled**. Enable the ones you want.

The charting dashboard is **not** a crontab job. Install the always-on logon task with `python scripts\pipeline\charting_server_service.py install-task` (task name `backTraderTest\ChartingServer`).

## Behaviour

- Same-minute re-entry is skipped (state in `logs/cron/state.json`).
- `skip_if_ran_today: true` skips a later tick if that job already **succeeded** today (used on `channel_touch_nightly` so a manual run does not double-fire at 23:00). Failed runs are retried.
- If a job is still running, the next due tick skips it (lock in `logs/cron/locks/`).
- Job stdout/stderr: `logs/cron/{name}_{timestamp}.log`
- Tick log: `logs/cron/tick_YYYYMMDD.log` (only when something is due, plus errors)

Do not put `pause` in a bat you schedule here (`daily_scanner.bat` and `update_and_scan.bat` wait for a key).
