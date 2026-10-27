@echo off
REM Cooperative stop for the overnight IB fill, then force-kill if it is still up.
REM 2026-10-26: the overnight slot is the 1d prefix (8827). The 5m stop file
REM stays armed so that fill does not resume. Used at 09:15 ET Mon-Fri so live
REM 15m (client 8826) owns Gateway.

cd /d %~dp0..
if not exist logs\data mkdir logs\data

echo stop requested %DATE% %TIME% > logs\data\ib_5m_universe.stop
echo stop requested %DATE% %TIME% > logs\data\ib_1d_prefix.stop
echo Wrote stop files logs\data\ib_5m_universe.stop and logs\data\ib_1d_prefix.stop

set PYTHONPATH=.
call venv\Scripts\activate.bat

REM Give the fill time to finish the current IB request and disconnect.
REM timeout.exe exits immediately when stdin is redirected (CronRunner).
powershell -NoProfile -Command "Start-Sleep -Seconds 20"

venv\Scripts\python.exe scripts\pipeline\cron_manager.py stop after_rth_ib_backfill --timeout 30
venv\Scripts\python.exe scripts\pipeline\cron_manager.py stop backfill_ib_5m_universe --timeout 15
venv\Scripts\python.exe scripts\pipeline\cron_manager.py stop backfill_ib_1d_prefix --timeout 15

exit /b 0
