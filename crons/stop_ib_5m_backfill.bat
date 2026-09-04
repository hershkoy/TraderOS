@echo off
REM Cooperative stop for the IB 5m backfill, then force-kill if it is still up.
REM Writes logs\data\ib_5m_universe.stop so the Python loop exits after the
REM current IB window (DB last_ts is the resume cursor).
REM Used at 09:15 ET Mon-Fri so live 15m (client 8826) owns Gateway.

cd /d %~dp0..
if not exist logs\data mkdir logs\data

echo stop requested %DATE% %TIME% > logs\data\ib_5m_universe.stop
echo Wrote stop file logs\data\ib_5m_universe.stop

set PYTHONPATH=.
call venv\Scripts\activate.bat

REM Give the 5m loop time to finish the current IB window and disconnect.
timeout /t 45 /nobreak >nul

venv\Scripts\python.exe scripts\pipeline\cron_manager.py stop after_rth_ib_backfill --timeout 30
venv\Scripts\python.exe scripts\pipeline\cron_manager.py stop backfill_ib_5m_universe --timeout 15

exit /b 0
