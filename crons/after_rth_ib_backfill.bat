@echo off
REM After US cash close (16:30 ET): catch up IB 15m, then exit so the 5m
REM watchdog can take Gateway (client 8823). Live 15m during RTH is 8826.
REM
REM 15m first so production books have bars. Client 8822. Do not plant the 5m
REM stop file and do not start the 1d prefix (8827) in this window.

cd /d %~dp0..
if not exist logs\data mkdir logs\data

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set datetime=%%I
set logfile=logs\data\after_rth_ib_backfill_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting after-RTH IB backfill at %DATE% %TIME% > "%logfile%"
echo Log: %logfile%
echo.

call venv\Scripts\activate.bat

echo === 15m universe catch-up (client 8822) === >> "%logfile%"
venv\Scripts\python.exe scripts\data\backfill_ib_15m_universe.py --sleep 1 --ib-client-id 8822 --reset-resume --fresh-hours 12 >> "%logfile%" 2>&1
set RC15=%ERRORLEVEL%
echo 15m EXIT_CODE=%RC15% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

exit /b %RC15%
