@echo off
REM After US cash close (16:30 ET): catch up 15m, then the 1d prefix to 2006.
REM Friday 16:30 runs through the weekend until Monday 09:15 ET.
REM
REM 15m first so production books have bars. 5m universe fill is paused until
REM the 1d prefix is done. Client 8822 then 8827. Live 15m during RTH is 8826.

cd /d %~dp0..
if not exist logs\data mkdir logs\data

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

REM Keep the 5m stop file in place so a leftover watchdog cannot resume 5m.
if not exist logs\data\ib_5m_universe.stop echo paused for 1d prefix %DATE% %TIME% > logs\data\ib_5m_universe.stop
if exist logs\data\ib_1d_prefix.stop del /f /q logs\data\ib_1d_prefix.stop

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

echo === 1d prefix to 2006 (client 8827, until next 09:15 ET) === >> "%logfile%"
call crons\backfill_ib_1d_prefix.bat >> "%logfile%" 2>&1
set RC1D=%ERRORLEVEL%
echo 1d EXIT_CODE=%RC1D% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

if %RC1D% NEQ 0 exit /b %RC1D%
exit /b %RC15%
