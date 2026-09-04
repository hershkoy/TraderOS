@echo off
REM After US cash close (16:30 ET): catch up 15m (and 1d already ran at nightly),
REM then start the long 5m historical backfill until next RTH.
REM Friday 16:30 runs through the weekend until Monday 09:15 ET.
REM
REM 15m first so production 15m/1d books have bars. 5m is research fill data.
REM Client 8822 then 8823. Live 15m during RTH is 8826.

cd /d %~dp0..
if not exist logs\data mkdir logs\data

set PYTHONPATH=.
set PYTHONUNBUFFERED=1
set IB_PORT=4001

if exist logs\data\ib_5m_universe.stop del /f /q logs\data\ib_5m_universe.stop

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\data\after_rth_ib_backfill_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting after-RTH IB backfill at %DATE% %TIME% > "%logfile%"
echo Log: %logfile%
echo.

call venv\Scripts\activate.bat

echo === 15m universe catch-up (client 8822) === >> "%logfile%"
venv\Scripts\python.exe scripts\data\backfill_ib_15m_universe.py --sleep 1 --ib-client-id 8822 --reset-resume --fresh-hours 12 >> "%logfile%" 2>&1
set RC15=%ERRORLEVEL%
echo 15m EXIT_CODE=%RC15% >> "%logfile%"

echo === 5m universe (client 8823, until next 09:15 ET) === >> "%logfile%"
venv\Scripts\python.exe scripts\data\backfill_ib_5m_universe.py --sleep 1 --ib-client-id 8823 --ib-port 4001 >> "%logfile%" 2>&1
set RC5=%ERRORLEVEL%
echo 5m EXIT_CODE=%RC5% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

if %RC5% NEQ 0 exit /b %RC5%
exit /b %RC15%
