@echo off
REM 15m channel-touch research loop (H5 + overshoot 0.08 + vol>=2).
REM Does NOT replace daily nightly H2 resist-break.
REM Prerequisite: IB 15m must be current (crons\backfill_ib_15m_universe.bat).
REM Schedule every 15m in RTH only after backfill is done.

cd /d %~dp0..
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\scanners\channel_touch_15m_bat_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting channel-touch 15m at %DATE% %TIME% > "%logfile%"
echo Log: %logfile%
echo.

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\scanners\channel_touch_15m.py --mode run %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
