@echo off
REM 15m channel-touch research loop (H5 + overshoot 0.08 + vol>=2).
REM Does NOT replace daily nightly H2 resist-break.
REM Prerequisite: IB Gateway (4001/4002) or TWS (7496/7497) on localhost.
REM Auto-detects the API port (IB_PORT / last-good cache / defaults). Overnight
REM universe backfill (8822) seeds the watchlist; this job pulls IB 15m on the
REM armed/hot list (client 8826) at each RTH bar close so H5 BUY NOW can fire.

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
