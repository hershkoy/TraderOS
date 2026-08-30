@echo off
REM Minute-tick Alpaca last-price refresh for 15m channel-touch armed list.
REM Does NOT rebuild rails or treat last price as a fill.
REM Prerequisite: channel_touch_15m --mode watchlist/run has populated TimescaleDB.

cd /d %~dp0..
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\scanners\channel_touch_15m_proximity_bat_%datetime:~0,8%.log

echo Starting channel-touch 15m proximity at %DATE% %TIME% >> "%logfile%"

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\scanners\channel_touch_15m.py --mode proximity %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
