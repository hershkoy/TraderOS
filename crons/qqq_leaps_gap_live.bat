@echo off
REM QQQ LEAPS gap: refresh /hot timeframe=lg + Telegram on approach/arm.
REM TimescaleDB QQQ 1d + Alpaca last/open for developing gap.

cd /d %~dp0..
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set datetime=%%I
set logfile=logs\scanners\qqq_leaps_gap_live_bat_%datetime:~0,8%.log

echo Starting qqq_leaps_gap_live at %DATE% %TIME% >> "%logfile%"

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\scanners\qqq_leaps_gap_live.py %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
