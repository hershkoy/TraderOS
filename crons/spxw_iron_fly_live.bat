@echo off
REM SPXW 0DTE iron fly: refresh /hot timeframe=if + Telegram (L5/S3/P2).
REM Dense around 13:30-14:00 ET entry window.

cd /d %~dp0..
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set datetime=%%I
set logfile=logs\scanners\spxw_iron_fly_live_bat_%datetime:~0,8%.log

echo Starting spxw_iron_fly_live at %DATE% %TIME% >> "%logfile%"

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\scanners\spxw_iron_fly_live.py %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
