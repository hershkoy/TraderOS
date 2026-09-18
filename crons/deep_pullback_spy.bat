@echo off
REM Deep Pullback SPY: refresh /hot timeframe=dp + Telegram on arm/exit.
REM Uses TimescaleDB SPY 1d + Alpaca last as developing close.

cd /d %~dp0..
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\scanners\deep_pullback_spy_bat_%datetime:~0,8%.log

echo Starting deep_pullback_spy at %DATE% %TIME% >> "%logfile%"

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\scanners\deep_pullback_spy.py %* >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
