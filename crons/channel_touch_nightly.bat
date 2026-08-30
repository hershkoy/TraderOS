@echo off
REM Nightly channel-touch: update ALPACA 1d -> H2 resist-break scan -> Telegram
REM Defaults: min-wait 6, span 365, unique-symbol/day (no RS cap), ATR k=2; no RSI / in-channel / beyond-width
REM Rollback L3: add --no-h2-resist-break --require-in-channel --max-rsi 50 --max-beyond-width 0.25
REM Schedule via: crons\install_channel_touch_nightly_task.bat

cd /d %~dp0..
if not exist logs\scanners mkdir logs\scanners

set PYTHONPATH=.
set PYTHONUNBUFFERED=1

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\scanners\channel_touch_nightly_bat_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting channel-touch nightly at %DATE% %TIME% > "%logfile%"
echo Log: %logfile%
echo.

call venv\Scripts\activate.bat
venv\Scripts\python.exe scripts\scanners\channel_touch_nightly.py --multi-symbol --batch-size 100 --delay-tickers 0 --delay-batches 0.25 >> "%logfile%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo EXIT_CODE=%EXITCODE% >> "%logfile%"
echo Finished at %DATE% %TIME% >> "%logfile%"

exit /b %EXITCODE%
