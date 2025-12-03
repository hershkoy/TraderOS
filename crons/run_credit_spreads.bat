@echo off
chcp 65001 >nul
cd /d %~dp0..
call .\venv\Scripts\activate.bat

REM Create log file with timestamp (YYYYMMDD_HHMMSS format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\credit_spreads_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting Credit Spreads Trader...
echo Logging to: %logfile%
echo Current directory: %CD%
echo.

REM Run script with config file (processes all orders in parallel)
echo Running spreads from daily_spreads.yaml...

REM Check if live-en argument is provided
set live_arg=
if "%1"=="live-en" set live_arg=--live-en

python scripts/trading/options_strategy_trader.py --conf-file crons\daily_spreads.yaml %live_arg% >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: Spreads trader failed with exit code %ERRORLEVEL%
) else (
    echo Spreads trader completed successfully.
)

echo.
echo Completed. Log saved to: %logfile%
echo.
pause