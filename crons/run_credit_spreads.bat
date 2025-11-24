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
python scripts/spreads_trader.py --conf-file daily_spreads.yaml >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: Spreads trader failed with exit code %ERRORLEVEL%
) else (
    echo Spreads trader completed successfully.
)

echo.
echo Completed. Log saved to: %logfile%
echo.
pause