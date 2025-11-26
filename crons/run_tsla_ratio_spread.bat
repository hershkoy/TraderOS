@echo off
chcp 65001 >nul
cd /d %~dp0..
call .\venv\Scripts\activate.bat

REM Create log file with timestamp (YYYYMMDD_HHMMSS format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\tsla_ratio_spread_%datetime:~0,8%_%datetime:~8,6%.log

echo Starting TSLA Ratio Spread Trader...
echo Logging to: %logfile%
echo Current directory: %CD%
echo.

REM Run TSLA vertical spread with hedging (1x2 ratio spread)
echo Running TSLA vertical_spread_with_hedging...
python scripts/options_strategy_trader.py --symbol TSLA --strategy vertical_spread_with_hedging --dte 14 --create-orders-en --quantity 1 --risk-profile balanced >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: TSLA ratio spread trader failed with exit code %ERRORLEVEL%
) else (
    echo TSLA ratio spread trader completed successfully.
)

echo.
echo Completed. Log saved to: %logfile%
echo.
pause

