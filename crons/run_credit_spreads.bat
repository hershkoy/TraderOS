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

REM Run scripts and log output (both stdout and stderr)
echo Running QQQ spread...
python scripts/spreads_trader.py --symbol QQQ --dte 7 --create-orders-en --quantity 2 --risk-profile balanced >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: QQQ spread failed with exit code %ERRORLEVEL%
) else (
    echo QQQ spread completed successfully.
)

echo.
echo Running IWM spread...
python scripts/spreads_trader.py --symbol IWM --dte 7 --create-orders-en --quantity 2 --risk-profile balanced >> %logfile% 2>&1
if errorlevel 1 (
    echo ERROR: IWM spread failed with exit code %ERRORLEVEL%
) else (
    echo IWM spread completed successfully.
)

echo.
echo Completed. Log saved to: %logfile%
echo.
pause