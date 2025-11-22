@echo off
cd /d %~dp0..
.\venv\Scripts\activate

REM Create log file with timestamp (YYYYMMDD_HHMMSS format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\credit_spreads_%datetime:~0,8%_%datetime:~8,6%.log

echo Logging to: %logfile%
echo.

REM Run scripts and log output (both stdout and stderr)
python scripts/spreads_trader.py --symbol QQQ --dte 7 --create-orders-en --quantity 2 --risk-profile balanced >> %logfile% 2>&1
python scripts/spreads_trader.py --symbol IWM --dte 7 --create-orders-en --quantity 2 --risk-profile balanced >> %logfile% 2>&1

echo.
echo Completed. Log saved to: %logfile%