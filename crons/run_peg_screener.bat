@echo off
chcp 65001 >nul
cd /d %~dp0..
call .\venv\Scripts\activate.bat

REM Create log file with timestamp (YYYYMMDD_HHMMSS format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\screeners\peg\peg_screener_%datetime:~0,8%_%datetime:~8,6%.log

echo ========================================
echo PEG Screener
echo ========================================
echo Starting PEG Screener...
echo Logging to: %logfile%
echo Current directory: %CD%
echo.

REM Check if FINNHUB_API_KEY is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); exit(0 if os.getenv('FINNHUB_API_KEY') else 1)" 2>nul
if errorlevel 1 (
    echo WARNING: FINNHUB_API_KEY not found in environment
    echo Make sure it's set in your .env file or environment variables
    echo.
)

REM Run the PEG screener with output report enabled
python scripts\scanners\peg_screener.py --output-report >> %logfile% 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: PEG screener failed with exit code %ERRORLEVEL%
    echo Check the log file for detailed error messages: %logfile%
) else (
    echo.
    echo PEG screener completed successfully.
)

echo.
echo Completed. Log saved to: %logfile%
echo Results saved to reports/peg_screener_*.csv
echo.
pause

