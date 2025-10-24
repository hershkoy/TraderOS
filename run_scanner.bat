@echo off
REM Unified Scanner Runner Batch File
REM Supports multiple scanner types

echo Starting Unified Scanner Runner...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the scanner with HL After LL scanner by default
python scanner_runner.py --scanner hl_after_ll --log-level INFO

REM Keep window open to see results
pause
