@echo off
REM HL After LL Scanner Batch Runner
REM Scans for LL -> HH -> HL patterns in weekly data

echo Starting HL After LL Scanner...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the scanner
python hl_after_ll_scanner_runner.py

REM Keep window open to see results
pause
