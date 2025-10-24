@echo off
REM Update Data and Scan Batch File
REM Fetches latest data before running scanners

echo Starting Update and Scan Workflow...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run update and scan
python update_and_scan.py --scanner hl_after_ll --provider ALPACA --days-back 30

REM Keep window open to see results
pause
