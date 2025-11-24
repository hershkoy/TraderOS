@echo off
cd /d %~dp0..
call .\venv\Scripts\activate.bat

REM Unified Scanner Runner Batch File
REM Supports multiple scanner types

echo Starting Unified Scanner Runner...
echo Current directory: %CD%
echo Python path: 
where python
echo.

REM Run the scanner with HL After LL scanner by default
python scanner_runner.py --scanner hl_after_ll squeeze --log-level INFO
if errorlevel 1 (
    echo.
    echo ERROR: Scanner failed with exit code %ERRORLEVEL%
    echo Check the logs directory for detailed error messages.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

REM Keep window open to see results
echo.
echo Scanner completed successfully.
pause
